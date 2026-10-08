# =====================================================================
# agent_core.py - 多 Agent 架构（RAG + 持仓管家）
# RAG 优化：多路改写 + 混合检索（BM25+向量）+ 父文档检索 + Rerank 精排
# 数据源：腾讯（实时价、前复权K线）
# 优化：腾讯行情本地缓存（30秒），防止限流
# =====================================================================

# ------------------------- 基础库导入 ---------------------------------
import os                                                   # 操作系统接口，用于读取环境变量
import json                                                 # JSON 库，用于解析配置和数据
import sqlite3                                              # SQLite 数据库，用于本地存储
import requests                                             # HTTP 请求库，用于调用腾讯接口
import time                                                 # 时间库，用于实现缓存超时控制
import jieba                                                # 中文分词库，用于 BM25 关键词切词
from datetime import datetime, timedelta                    # 日期时间库，用于记录时间戳

# ------------------------- 环境变量 -----------------------------------
from dotenv import load_dotenv                              # 加载 .env 文件的环境变量

# ------------------------- LangChain 核心 -----------------------------
from langchain.agents import create_agent                   # 创建 Agent 的工厂函数
from langchain_core.tools import tool                       # 工具装饰器，把普通函数变成 Agent 可调用的工具
from langchain_core.messages import HumanMessage, AIMessage # 消息类，用于构建对话历史

# ------------------------- 向量与检索 ---------------------------------
from langchain_community.embeddings import DashScopeEmbeddings # 阿里云百炼向量模型
from langchain_community.vectorstores import FAISS          # FAISS 向量数据库
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter # 文本切分器
from langchain_community.retrievers import ParentDocumentRetriever # 父文档检索器
from langchain.storage import InMemoryStore                 # 内存存储，用于存放父文档
from langchain_community.retrievers import BM25Retriever    # BM25 关键词检索器

# ------------------------- 多 Agent 编排 ------------------------------
from langgraph_supervisor import create_supervisor          # Supervisor 多 Agent 编排器
from langchain_openai import ChatOpenAI                     # OpenAI 兼容模型客户端

# ------------------------- 金融数据与绘图 -----------------------------
import akshare as ak                                        # AKShare 财经数据库，用于获取基本面数据
import mplfinance as mpf                                    # mplfinance，用于绘制 K 线图
import pandas as pd                                         # pandas，用于数据处理

# ------------------------- 百炼 SDK 与大模型客户端 ---------------------
import dashscope                                            # dashscope SDK
from dashscope import TextReRank                            # 百炼的重排序模型
from openai import OpenAI                                   # OpenAI 客户端，用于调用大模型

# =====================================================================
# ==================== 环境变量配置 ==================================
# =====================================================================
load_dotenv()                                                                        # 读取当前目录下的 .env 文件
os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "")                    # 将百炼 Key 映射给 OpenAI 接口变量
os.environ["OPENAI_BASE_URL"] = os.getenv("DATA_BASE_URL", "")                       # 将百炼 Base URL 映射给 OpenAI 接口变量
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")                                   # 单独设置 dashscope 的 API Key

DB_PATH = "chat_memory.db"                                                           # 本地 SQLite 数据库路径
MAX_HISTORY_MESSAGES = 20                                                            # 滑动窗口大小，最多保留 20 条历史

# =====================================================================
# ==================== 腾讯行情本地缓存 ==============================
# =====================================================================
_tencent_cache = {}                                                                  # 全局缓存字典：{股票代码: (时间戳, 数据字典)}

def _fetch_tencent_stock_info(code: str) -> dict:                                    # 定义真正请求腾讯接口的函数
    """真正去请求腾讯接口的函数"""                                                  # 函数说明
    result = {                                                                       # 初始化结果字典（带默认值）
        "name": code, "price": 0.0, "change_pct": 0.0,                               # 默认名称、价格、涨跌幅
        "pe": "未知", "pb": "未知", "volume": "未知", "amount": "未知"                # 默认市盈率、市净率、成交量、成交额
    }                                                                                # 字典初始化结束
    try:                                                                             # 尝试执行网络请求
        prefix = "sh" if code.startswith("6") else "sz"                              # 判断股票交易所，6开头为沪市，否则为深市
        url = f"http://qt.gtimg.cn/q={prefix}{code}"                                 # 拼接腾讯行情 API URL
        resp = requests.get(url, timeout=5)                                          # 发起 GET 请求，超时时间 5 秒
        resp.encoding = "gbk"                                                        # 设置响应编码为 GBK，防止中文乱码
        if resp.status_code == 200:                                                  # 判断 HTTP 状态码是否为 200（成功）
            parts = resp.text.split("~")                                             # 将返回文本按波浪号 "~" 切分为列表
            if len(parts) > 50:                                                      # 确保返回的数据字段足够多
                result["name"] = parts[1]                                            # 提取股票名称（索引 1）
                result["price"] = float(parts[3])                                    # 提取最新价（索引 3）
                result["change_pct"] = float(parts[32]) if parts[32] else 0.0        # 提取涨跌幅（索引 32），空值兜底
                result["pe"] = parts[39] if parts[39] else "未知"                    # 提取市盈率（索引 39），空值兜底
                result["pb"] = parts[46] if parts[46] else "未知"                    # 提取市净率（索引 46），空值兜底
                result["volume"] = parts[6]                                          # 提取成交量（索引 6）
                result["amount"] = parts[37]                                         # 提取成交额（索引 37）
    except Exception:                                                                # 捕获所有异常
        pass                                                                         # 忽略异常，确保程序不崩溃
    return result                                                                    # 返回结果字典


def _get_cached_stock_info(code: str) -> dict:                                       # 定义带缓存的行情获取函数
    """获取腾讯行情（带 30 秒本地缓存，防止被限流）"""                                # 函数说明
    now = time.time()                                                                # 获取当前时间戳
    if code in _tencent_cache:                                                       # 如果缓存中存在该股票
        ts, data = _tencent_cache[code]                                              # 取出缓存的时间戳和数据
        if now - ts < 30:                                                            # 判断时间戳距离现在是否小于 30 秒
            return data                                                              # 如果在 30 秒内，直接返回缓存数据
    data = _fetch_tencent_stock_info(code)                                           # 如果缓存过期或不存在，重新请求接口
    _tencent_cache[code] = (now, data)                                               # 将新数据写入缓存
    return data                                                                      # 返回最新数据


def _get_realtime_price(code: str) -> float:                                         # 定义获取实时价格的函数
    """拉取股票最新价（带缓存）"""                                                  # 函数说明
    return _get_cached_stock_info(code)["price"]                                     # 从缓存中提取最新价返回


def _get_tencent_stock_info(code: str) -> dict:                                      # 定义获取完整腾讯行情的函数
    """拉取股票完整行情（带缓存）"""                                                  # 函数说明
    return _get_cached_stock_info(code)                                              # 调用缓存逻辑并返回


def _get_tencent_prefix(code: str) -> str:                                           # 定义判断股票前缀的函数
    """判断股票代码是沪市还是深市"""                                                # 函数说明
    return "sh" if code.startswith("6") else "sz"                                    # 返回 sh 或 sz

# =====================================================================
# ==================== 数据库初始化（对话历史） ======================
# =====================================================================
def init_db():                                                                       # 定义初始化对话历史表函数
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)                         # 连接数据库，禁用线程检查
    cursor = conn.cursor()                                                           # 创建游标对象
    cursor.execute("""                                                               # 执行 SQL 语句
        create table if not exists chat_history (                                    # 创建对话历史表
            id integer primary key autoincrement,                                    # 自增主键
            role text not null,                                                      # 角色（user/assistant）
            content text not null,                                                   # 消息内容
            timestamp text not null                                                  # 时间戳
        )                                                                            # 字段定义结束
    """)                                                                             # SQL 语句结束
    conn.commit()                                                                    # 提交事务，保存更改
    conn.close()                                                                     # 关闭数据库连接

def save_message(role, content):                                                     # 定义保存消息到数据库的函数
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)                         # 连接数据库
    cursor = conn.cursor()                                                           # 创建游标
    cursor.execute(                                                                  # 执行插入语句
        "insert into chat_history (role, content, timestamp) values (?, ?, ?)",      # SQL 插入模板
        (role, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))                # 填充参数：角色、内容、当前时间
    )                                                                                # 执行结束
    conn.commit()                                                                    # 提交事务
    conn.close()                                                                     # 关闭连接

def load_messages():                                                                 # 定义加载最近历史消息的函数
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)                         # 连接数据库
    cursor = conn.cursor()                                                           # 创建游标
    cursor.execute(f"select role, content from chat_history order by id desc limit {MAX_HISTORY_MESSAGES}") # 倒序查询最近20条
    rows = cursor.fetchall()                                                         # 获取所有结果行
    conn.close()                                                                     # 关闭连接
    rows = reversed(rows)                                                            # 反转列表，使其按时间正序排列
    history = []                                                                     # 初始化历史消息列表
    for role, content in rows:                                                       # 遍历数据库行
        if role == "user":                                                           # 如果是用户消息
            history.append(HumanMessage(content=content))                            # 转换为 HumanMessage 对象
        else:                                                                        # 如果是 AI 消息
            history.append(AIMessage(content=content))                               # 转换为 AIMessage 对象
    return history                                                                   # 返回历史消息列表

def clear_db():                                                                      # 定义清空对话历史的函数
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)                         # 连接数据库
    cursor = conn.cursor()                                                           # 创建游标
    cursor.execute("delete from chat_history")                                       # 执行删除全部语句
    conn.commit()                                                                    # 提交事务
    conn.close()                                                                     # 关闭连接

init_db()                                                                            # 调用初始化函数，确保表存在

# =====================================================================
# ==================== 持仓表初始化 ==================================
# =====================================================================
def init_portfolio():                                                                # 定义初始化持仓表函数
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)                         # 连接数据库
    cursor = conn.cursor()                                                           # 创建游标
    cursor.execute("""                                                               # 执行建表 SQL
        create table if not exists my_portfolio (                                    # 创建 my_portfolio 表
            code text primary key,                                                   # 股票代码作为主键
            name text not null,                                                      # 股票名称
            shares integer not null,                                                 # 持有股数
            cost real not null,                                                      # 成本价
            buy_date text,                                                           # 买入日期
            note text                                                                # 备注
        )                                                                            # 字段结束
    """)                                                                             # SQL 结束
    conn.commit()                                                                    # 提交事务
    conn.close()                                                                     # 关闭连接
    print("[持仓] 初始化完成")                                                       # 打印初始化完成日志

init_portfolio()                                                                     # 调用初始化函数

# =====================================================================
# ==================== RAG 构建（混合检索 + 父文档检索） =============
# =====================================================================
print("[系统日志] 正在加载 Markdown 文档并构建 RAG 检索系统...")                     # 打印日志
md_path = "data/投资知识库.md"                                                       # 定义知识库文件路径

if not os.path.exists(md_path):                                                      # 判断文件是否存在
    print(f"❌ 找不到文件：{md_path}")                                               # 如果不存在，打印错误
    md_header_splits = []                                                            # 切分结果设为空列表
    parent_splitter = None                                                           # 父块切分器设为空
    child_splitter = None                                                            # 子块切分器设为空
    bm25_retriever = None                                                            # BM25 检索器设为空
else:                                                                                # 如果文件存在
    with open(md_path, "r", encoding="utf-8") as f:                                  # 以只读方式打开文件
        markdown_document = f.read()                                                 # 读取全部内容

    # ---------- 1. 准备两层切分器 ----------
    # 父块：按一级标题切分，保留完整章节，供大模型看
    parent_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "一级标题")]                                      # 只按一级标题切分
    )
    # 子块：按二级和三级标题切分，颗粒度小，供检索匹配
    child_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "二级标题"), ("###", "三级标题")]                 # 按二级、三级标题切分
    )

    # ---------- 2. 初始化向量模型 ----------
    embeddings = DashScopeEmbeddings(                                                # 初始化百炼向量模型
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),                            # 传入 API Key
        model="qwen3.7-text-embedding-flash"                                         # 指定有免费额度的向量模型
    )                                                                                # 初始化结束

    # ---------- 3. 初始化父文档存储器 ----------
    store = InMemoryStore()                                                          # 内存存储，用于存放父块

    # ---------- 4. 初始化空 FAISS 向量库 ----------
    vectorstore = FAISS.from_texts(                                                  # 创建一个带初始占位数据的 FAISS 实例
        ["init"], embeddings, metadatas=[{"source": "init"}]                         # 占位文本，稍后由 Retriever 自动填充
    )

    # ---------- 5. 初始化父文档检索器 ----------
    parent_retriever = ParentDocumentRetriever(                                      # 创建父文档检索器
        vectorstore=vectorstore,                                                     # 挂载向量库
        docstore=store,                                                              # 挂载父文档存储器
        child_splitter=child_splitter,                                               # 指定子块切分器
        parent_splitter=parent_splitter,                                             # 指定父块切分器
    )

    # ---------- 6. 把文档喂给检索器 ----------
    # 这一步会自动完成：切父块 → 存父块 → 切子块 → 子块向量化入库
    from langchain_core.documents import Document                                    # 导入 Document 类
    parent_retriever.add_documents([Document(page_content=markdown_document)])       # 添加文档

    # ---------- 7. 从向量库取出所有子块，用于构建 BM25 ----------
    all_docs = list(vectorstore.docstore._dict.values())                             # 取出向量库中的所有子块
    all_docs = [d for d in all_docs if d.metadata.get("source") != "init"]           # 过滤掉初始化的占位数据

    # ---------- 8. 用 jieba 分词后构建 BM25 检索器 ----------
    bm25_retriever = BM25Retriever.from_documents(                                   # 从文档构建 BM25 检索器
        all_docs,                                                                    # 传入所有子块
        preprocess_func=jieba.lcut,                                                  # 用 jieba 做中文分词
    )
    bm25_retriever.k = 5                                                             # BM25 召回 Top5

    # ---------- 9. 兼容旧代码：暴露 vectorstore 供 rag_eval.py 使用 ----------
    vectorstore = parent_retriever.vectorstore                                       # 指向父文档检索器内部的向量库

    print(f"✅ RAG 检索系统构建完成（父子文档切分 + 混合检索）")                     # 打印构建完成日志

# =====================================================================
# ============ 多路改写 + 混合检索 + Rerank 精排 =====================
# =====================================================================
def generate_multi_queries(original_query: str) -> list:                             # 定义多路改写函数
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=os.getenv("DATA_BASE_URL")) # 初始化 OpenAI 客户端
    prompt = f"""你是一个搜索专家。请把用户的原始问题，改写成 3 个不同角度的问法。
要求：保持原意、不同角度、只输出3个问题每行一个无序号无解释。

原始问题：{original_query}
"""                                                                                  # 构造改写提示词
    try:                                                                             # 尝试执行
        resp = client.chat.completions.create(                                       # 调用大模型
            model="qwen3.8-max",                                                     # 指定模型
            messages=[{"role": "user", "content": prompt}],                          # 传入提示词
            temperature=0.7                                                          # 设置温度，增加多样性
        )                                                                            # 调用结束
        raw = resp.choices[0].message.content.strip()                                # 获取返回文本并去除两端空白
        queries = [q.strip() for q in raw.split("\n") if q.strip()][:3]              # 按行切分，取前3个
        return [original_query] + queries                                            # 返回原始问题加改写问题
    except Exception:                                                                # 异常捕获
        return [original_query]                                                      # 发生异常时仅返回原始问题


def rerank_documents(query: str, candidates: list, top_n: int = 2):                  # 定义重排序函数
    if not candidates:                                                               # 如果候选列表为空
        return []                                                                    # 返回空列表
    docs_text = [doc.page_content for doc in candidates]                             # 提取候选文档的文本内容
    try:                                                                             # 尝试执行
        resp = TextReRank.call(                                                      # 调用百炼重排接口
            model="qwen3.7-text-rerank",                                             # 指定重排模型（有免费额度）
            query=query,                                                             # 传入原始查询
            documents=docs_text,                                                     # 传入候选文档文本
            top_n=top_n,                                                             # 指定保留数量
            return_documents=False                                                   # 不返回原始文档
        )                                                                            # 调用结束
        if resp is None or not hasattr(resp, 'output') or resp.output is None:       # 检查返回是否有效
            return candidates[:top_n]                                                # 无效则回退到原始排序
        return [candidates[item.index] for item in resp.output.results]              # 按重排结果提取文档
    except Exception:                                                                # 异常捕获
        return candidates[:top_n]                                                    # 发生异常时回退到原始排序

# =====================================================================
# ==================== RAG 工具（混合检索版） ========================
# =====================================================================
@tool                                                                                # 标记为 LangChain 工具
def search_knowledge_base(query: str) -> str:                                        # 定义 RAG 检索工具
    """RAG 检索（多路改写 + 混合检索 BM25/向量 + 父文档检索 + Rerank 精排）"""        # 工具说明（大模型靠这句话判断何时调用）
    if bm25_retriever is None:                                                       # 如果没有知识库
        return "知识库为空。"                                                        # 返回提示
    multi_queries = generate_multi_queries(query)                                    # 调用多路改写，生成 4 个查询
    all_candidates = []                                                              # 初始化候选列表
    seen = set()                                                                     # 初始化去重集合

    for q in multi_queries:                                                          # 遍历每个改写后的问题
        # 第一路：父文档检索（子块匹配 → 返回完整父块）
        try:
            parent_docs = parent_retriever.invoke(q)                                 # 调用父文档检索器
        except Exception:
            parent_docs = []                                                         # 失败时返回空列表
        # 第二路：BM25 关键词检索（返回子块）
        try:
            bm25_docs = bm25_retriever.invoke(q)                                     # 调用 BM25 检索器
        except Exception:
            bm25_docs = []                                                           # 失败时返回空列表

        # 合并两路结果，按内容去重
        for doc in parent_docs + bm25_docs:                                          # 遍历两路结果
            if doc.page_content not in seen:                                         # 如果内容未见过
                seen.add(doc.page_content)                                           # 加入去重集合
                all_candidates.append(doc)                                           # 加入候选列表

    final_docs = rerank_documents(query, all_candidates, top_n=2)                    # 调用重排序取 Top2
    return "\n".join([f"【检索结果】: {d.page_content}" for d in final_docs])         # 格式化返回


@tool                                                                                # 标记为工具
def search_wiki(query: str) -> str:                                                  # 定义 Wiki 查询工具
    """查询 LLM Wiki 结构化知识库"""                                                # 工具说明
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)                         # 连接数据库
    cursor = conn.cursor()                                                           # 创建游标
    cursor.execute(                                                                  # 执行查询
        "select topic, content from wiki_knowledge where topic like ? or content like ? limit 3", # SQL 语句
        (f"%{query}%", f"%{query}%")                                                 # 模糊匹配参数
    )                                                                                # 执行结束
    rows = cursor.fetchall()                                                         # 获取结果
    conn.close()                                                                     # 关闭连接
    if not rows:                                                                     # 如果没有结果
        return "Wiki 知识库中没有找到相关信息。"                                     # 返回提示
    return "\n".join([f"【{r[0]}】: {r[1]}" for r in rows])                           # 格式化返回

# =====================================================================
# ==================== 持仓管理工具 ==================================
# =====================================================================
@tool                                                                                # 标记为工具
def view_portfolio() -> str:                                                         # 定义查看持仓工具
    """查看当前所有持仓股票，包含成本、现价、浮动盈亏"""                              # 工具说明
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)                         # 连接数据库
    cursor = conn.cursor()                                                           # 创建游标
    cursor.execute("select code, name, shares, cost from my_portfolio")              # 查询持仓表
    rows = cursor.fetchall()                                                         # 获取结果
    conn.close()                                                                     # 关闭连接
    if not rows:                                                                     # 如果没有持仓
        return "当前没有持仓记录。"                                                  # 返回提示
    result = "📊 当前持仓：\n"                                                       # 初始化结果字符串
    total_cost = 0                                                                   # 总成本初始化
    total_value = 0                                                                  # 总市值初始化
    for code, name, shares, cost in rows:                                            # 遍历持仓
        price = _get_realtime_price(code)                                            # 获取实时价格
        cost_amount = cost * shares                                                  # 计算成本金额
        value_amount = price * shares                                                # 计算市值金额
        profit = value_amount - cost_amount                                          # 计算盈亏
        total_cost += cost_amount                                                    # 累加总成本
        total_value += value_amount                                                  # 累加总市值
        result += f"- {name}（{code}）：{shares} 股，成本 {cost:.2f}，现价 {price:.2f}，浮动盈亏 {profit:+.2f}\n" # 拼接明细
    result += f"\n💰 总成本：{total_cost:.2f} 元\n💼 总市值：{total_value:.2f} 元\n📈 总盈亏：{total_value - total_cost:+.2f} 元" # 拼接汇总
    return result                                                                    # 返回结果


@tool                                                                                # 标记为工具
def add_or_update_holding(code: str, name: str, shares: int, cost: float, note: str = "") -> str: # 定义更新持仓工具
    """新增或更新持仓。用户说“我买了X股XX”时调用"""                                # 工具说明
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)                         # 连接数据库
    cursor = conn.cursor()                                                           # 创建游标
    cursor.execute(                                                                  # 执行替换语句
        "insert or replace into my_portfolio (code, name, shares, cost, buy_date, note) values (?, ?, ?, ?, ?, ?)", # SQL
        (code, name, shares, cost, datetime.now().strftime("%Y-%m-%d"), note)        # 参数
    )                                                                                # 执行结束
    conn.commit()                                                                    # 提交事务
    conn.close()                                                                     # 关闭连接
    return f"✅ 已记录：{name}（{code}）{shares} 股，成本 {cost} 元。"                 # 返回成功信息


@tool                                                                                # 标记为工具
def remove_holding(code: str) -> str:                                                # 定义删除持仓工具
    """删除一只股票的持仓记录"""                                                    # 工具说明
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)                         # 连接数据库
    cursor = conn.cursor()                                                           # 创建游标
    cursor.execute("delete from my_portfolio where code = ?", (code,))               # 执行删除
    conn.commit()                                                                    # 提交事务
    conn.close()                                                                     # 关闭连接
    return f"✅ 已删除 {code} 的持仓记录。"                                           # 返回成功信息


@tool                                                                                # 标记为工具
def update_cost(code: str, new_cost: float) -> str:                                  # 定义修改成本价工具
    """修改持仓的成本价"""                                                          # 工具说明
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)                         # 连接数据库
    cursor = conn.cursor()                                                           # 创建游标
    cursor.execute("update my_portfolio set cost = ? where code = ?", (new_cost, code)) # 执行更新
    conn.commit()                                                                    # 提交事务
    conn.close()                                                                     # 关闭连接
    return f"✅ 已将 {code} 的成本价更新为 {new_cost} 元。"                           # 返回成功信息

# =====================================================================
# ==================== 行情 + 基本面 + 估值工具 ======================
# =====================================================================
@tool                                                                                # 标记为工具
def get_stock_realtime(code: str) -> str:                                            # 定义查询实时行情工具
    """查询股票实时行情：最新价、涨跌幅、成交量、成交额、PE、PB"""                    # 工具说明
    info = _get_tencent_stock_info(code)                                             # 获取腾讯行情数据
    if info["price"] == 0.0:                                                         # 如果获取失败
        return f"⚠️ 接口暂时不可用，请稍后再试。"                                     # 返回错误提示
    return (                                                                         # 返回格式化数据
        f"【腾讯行情接口返回，真实数据】\n"                                          # 数据源说明
        f"股票：{info['name']}（{code}）\n"                                          # 股票名称
        f"最新价：{info['price']} 元\n"                                              # 最新价
        f"涨跌幅：{info['change_pct']}%\n"                                           # 涨跌幅
        f"成交量：{info['volume']} 手\n"                                             # 成交量
        f"成交额：{info['amount']} 万元\n"                                           # 成交额
        f"市盈率(TTM)：{info['pe']}\n"                                               # 市盈率
        f"市净率：{info['pb']}\n"                                                    # 市净率
    )                                                                                # 返回结束


@tool                                                                                # 标记为工具
def get_fundamentals(code: str) -> str:                                              # 定义查询基本面工具
    """查询股票基本面：ROE、毛利率、净利润、营收、现金流"""                          # 工具说明
    try:                                                                             # 尝试执行
        df = ak.stock_financial_analysis_indicator(symbol=code)                      # 调用 AKShare 接口
        latest = df.iloc[0]                                                          # 获取最新一期数据
        return (                                                                     # 返回格式化数据
            f"股票：{code}\n"                                                        # 股票代码
            f"ROE：{latest.get('净资产收益率(%)')}\n"                                # ROE
            f"毛利率：{latest.get('销售毛利率(%)')}\n"                               # 毛利率
            f"净利润：{latest.get('净利润(元)')}\n"                                  # 净利润
            f"主营业务收入：{latest.get('主营业务收入(元)')}\n"                      # 营收
        )                                                                            # 返回结束
    except Exception as e:                                                           # 异常捕获
        return f"查询基本面失败：{e}"                                                # 返回错误信息


@tool                                                                                # 标记为工具
def get_valuation(code: str) -> str:                                                 # 定义查询估值工具
    """查询股票估值：PE、PB（腾讯接口，实时准确）"""                                # 工具说明
    info = _get_tencent_stock_info(code)                                             # 获取腾讯行情数据
    if info["price"] == 0.0:                                                         # 如果获取失败
        return f"⚠️ 接口暂时不可用，请稍后再试。"                                     # 返回错误提示
    return (                                                                         # 返回格式化数据
        f"股票：{info['name']}（{code}）\n"                                          # 股票名称
        f"市盈率(TTM)：{info['pe']}\n"                                               # 市盈率
        f"市净率：{info['pb']}\n"                                                    # 市净率
    )                                                                                # 返回结束

# =====================================================================
# ==================== K线图工具 =====================================
# =====================================================================
@tool                                                                                # 标记为工具
def plot_stock_chart(code: str, period: str = "daily", days: int = 120) -> str:      # 定义 K 线图工具
    """生成 K 线走势图（腾讯前复权数据）"""                                          # 工具说明
    try:                                                                             # 尝试执行
        prefix = _get_tencent_prefix(code)                                           # 获取沪深前缀
        url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{code},day,,,{days},qfq" # 拼接 URL
        resp = requests.get(url, timeout=10).json()                                  # 请求接口并解析 JSON
        data = resp["data"][f"{prefix}{code}"]["qfqday"]                             # 提取前复权日线数据

        df = pd.DataFrame(data, columns=["Date", "Open", "Close", "High", "Low", "Volume"]) # 创建 DataFrame
        df["Date"] = pd.to_datetime(df["Date"])                                      # 转换日期格式
        df = df.set_index("Date")                                                    # 将日期设为索引
        for col in ["Open", "Close", "High", "Low", "Volume"]:                       # 遍历数值列
            df[col] = pd.to_numeric(df[col])                                         # 转换为数值类型

        if period == "weekly":                                                       # 如果是周线
            df = df.resample("W").agg({"Open": "first", "Close": "last", "High": "max", "Low": "min", "Volume": "sum"}).dropna() # 重采样
        elif period == "monthly":                                                    # 如果是月线
            df = df.resample("M").agg({"Open": "first", "Close": "last", "High": "max", "Low": "min", "Volume": "sum"}).dropna() # 重采样

        os.makedirs("charts", exist_ok=True)                                         # 创建 charts 目录
        filename = f"charts/{code}_{period}_{days}.png"                              # 生成文件名
        mpf.plot(df, type="candle", mav=(5, 10, 20, 60), volume=True, style="yahoo", # 绘制 K 线图
                 title=f"{code} - {period} K线", savefig=filename)                   # 设置标题并保存
        return f"✅ K线图已生成：{filename}"                                          # 返回成功信息
    except Exception as e:                                                           # 异常捕获
        return f"生成K线图失败：{e}"                                                 # 返回错误信息

# =====================================================================
# ==================== 每日复盘报告 ==================================
# =====================================================================
@tool                                                                                # 标记为工具
def daily_review_report() -> str:                                                    # 定义生成复盘报告工具
    """生成今日持仓复盘报告，包含盈亏、涨跌、风险提示"""                              # 工具说明
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)                         # 连接数据库
    cursor = conn.cursor()                                                           # 创建游标
    cursor.execute("select code, name, shares, cost from my_portfolio")              # 查询持仓
    rows = cursor.fetchall()                                                         # 获取结果
    conn.close()                                                                     # 关闭连接

    if not rows:                                                                     # 如果没有持仓
        return "当前没有持仓，无法生成复盘报告。"                                    # 返回提示

    today = datetime.now().strftime("%Y-%m-%d")                                      # 获取今日日期
    report = f"# {today} 持仓复盘报告\n\n"                                           # 初始化报告标题

    total_cost = 0                                                                   # 总成本
    total_value = 0                                                                  # 总市值
    details = []                                                                     # 明细列表
    for code, name, shares, cost in rows:                                            # 遍历持仓
        price = _get_realtime_price(code)                                            # 获取实时价格
        cost_amount = cost * shares                                                  # 成本金额
        value_amount = price * shares                                                # 市值金额
        profit = value_amount - cost_amount                                          # 盈亏
        profit_pct = (profit / cost_amount * 100) if cost_amount else 0              # 盈亏百分比
        total_cost += cost_amount                                                    # 累加总成本
        total_value += value_amount                                                  # 累加总市值
        details.append(f"| {name} | {shares} | {cost:.2f} | {price:.2f} | {profit:+.2f} | {profit_pct:+.2f}% |") # 添加明细

    report += "## 一、今日持仓总览\n\n"                                              # 添加章节
    report += "| 股票 | 股数 | 成本 | 现价 | 盈亏 | 盈亏% |\n"                        # 表格头
    report += "|------|------|------|------|------|-------|\n"                       # 分隔线
    report += "\n".join(details) + "\n\n"                                            # 表格内容

    report += f"## 二、总资产\n\n- 总成本：{total_cost:.2f} 元\n- 总市值：{total_value:.2f} 元\n- 总盈亏：{total_value - total_cost:+.2f} 元\n\n" # 总资产章节

    report += "## 三、风险提示\n\n"                                                  # 风险提示章节
    for code, name, shares, cost in rows:                                            # 遍历持仓
        price = _get_realtime_price(code)                                            # 获取实时价格
        profit_pct = (price - cost) / cost * 100 if cost else 0                      # 盈亏百分比
        if profit_pct <= -5:                                                         # 如果浮亏超过 5%
            report += f"- ⚠️ {name} 浮亏 {profit_pct:.2f}%，请关注\n"                 # 添加风险提示
        elif profit_pct >= 20:                                                       # 如果浮盈超过 20%
            report += f"- 💰 {name} 浮盈 {profit_pct:.2f}%，可考虑止盈\n"             # 添加止盈提示

    os.makedirs("reports", exist_ok=True)                                            # 创建 reports 目录
    filepath = f"reports/{today}.md"                                                 # 报告文件路径
    with open(filepath, "w", encoding="utf-8") as f:                                 # 写入文件
        f.write(report)                                                              # 写入内容

    return f"✅ 复盘报告已生成：{filepath}\n\n{report}"                               # 返回报告路径和内容

# =====================================================================
# ============ 创建 Worker ===========================================
# =====================================================================
rag_worker = create_agent(                                                           # 创建 RAG 专家 Worker
    model="openai:qwen3.8-max",                                                      # 使用 qwen3.8-max 模型
    tools=[search_knowledge_base, search_wiki],                                      # 绑定 RAG 和 Wiki 工具
    system_prompt="你是一个投资知识专家。你只负责回答投资知识、财务指标含义、价值投资原则等相关问题。", # 系统提示词
    name="rag_worker"                                                                # 指定 Worker 名称
)

portfolio_worker = create_agent(                                                     # 创建持仓管家 Worker
    model="openai:qwen3.8-max",                                                      # 使用 qwen3.8-max 模型
    tools=[                                                                          # 绑定所有持仓相关工具
        view_portfolio, add_or_update_holding, remove_holding, update_cost,           # 持仓管理工具
        get_stock_realtime, get_fundamentals, get_valuation,                          # 行情/基本面/估值工具
        plot_stock_chart, daily_review_report                                         # 图表和复盘工具
    ],                                                                               # 工具列表结束
    system_prompt=(                                                                  # 系统提示词
        "你是一个专业的持仓管家。你负责管理用户的真实持仓，并能分析股票。\n"       # 角色说明
        "你能做的事情：\n"                                                          # 能力列表开始
        "1. 管理持仓：查看、新增、删除、修改成本价\n"                               # 能力1
        "2. 分析股票：实时行情、基本面、估值\n"                                     # 能力2
        "3. 生成 K 线图\n"                                                          # 能力3
        "4. 生成每日复盘报告\n"                                                     # 能力4
        "重要规则：\n"                                                              # 规则开始
        "- 用户说'我买了X股XX'时，调用 add_or_update_holding\n"                      # 规则1
        "- 用户说'我卖了XX'时，调用 remove_holding\n"                               # 规则2
        "- 用户说'XX成本改成Y'时，调用 update_cost\n"                               # 规则3
        "- 每次操作完成后，简要总结结果。\n"                                        # 规则4
        "- 【重要】工具返回的数据就是真实数据，直接使用，不要怀疑、不要编造、不要道歉。\n" # 规则5
        "- 【重要】如果工具返回失败，直接告诉用户'接口暂时不可用，请稍后再试'，不要说自己是编的。" # 规则6
    ),                                                                               # 提示词结束
    name="portfolio_worker"                                                          # 指定 Worker 名称
)

# =====================================================================
# ============ 创建 Supervisor =======================================
# =====================================================================
supervisor_llm = ChatOpenAI(                                                         # 初始化 Supervisor 的 LLM
    model="qwen3.8-max",                                                             # 使用 qwen3.8-max 模型
    api_key=os.getenv("DASHSCOPE_API_KEY"),                                          # 传入 API Key
    base_url=os.getenv("DATA_BASE_URL"),                                             # 传入 Base URL
    temperature=0                                                                    # 温度设为 0，保证决策稳定
)

supervisor = create_supervisor(                                                      # 创建 Supervisor
    agents=[rag_worker, portfolio_worker],                                           # 下属 Worker 列表
    model=supervisor_llm,                                                            # 指定模型
    prompt="""你是一个主管。请根据用户问题分派任务：                                 # 主管提示词
- 投资知识、财务指标含义 → rag_worker                                               # 分派规则1
- 持仓管理、股票分析、K线图、复盘报告 → portfolio_worker                             # 分派规则2
- 跟以上都无关 → 礼貌拒绝                                                          # 分派规则3

【重要规则】                                                                        # 强调规则
1. 不要自己回答用户的问题，只负责分派。                                             # 规则1
2. 分派后，直接信任 Worker 返回的结果，不要质疑、不要道歉、不要二次验证。             # 规则2
3. 如果 Worker 返回失败，直接告诉用户“接口暂时不可用”，不要说“数据是编造的”。         # 规则3
"""                                                                                  # 提示词结束
)

agent = supervisor.compile()                                                         # 编译 Supervisor 图，生成最终 Agent

__all__ = ["agent", "vectorstore", "md_header_splits", "DB_PATH", "init_db", "save_message", "load_messages", "clear_db"] # 导出公共接口