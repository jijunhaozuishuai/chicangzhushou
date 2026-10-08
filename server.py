# =====================================================================
# server.py - FastAPI 后端服务
# 功能：流式输出 + 鉴权 + 中断 + 历史记录 + 持仓查询 + 静态文件挂载
# 运行方式：uvicorn server:app --reload
# =====================================================================
import json
import os
import sqlite3
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from agent_core import agent, save_message, load_messages, clear_db, MAX_HISTORY_MESSAGES, DB_PATH

app = FastAPI(title="AI 持仓管家", description="多Agent + RAG + 持仓管理 + 行情分析")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# ==================== 静态文件挂载 ==================================
# =====================================================================
# 让前端可以直接访问 /charts/xxx.png 和 /reports/xxx.md
os.makedirs("charts", exist_ok=True)
os.makedirs("reports", exist_ok=True)
app.mount("/charts", StaticFiles(directory="charts"), name="charts")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# =====================================================================
# ==================== 鉴权 ==========================================
# =====================================================================
API_SECRET = os.getenv("API_SECRET", "default-secret-change-me")

def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="缺少 Authorization 请求头")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization 格式错误")
    token = authorization.replace("Bearer ", "").strip()
    if token != API_SECRET:
        raise HTTPException(status_code=403, detail="Token 无效")
    return True

class ChatRequest(BaseModel):
    message: str

chat_memory = load_messages()

# =====================================================================
# ==================== 接口：/history ================================
# =====================================================================
@app.get("/history")
async def get_history(authorized: bool = Depends(verify_token)):
    """拉取最近的历史对话"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(f"select role, content from chat_history order by id desc limit {MAX_HISTORY_MESSAGES * 2}")
    rows = cursor.fetchall()
    conn.close()
    rows = list(reversed(rows))
    return {"history": [{"role": r[0], "content": r[1]} for r in rows]}

# =====================================================================
# ==================== 接口：/portfolio ==============================
# =====================================================================
@app.get("/portfolio")
async def get_portfolio(authorized: bool = Depends(verify_token)):
    """拉取当前持仓，供前端左侧边栏实时显示"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("select code, name, shares, cost from my_portfolio")
    rows = cursor.fetchall()
    conn.close()

    # 对每只股票拉实时价，算盈亏
    import akshare as ak
    result = []
    total_cost = 0
    total_value = 0
    for code, name, shares, cost in rows:
        try:
            df = ak.stock_individual_info_em(symbol=code)
            info = dict(zip(df["item"], df["value"]))
            price = float(info.get("最新", 0))
        except Exception:
            price = 0.0
        cost_amount = cost * shares
        value_amount = price * shares
        profit = value_amount - cost_amount
        profit_pct = (profit / cost_amount * 100) if cost_amount else 0
        total_cost += cost_amount
        total_value += value_amount
        result.append({
            "code": code, "name": name, "shares": shares,
            "cost": cost, "price": price,
            "profit": round(profit, 2), "profit_pct": round(profit_pct, 2)
        })

    return {
        "positions": result,
        "total_cost": round(total_cost, 2),
        "total_value": round(total_value, 2),
        "total_profit": round(total_value - total_cost, 2)
    }

# =====================================================================
# ==================== 接口：/chat ===================================
# =====================================================================
@app.post("/chat")
async def chat_endpoint(
    req: ChatRequest,
    request: Request,
    authorized: bool = Depends(verify_token)
):
    global chat_memory
    chat_memory.append(HumanMessage(content=req.message))
    save_message("user", req.message)

    async def generate_stream():
        global chat_memory
        full_answer = ""
        iteration_count = 0
        max_iterations = 5

        try:
            async for msg, metadata in agent.astream(
                {"messages": chat_memory},
                stream_mode="messages"
            ):
                # 主管分派日志
                if msg.__class__.__name__ == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get("name", "")
                        if tool_name.startswith("transfer_to"):
                            target = tool_name.replace("transfer_to_", "")
                            print(f"\n[主管分派] → {target}")

                # 中断检查
                if await request.is_disconnected():
                    print("\n[中断机制] 前端已断开，停止调用大模型")
                    break

                # ReAct 循环保护
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    iteration_count += 1
                    if iteration_count > max_iterations:
                        yield f"data: {json.dumps({'content': '⚠️ 已达最大思考轮数'}, ensure_ascii=False)}\n\n"
                        break
                    for tc in msg.tool_calls:
                        print(f"  🛠️ 调用 `{tc.get('name')}`")

                if msg.__class__.__name__ == "ToolMessage":
                    print(f"  👀 工具返回")

                # 流式输出
                if hasattr(msg, "content") and msg.content:
                    content = msg.content
                    if isinstance(content, list):
                        content = "".join([p.get('text', '') for p in content if isinstance(p, dict) and 'text' in p])
                    if content and isinstance(content, str):
                        full_answer += content
                        yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

            save_message("assistant", full_answer)
            chat_memory.append(AIMessage(content=full_answer))
            if len(chat_memory) > MAX_HISTORY_MESSAGES:
                chat_memory = chat_memory[-MAX_HISTORY_MESSAGES:]

        except Exception as e:
            print(f"流式输出出错: {e}")
            yield f"data: {json.dumps({'content': '抱歉，处理你的请求时发生错误。'})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

# =====================================================================
# ==================== 接口：/clear ==================================
# =====================================================================
@app.post("/clear")
async def clear_endpoint(authorized: bool = Depends(verify_token)):
    global chat_memory
    chat_memory = []
    clear_db()
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)