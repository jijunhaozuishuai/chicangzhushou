# =====================================================================
# rag_eval.py - RAG 双层评估
# 运行方式：python rag_eval.py
# =====================================================================
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent_core import agent, vectorstore, md_header_splits

load_dotenv()

def run_rag_evaluation():
    print("-" * 50)
    print("😎 RAG 双层评估（检索命中 + 答案命中）")
    print("-" * 50)

    try:
        with open("eval_data.json", "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        print(f"✅ 成功加载测试集：{len(test_cases)} 个问题")
    except FileNotFoundError:
        print("❌ 找不到 eval_data.json")
        return
    except json.JSONDecodeError:
        print("❌ eval_data.json 格式错误")
        return

    hit_count = 0
    mrr_score = 0
    answer_correct_count = 0

    judge_client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DATA_BASE_URL")
    )

    for i, case in enumerate(test_cases, 1):
        query = case["question"]
        expected_id = case["expected_id"]
        expected_answer = case.get("expected_answer", "")

        print(f"\n===== 第 {i} 题：{query} =====")

        # 第一层：检索命中
        results = vectorstore.similarity_search(query, k=3)
        is_hit = False
        for rank, doc in enumerate(results):
            if doc.metadata.get("chunk_id") == expected_id:
                hit_count += 1
                is_hit = True
                mrr_score += 1 / (rank + 1)
                break
        print(f"  🔍 检索：{'🟢 命中' if is_hit else '🔴 未命中'}")

        # 第二层：答案命中
        try:
            result = agent.invoke({"messages": [HumanMessage(content=query)]})
            ai_answer = result["messages"][-1].content
        except Exception as e:
            ai_answer = f"（Agent 调用失败：{e}）"

        print(f"  🤖 AI 回答：{ai_answer[:60]}...")

        judge_prompt = f"""你是一个严格的评分员。请判断下面"AI回答"是否准确回答了"用户问题"，并且与"标准答案"的核心意思一致。
注意：不要因为格式差异（如Markdown符号、标点不同）而判错，只看语义是否一致。
只需要输出一个单词：YES 或 NO。

用户问题：{query}
标准答案：{expected_answer}
AI回答：{ai_answer}
"""
        try:
            judge_res = judge_client.chat.completions.create(
                model="qwen-max",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0
            )
            judge_result = judge_res.choices[0].message.content.strip().upper()
        except Exception as e:
            judge_result = f"ERROR（{e}）"

        if "YES" in judge_result:
            answer_correct_count += 1
            print(f"  📝 答案：🟢 正确")
        else:
            print(f"  📝 答案：🔴 错误（裁判判定：{judge_result}）")

    total = len(test_cases)
    print("\n" + "=" * 50)
    print("📊 RAG 双层评估报告：")
    print(f"总测试题数：{total}")
    print("--- 第一层：检索命中 ---")
    print(f"命中率 (Hit Rate)：{hit_count / total * 100:.2f}%")
    print(f"平均倒数排名 (MRR)：{mrr_score / total:.4f}")
    print("--- 第二层：答案命中 ---")
    print(f"答案正确率 (Answer Accuracy)：{answer_correct_count / total * 100:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    run_rag_evaluation()