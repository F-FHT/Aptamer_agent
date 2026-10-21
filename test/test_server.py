import asyncio
import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from mcp import ClientSession
# 注意这里导入的是 streamable_http_client
from mcp.client.streamable_http import streamable_http_client

# 与 MCP Server 读取同一份 .env，避免自定义端口后测试仍连默认端口
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@pytest.mark.asyncio
async def test_pipeline():
    # streamable-http 的默认路由是 /mcp
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = os.getenv("MCP_SERVER_PORT", "8000")
    server_url = f"http://{host}:{port}/mcp"

    print(f"尝试连接到 MCP Server: {server_url} ...")

    try:
        async with streamable_http_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                # 1. 初始化握手
                await session.initialize()
                print("✅ 成功连接到 MCP Server！\n")

                # 2. 查看可用工具
                tools = await session.list_tools()
                print(f"检测到 {len(tools.tools)} 个工具：")
                for t in tools.tools:
                    print(f" - {t.name}")
                print("\n" + "=" * 40 + "\n")

                # 3. 模拟调用：第一步 (Load)
                print("▶️ 测试调用 tool_load_sequences (直接传序列)...")
                res_load = await session.call_tool(
                    name="tool_load_sequences",
                    arguments={"sequences": ["AAAACCCCGGGGTTTT", "AAAACCCCTGGGTTTT"]}
                )
                load_data = json.loads(res_load.content[0].text)
                dataset_id = load_data["dataset_id"]
                print(f"Load 返回 dataset_id: {dataset_id}")
                print(f"Load meta: {load_data['meta']}\n")

                # 4. 模拟调用：第二步 (Clean)
                print("▶️ 测试调用 tool_clean_sequences (引物截取)...")
                res_clean = await session.call_tool(
                    name="tool_clean_sequences",
                    arguments={
                        "dataset_id": dataset_id,
                        "fwd_primer": "AAAA",
                        "rev_primer": "TTTT",
                        "target_len": 8,
                        "tolerance": 0.5
                    }
                )
                clean_data = json.loads(res_clean.content[0].text)
                clean_id = clean_data["dataset_id"]
                print(f"Clean 返回 dataset_id: {clean_id}")
                print(f"Clean meta: {clean_data['meta']}\n")

                # 5. 模拟调用：第三步 (Structure)
                print("▶️ 测试调用 tool_compare_structures (二级结构预测)...")
                try:  # 👈 在调用工具的地方单独捕获异常
                    res_struct = await session.call_tool(
                        name="tool_compare_structures",
                        arguments={
                            "dataset_id": clean_id,
                            "source": "direct"
                        }
                    )
                    struct_data = json.loads(res_struct.content[0].text)
                    seq_id = struct_data["dataset_id"]
                    pairs_id = struct_data["pairs_dataset_id"]
                    print(f"Structure 返回 dataset_id: {seq_id}")
                    print(f"Structure 返回 pairs_dataset_id: {pairs_id}\n")

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    pytest.fail(f"调用工具失败，真实错误是: {e}")

                # 6. 模拟调用：第四步 (Rank)
                print("▶️ 测试调用 tool_rank_candidates (综合打分)...")
                res_rank = await session.call_tool(
                    name="tool_rank_candidates",
                    arguments={
                        "dataset_id": seq_id,
                        "pairs_dataset_id": pairs_id,
                        "top_n": 2
                    }
                )
                rank_data = json.loads(res_rank.content[0].text)
                print("Rank 返回结果:")
                print(json.dumps(rank_data, indent=2, ensure_ascii=False))

    except Exception as e:
        # 抛出异常，让 pytest 能真正识别为失败
        pytest.fail(f"连接或测试失败: {e}")