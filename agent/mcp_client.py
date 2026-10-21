# agent/mcp_client.py
from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from observability.logger import get_logger

log = get_logger(__name__)

# 读取项目根目录的 .env（已存在的系统环境变量优先）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
port = os.getenv("MCP_SERVER_PORT", "8000")
MCP_SERVER_URL = f"http://{host}:{port}/mcp"

tools_cache: list[StructuredTool] = []  # 避免每次对话或推理时都重新请求网络拉取工具列表。
_session: ClientSession | None = None  # 全局唯一的 MCP 客户端会话
_exit_stack: AsyncExitStack | None = None# 异步退出栈（Async Exit Stack）,作用：持有并管理异步上下文管理器（Context Managers）的生命周期。
_session_lock = asyncio.Lock()  # 防止并发初始化冲突


async def _get_session() -> ClientSession:
    """获取（或首次建立）与 MCP Server 的连接。"""
    global _session, _exit_stack
    if _session is None:
        async with _session_lock:
            if _session is None:
                stack = AsyncExitStack()  # 准备一个“收纳盒”（ExitStack），用来暂存长连接对象。
                # mcp 2.x 返回 (read_stream, write_stream)
                read_stream, write_stream = await stack.enter_async_context(
                    streamable_http_client(MCP_SERVER_URL)
                )
                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                _session, _exit_stack = session, stack
    return _session


async def close_session() -> None:
    """关闭连接（建议在程序生命周期的收尾阶段调用，如 FastAPI 的 shutdown 事件）。"""
    global _session, _exit_stack
    if _exit_stack:
        await _exit_stack.aclose()
        _session = None  # 把全局变量重置为 None。
        _exit_stack = None


def _content_to_text(result: Any) -> str:  # 把 MCP 工具调用的返回结果（CallToolResult）转换为纯字符串。
    blocks = getattr(result, "content", None) or []  # 1. 获取内容块列表。
    texts = [  # 2. 提取所有的文本块。
        block.text for block in blocks
        if getattr(block, "text", None) is not None
    ]
    return "\n".join(texts) if texts else str(blocks)  # 3. 组装返回结果。

"""
    把一个远程的 MCP Tool 描述（工具名、说明、参数结构），
    转换成 LangChain 能够直接发给大模型使用的 StructuredTool。
"""
def _build_tool(
    tool: Any,
    session: ClientSession,
    args_schema: dict[str, Any] | None = None,
) -> StructuredTool:
    async def _call(**kwargs: Any) -> str:
        log.info(f"调用工具: {tool.name}, 参数: {kwargs}")
        result = await session.call_tool(tool.name, kwargs)
        log.info(f"工具 {tool.name} 执行完成")
        return _content_to_text(result)

    tool_kwargs: dict[str, Any] = {
        "coroutine": _call,  # 指定真正的异步执行逻辑
        "name": tool.name,  # 告诉模型：这个工具叫什么名字
        "description": tool.description or f"MCP tool: {tool.name}",  # 告诉模型：这个工具能干啥
    }
    # 仅在成功拿到参数定义时传入；拿不到就省略，避免不同 SDK 版本直接报错
    if args_schema:
        tool_kwargs["args_schema"] = args_schema

    return StructuredTool.from_function(**tool_kwargs)


async def get_mcp_tools() -> list[StructuredTool]:
    log.info("正在连接 MCP Server 并拉取工具...")
    session = await _get_session()  # 获取或建立与服务器的会话连接
    listed = await session.list_tools()  # 把服务器上所有的工具列表拉下来。

    tools: list[StructuredTool] = []
    for tool in listed.tools:
        # 兼容不同版本 MCP SDK：inputSchema / input_schema / model_dump
        schema = getattr(tool, "inputSchema", getattr(tool, "input_schema", {}))
        if not schema and hasattr(tool, "model_dump"):
            try:
                dumped = tool.model_dump()
            except Exception:
                dumped = {}
            schema = dumped.get("inputSchema") or dumped.get("input_schema") or {}
        if not isinstance(schema, dict):
            schema = {}

        tools.append(_build_tool(tool, session, schema))

    log.info(f"成功获取 {len(tools)} 个 MCP 工具")
    return tools


async def init_tools() -> list[StructuredTool]:
    global tools_cache
    tools_cache = await get_mcp_tools()
    return tools_cache