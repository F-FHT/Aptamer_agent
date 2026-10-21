from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired


class AgentState(TypedDict):
    """LangGraph 智能体在节点之间传递的状态。"""

    messages: Annotated[list[AnyMessage], add_messages]
    dataset_id: NotRequired[str | None]
    pairs_dataset_id: NotRequired[str | None]
    # 用户原始输入，用于 tool_node 在缺少 dataset_id 时自动加载数据
    raw_input_path: NotRequired[str | None]
    raw_input_sequences: NotRequired[list[str] | None]
    # 各工具返回的原始表格：{工具名: [行字典, ...]}，前端可据此做多表格展示。
    # 未传入时请以 {} 初始化（TypedDict 无法声明可变默认值）。
    history_tables: NotRequired[dict[str, list[dict]]]
    metadata: NotRequired[dict[str, Any]]
