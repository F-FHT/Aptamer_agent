# agent/graph.py
# 职责：用 LangGraph 把 agent / tools 节点串成可执行工作流并编译导出。
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import agent_node, should_continue, tool_node
from agent.state import AgentState
from observability.logger import get_logger

log = get_logger(__name__)

graph = StateGraph(AgentState)

# ---- 节点 ----
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# ---- 边与路由 ----
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END},
)
# 工具执行完后回到 agent，让模型基于工具结果继续推理或给出最终回答
graph.add_edge("tools", "agent")

# ---- 长期记忆（Checkpointer）----
# MemorySaver 把每轮状态按 thread_id 存在内存中，多轮对话可沿用 dataset_id
memory = MemorySaver()

# ---- 编译导出 ----
# interrupt_after=["tools"]：工具执行完立即挂起，等待用户确认/追问后再恢复
log.info("正在编译 LangGraph 智能体工作流...")
app = graph.compile(checkpointer=memory, interrupt_after=["tools"])
