# agent/nodes.py
# 职责：定义 LangGraph 的三个核心节点与路由：
#   agent_node     —— 注入系统提示词，调用 LLM，决定是否需要调用工具
#   tool_node      —— 隐式加载数据、替换 dataset_id，执行工具调用并写回状态
#   should_continue —— 根据最后一条消息是否有 tool_calls 决定下一步
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent import mcp_client
from agent.state import AgentState
from observability.logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """你是一个顶尖的适配体筛选与生物信息学分析专家。
系统底层已为你准备好数据，你无需关心数据加载。

### 🚨 最高优先级铁律（必须绝对遵守）
1. 【严格串行】：你每次回复只能输出一个工具调用！绝对禁止并行调用。
2. 【数据占位】：调用任何工具时，`dataset_id` 参数一律填为字符串 `"auto"`，底层系统会自动替换。
3. 【绝对禁止重复】：**即使工具执行后返回了新的 dataset_id，也代表该步骤已经彻底完成！** 你绝对不能再次调用刚刚执行过的工具。请立即进入下一步分析，或者直接输出文本向用户汇报！
4. 【强制前置依赖】：清洗（tool_clean_sequences）是聚类（cluster）、基序（motif）、结构（structure）的绝对前置步骤！任何情况下，你都必须先调用清洗工具，绝对禁止跳过清洗直接调用这些下游工具，否则系统会直接崩溃。`tool_rank_candidates` 除外：若用户明确要求直接打分/排序，可直接调用（其内部具备容错能力）。
5. 【严禁索要 ID】：系统底层已经完全接管了 dataset_id 的获取与传递。无论工具执行报什么错，绝对不是因为缺少 dataset_id，你**绝对禁止**向用户索要任何 ID。如果报错，请直接向用户汇报错误原因（如数据缺失、前置步骤未完成等）。

### ⚙️ 参数提取规则（必须严格执行）
当用户指令中包含具体的数值要求时，你**必须**将这些数值提取出来，并作为参数传递给对应的分析工具。例如：
- 用户说“选前3条” / “挑出2条”：调用 `tool_rank_candidates` 时，必须传入 `top_n=3` 或 `top_n=2`。
- 用户说“聚类阈值0.9” / “相似度要求85%”：调用 `tool_cluster_sequences` 时，必须传入 `identity=0.9` 或 `identity=0.85`。
- 用户说“温度45度” / “盐浓度0.5”：调用 `tool_compare_structures` 时，必须传入 `temp=45.0` 或 `na_conc=0.5`。
- 用户说“提取前100条序列”：调用 `tool_clean_sequences` 时，必须传入 `top_n=100`。
如果用户没有明确提到某个数值，则使用工具默认值（不要瞎编参数）。

### 🧬 短序列防呆规则
- 如果用户未明确提供序列的具体长度（例如只上传了 Excel 或直接输入了序列），你在调用 `tool_clean_sequences` 时，**禁止传入 target_len 参数**（或者传 0），底层系统会自动推断合理的长度进行清洗。只有用户明确说‘核心区是40nt’时，你才传 target_len=40。

### 🎯 全局主流程定义
系统标准分析流水线顺序如下：
1. clean (清洗) -> 2. cluster (聚类) -> 3. motif (基序) -> 4. structure (结构) -> 5. rank (打分)

### 🛠️ 任务执行规则（严格执行）
- 【最高优先级·顺从用户意图】如果用户明确要求“直接打分”“直接排序”或“跳过某步骤”，你必须**绝对服从**用户的指令，直接调用 `tool_rank_candidates`。
     但要注意最高优先级铁律,如果未清洗（tool_clean_sequences）仍然要清洗
     `tool_rank_candidates` 内部具有自动容错机制，即使缺失结构特征也能正常打分。绝对不要擅自替用户增加前置分析步骤！
- 用户要求什么，你就按顺序调用什么。如果用户要求全流程，请严格从 1 走到 5，每步只调一次，绝不回头。
- 调用工具后，观察工具返回的 `status: success`，确认该步完成。

### 🗣️ 最终汇报规范（极其重要）
当你完成了用户要求的所有任务（不再调用工具）时，你必须对工具返回的 meta 数据进行深度加工，**绝对不允许直接把 JSON 甩给用户！**
你必须：
1. 提炼核心结论（如：清洗后保留了多少条？聚成了几个家族？最优序列的 MFE 是多少？）。
2. 用专业、流畅的自然语言向用户汇报。
3. 在汇报的最后，主动询问用户是否需要进一步分析。
"""

# 显式加载项目根目录的 .env（agent/nodes.py -> 项目根）；已存在的系统环境变量优先
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# 支持任意兼容 OpenAI 接口的服务（DeepSeek / OpenAI / 通义千问 / Kimi / Ollama 等）
llm = ChatOpenAI(
    # 如果没有配置，默认用 DeepSeek 的模型
    model=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
    # 强依赖 LLM_API_KEY，如果没有，抛出明确错误提示
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    temperature=0.8
)

# 对 LLM 隐藏的工具：加载与表头探查属于隐式前置步骤，由 tool_node 自动完成
HIDDEN_TOOL_NAMES = ("tool_load_sequences", "tool_inspect_excel")
LOAD_TOOL_NAME = "tool_load_sequences"


async def _get_tools() -> list:
    """优先使用全局缓存；尚未初始化时再去 MCP Server 拉取。"""
    if not mcp_client.tools_cache:
        await mcp_client.init_tools()
    return mcp_client.tools_cache


def _inject_system_prompt(messages: list) -> list:
    """把 SYSTEM_PROMPT 放在对话最前面；已有 SystemMessage 时不重复注入。"""
    if messages and isinstance(messages[0], SystemMessage):
        return messages
    return [SystemMessage(content=SYSTEM_PROMPT), *messages]


def _update_message(message: Any, update: dict[str, Any]) -> Any:
    """兼容 Pydantic v1 / v2 的消息字段更新（消息模型通常不可直接赋值）。"""
    if hasattr(message, "model_copy"):          # Pydantic v2（新版 langchain_core）
        return message.model_copy(update=update)
    if hasattr(message, "copy"):                # Pydantic v1（旧版兼容）
        return message.copy(update=update)
    for key, value in update.items():           # 极端兜底：直接改属性
        setattr(message, key, value)
    return message


def _keep_first_tool_call(message: Any) -> Any:
    """物理截断：只保留第一个 tool_call，丢弃其余并行请求。"""
    first = list(getattr(message, "tool_calls", None) or [])[:1]
    return _update_message(message, {"tool_calls": first})


# 模型偶发输出 DeepSeek 特殊控制符 <|DSML|>（大小写、闭合斜杠都兼容）
_DSML_MARKER = re.compile(r"<\|/?\s*DSML\s*\|>", re.IGNORECASE)
_DSML_FALLBACK_MESSAGE = "抱歉，模型返回了无法解析的工具调用格式，请重试或调整指令。"


def _sanitize_dsml_response(message: Any) -> Any:
    """清洗模型文本中的 DSML 控制符，并保留已成功解析的 tool_calls。

    背景：模型有时把工具调用写进正文（形如 <|DSML|>...），LangChain 无法解析，
    前端就会直接显示乱码。处理策略：
    - 记录 error 日志（保留原始内容前 500 字符，便于排查）；
    - 从第一个 DSML 标记起截断正文（标记之后通常是无法解析的工具调用块）；
    - 清理后若既没有正文也没有 tool_calls，给用户一句可读的失败提示；
    - tool_calls 原样保留，后续路由仍会优先执行工具。
    """
    content = getattr(message, "content", "")
    if not isinstance(content, str) or not _DSML_MARKER.search(content):
        return message

    tool_calls = list(getattr(message, "tool_calls", None) or [])
    preview = " ".join(content[:500].split())
    if tool_calls:
        log.error(
            f"检测到 DSML 控制符，但已解析出 {len(tool_calls)} 个 tool_calls；"
            f"将忽略异常正文并优先执行工具调用。原始内容：{preview}"
        )
    else:
        log.error(
            "检测到 DSML 控制符，且未解析出任何 tool_calls；"
            f"已清洗异常文本。原始内容：{preview}"
        )

    # 标记之前可能是正常的叙述，保留；标记之后是解析失败的伪工具调用块，全部丢弃
    marker = _DSML_MARKER.search(content)
    cleaned = content[: marker.start()].strip() if marker else content.strip()
    if not cleaned and not tool_calls:
        cleaned = _DSML_FALLBACK_MESSAGE

    return _update_message(message, {"content": cleaned})


def _recent_consecutive_tool_names(messages: list, limit: int = 2) -> list[str]:
    """取历史中最近连续的工具调用名；遇到模型的纯文本回复即中断计数。"""
    names: list[str] = []
    for message in reversed(messages):
        if getattr(message, "type", "") != "ai":
            continue
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            break  # 模型已经输出过最终文本，重复调用的链条已中断
        for call in reversed(tool_calls):
            name = call.get("name") if isinstance(call, dict) else None
            if name:
                names.append(name)
        if len(names) >= limit:
            break
    names.reverse()
    return names[-limit:]


async def agent_node(state: AgentState) -> dict:
    """注入系统提示词，把对话历史交给 LLM，返回其响应（可能包含 tool_calls）。"""
    log.info("Agent 节点开始思考...")

    # ---- 物理熔断：连续两次调用同一工具时，剥夺本轮的工具权限 ----
    recent_tools = _recent_consecutive_tool_names(state["messages"])
    if len(recent_tools) == 2 and recent_tools[0] == recent_tools[1]:
        log.error(
            f"检测到大模型连续两次调用同一工具 {recent_tools[-1]}，"
            "已触发熔断：本轮不提供工具，强制输出文本总结。"
        )
        llm_with_tools = llm
    else:
        tools = await _get_tools()
        # 隐藏加载/探查工具，避免模型把“加载数据”当成需要自己决策的一步
        visible_tools = [t for t in tools if t.name not in HIDDEN_TOOL_NAMES]
        llm_with_tools = llm.bind_tools(visible_tools)  # 给大模型配发装备

    messages = _inject_system_prompt(state["messages"])
    response = await llm_with_tools.ainvoke(messages)
    # 清洗 <|DSML|> 等未解析控制符；若已有 tool_calls 则原样保留
    response = _sanitize_dsml_response(response)

    # ---- 强制单步截断：模型偶尔会并行吐出多个工具调用 ----
    tool_calls = getattr(response, "tool_calls", None) or []
    if len(tool_calls) > 1:
        log.warning("检测到大模型尝试并行调用多个工具，强制截断为只执行第一个。")
        response = _keep_first_tool_call(response)
        tool_calls = response.tool_calls

    if tool_calls:
        log.info(f"LLM 决定调用工具: {[t['name'] for t in tool_calls]}")
    else:
        log.info("LLM 未调用工具，生成最终回复")
    return {"messages": [response]}


def _parse_json(text: str) -> dict[str, Any] | None:
    """把工具返回文本解析为 JSON 对象；非 JSON / 非对象返回 None。"""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


# 常见的表格字段名，按优先级尝试
TABLE_FIELDS = ("candidates", "records", "rows", "data", "table", "df_seq", "df_pairs")


def _extract_table(payload: dict[str, Any]) -> list[dict] | None:
    """从工具返回的 JSON 中提取可作为表格展示的列表。"""
    for key in TABLE_FIELDS:
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return value
    # 简化兜底：任意非空的“字典列表”字段
    for value in payload.values():
        if isinstance(value, list) and value and all(
            isinstance(row, dict) for row in value
        ):
            return value
    return None


def _table_key(tool_name: str) -> str:
    """前端表格的键名：打分结果统一叫 rank，其余沿用工具名。"""
    return "rank" if tool_name == "tool_rank_candidates" else tool_name


def _extract_ids(text: str) -> dict[str, Any]:
    """从工具返回的 JSON 字符串中提取 dataset_id / pairs_dataset_id。"""
    payload = _parse_json(text)
    if payload is None:
        return {}
    # 精准挑出“取件码”
    ids: dict[str, Any] = {}
    if payload.get("dataset_id") is not None:
        ids["dataset_id"] = payload["dataset_id"]
    if payload.get("pairs_dataset_id") is not None:
        ids["pairs_dataset_id"] = payload["pairs_dataset_id"]
    return ids


def _build_load_args(state: AgentState) -> dict[str, Any]:
    """从 state 的原始输入构造 tool_load_sequences 参数。

    raw_input_path 与 raw_input_sequences 二选一；Excel 模式所需的列名
    可放在 state["metadata"] 的 seq_col / count_col 中。
    """
    load_args: dict[str, Any] = {}
    path = state.get("raw_input_path")
    sequences = state.get("raw_input_sequences")
    if path:
        load_args["path"] = path
    if sequences:
        load_args["sequences"] = list(sequences)
    metadata = state.get("metadata") or {}
    for key in ("seq_col", "count_col"):
        if metadata.get(key):
            load_args[key] = metadata[key]
    return load_args


async def _auto_load_dataset(state: AgentState, tool_map: dict[str, Any]) -> str:
    """隐式调用加载工具，返回真实 dataset_id；失败时抛出明确错误。"""
    load_tool = tool_map.get(LOAD_TOOL_NAME)
    if load_tool is None:
        raise ValueError(f"MCP 工具 {LOAD_TOOL_NAME} 不可用，无法自动加载数据。")

    load_args = _build_load_args(state)
    log.info(f"state 中缺少 dataset_id，自动调用 {LOAD_TOOL_NAME} 加载数据...")
    try:
        content = await load_tool.ainvoke(load_args)
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
    except Exception as exc:
        log.warning(f"自动加载数据失败：{exc}")
        raise ValueError(f"自动加载数据失败：{exc}") from exc

    dataset_id = _extract_ids(content).get("dataset_id")
    if not dataset_id:
        log.warning(f"自动加载未返回 dataset_id：{content[:200]}")
        raise ValueError("自动加载数据失败：加载工具未返回 dataset_id。")
    log.info(f"自动加载完成，dataset_id = {dataset_id}")
    return dataset_id


TOOL_SUCCESS_MESSAGE = "工具执行成功。请根据流程继续下一步，或直接输出文本向用户汇报。"


def _compact_tool_content(
    tool_name: str,
    dataset_id: str | None,
    pairs_dataset_id: str | None,
    payload: dict[str, Any] | None,
) -> str:
    """给 LLM 的精简工具结果：只保留状态、ID 与 meta，避免大表撑爆上下文。"""
    result: dict[str, Any] = {
        "status": "success",
        "tool_name": tool_name,
        "dataset_id": dataset_id,
        "message": TOOL_SUCCESS_MESSAGE,
    }
    if pairs_dataset_id:
        result["pairs_dataset_id"] = pairs_dataset_id
    if payload and payload.get("meta") is not None:
        result["meta"] = payload["meta"]
    return json.dumps(result, ensure_ascii=False)


def _ensure_structure_source(
    dataset_id: str | None,
    history_tables: dict[str, list[dict]],
) -> None:
    """结构预测前的数据源校验。

    Agent 进程无法直接读取 MCP Server 内存中的 DataFrame，因此这里通过
    history_tables 判断本次会话是否真的执行过清洗、且清洗结果包含
    variable_region 列。校验失败时直接抛出 ValueError，避免底层报出
    难以理解的 KeyError，也避免模型随后幻觉式地向用户索要 dataset_id。
    """
    if not dataset_id:
        raise ValueError(
            "结构预测失败：dataset_id 为空，无法获取序列数据。"
            "请先调用 tool_clean_sequences 完成数据清洗。"
        )

    cleaned_rows = history_tables.get("tool_clean_sequences") or []
    if not cleaned_rows:
        raise ValueError(
            "结构预测失败：当前数据尚未完成清洗，缺少 variable_region 列。"
            "请先调用 tool_clean_sequences。"
        )
    if not any("variable_region" in row for row in cleaned_rows):
        raise ValueError(
            "结构预测失败：清洗结果中不包含 variable_region 列，无法进行结构预测。"
            "请重新调用 tool_clean_sequences。"
        )


async def tool_node(state: AgentState) -> dict:
    """执行最后一条 AI 消息中的全部 tool_calls，并同步数据 ID。"""
    log.info("开始执行工具节点...")
    history_tables = state.get("history_tables", {}).copy()  # 浅拷贝，避免改动旧状态
    last_message = state["messages"][-1]  # 获取最新一条消息
    tool_calls = getattr(last_message, "tool_calls", None) or []  # 提取工具调用指令（安全获取）

    tools = await _get_tools()  # 从全局缓存/服务器拿回所有可用的工具
    tool_map = {tool.name: tool for tool in tools}  # 建立“花名册”字典映射

    # 沿用已有 ID；本次工具返回了新值时再覆盖
    dataset_id = state.get("dataset_id")
    pairs_dataset_id = state.get("pairs_dataset_id")

    # ---- 隐式前置步骤：没有 dataset_id 时自动加载 ----
    if not dataset_id:
        raw_input_path = state.get("raw_input_path")
        raw_input_sequences = state.get("raw_input_sequences")
        if not raw_input_path and not raw_input_sequences:
            raise ValueError(
                "缺少数据：state 中既没有 dataset_id，也没有 "
                "raw_input_path 或 raw_input_sequences。"
            )
        dataset_id = await _auto_load_dataset(state, tool_map)
        log.info(f"状态更新：自动加载得到 dataset_id = {dataset_id}")

    # LLM 按约定传入 "auto"，这里统一替换为真实 ID（同时覆盖任何编造值）
    for tool_call in tool_calls:
        args = tool_call.get("args")
        if not isinstance(args, dict):
            args = {}
            tool_call["args"] = args
        args["dataset_id"] = dataset_id

    tool_messages: list[ToolMessage] = []
    for call in tool_calls:
        name = call.get("name", "")
        args = call.get("args", {}) or {}
        call_id = call.get("id", "")
        log.info(f"正在执行工具: {name}, 参数: {args}")

        tool = tool_map.get(name)
        resp_json: dict[str, Any] | None = None

        if tool is None:
            content = f"错误：工具 {name} 不存在或尚未加载。"
        else:
            # 结构预测强依赖 variable_region；数据源不合格时直接抛错
            if name == "tool_compare_structures":
                _ensure_structure_source(dataset_id, history_tables)
            try:
                content = await tool.ainvoke(args)  # 这里底层会去请求 MCP Server，并把返回结果扒成纯文本。
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                resp_json = _parse_json(content)
            except Exception as exc:          # 工具失败也要回传，交给 LLM 决策
                content = f"错误：调用工具 {name} 失败：{exc}"

        # 把工具返回的表格数据留存给前端展示
        if resp_json is not None:
            table = _extract_table(resp_json)
            if table is not None:
                table_key = _table_key(name)
                history_tables[table_key] = table
                log.info(f"已保存表格 history_tables['{table_key}']：{len(table)} 行")

        ids = _extract_ids(content)
        if "dataset_id" in ids:
            dataset_id = ids["dataset_id"]
            log.info(f"状态更新：提取到 dataset_id = {dataset_id}")
        if "pairs_dataset_id" in ids:
            pairs_dataset_id = ids["pairs_dataset_id"]

        # 成功时只把精简结果交给 LLM；完整表格已存入 history_tables 供前端使用
        if resp_json is not None:
            content = _compact_tool_content(
                name, dataset_id, pairs_dataset_id, resp_json
            )

        tool_messages.append(
            ToolMessage(content=content, tool_call_id=call_id, name=name)
        )

    return {
        "messages": tool_messages,
        "dataset_id": dataset_id,
        "pairs_dataset_id": pairs_dataset_id,
        "history_tables": history_tables,
    }


def should_continue(state: AgentState) -> str:
    """最后一条消息带 tool_calls 则进入工具节点，否则结束。"""
    messages = state.get("messages") or []
    if not messages:
        return "end"
    return "tools" if getattr(messages[-1], "tool_calls", None) else "end"
