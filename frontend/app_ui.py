# frontend/app_ui.py
# 职责：Streamlit 多轮聊天前端。维护对话历史与固定 thread_id，驱动 LangGraph
#      智能体，并实时展示模型文字与工具返回的多张表格。
# 数据源：上传 Excel 或 粘贴序列（二选一）；Excel 会落盘到临时目录后以
#        raw_input_path 传给图，序列则以 raw_input_sequences 传入。
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage

from agent import mcp_client
from agent.graph import app as graph_app
from observability.logger import get_logger

log = get_logger(__name__)

MAX_RESUMES = 20  # 工具节点后会中断等待；UI 自动恢复，最多恢复这么多轮

# history_tables 的键 -> 前端占位符
TABLE_TARGETS = {
    "clean": ("tool_clean_sequences",),
    "cluster": ("tool_cluster_sequences",),
    "motif": ("tool_discover_motifs",),
    "structure": ("tool_compare_structures",),
    "rank": ("rank", "tool_rank_candidates"),
}

# Excel 列名猜测：命中即作为默认选中项
SEQ_COL_KEYWORDS = ("sequence", "seq", "序列", "可变区", "region")
COUNT_COL_KEYWORDS = ("count", "reads", "abundance", "丰度", "频率", "拷贝")


def _message_text(message: Any) -> str:
    """安全提取消息文本（兼容 str 与 content blocks 两种形式）。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
        return "".join(parts)
    return str(content) if content else ""


def _read_excel_columns(uploaded: Any) -> list[str]:
    """只读取上传 Excel 的表头列名；失败时返回空列表。"""
    try:
        uploaded.seek(0)
        header = pd.read_excel(uploaded, nrows=0)
        return [str(col) for col in header.columns]
    except Exception as exc:
        log.warning(f"读取 Excel 表头失败：{exc}")
        return []
    finally:
        uploaded.seek(0)


def _guess_column(columns: list[str], keywords: tuple[str, ...]) -> int | None:
    """按关键字猜测列下标；没有命中返回 None。"""
    for index, name in enumerate(columns):
        lowered = name.lower()
        if any(keyword in lowered for keyword in keywords):
            return index
    return None


def _cleanup_upload() -> None:
    """删除本次会话写入的临时上传目录。"""
    upload_dir = st.session_state.get("upload_dir")
    if upload_dir and os.path.isdir(upload_dir):
        shutil.rmtree(upload_dir, ignore_errors=True)
        log.info(f"已清理临时上传目录: {upload_dir}")
    st.session_state.upload_dir = None
    st.session_state.uploaded_path = None


def _save_upload(uploaded: Any) -> str:
    """把上传文件写入临时目录，返回绝对路径。"""
    _cleanup_upload()  # 先清掉上一次的临时文件，避免堆积
    upload_dir = tempfile.mkdtemp(prefix="aptamer_upload_")
    file_name = Path(uploaded.name).name or "upload.xlsx"
    path = Path(upload_dir) / file_name
    path.write_bytes(uploaded.getbuffer())
    uploaded.seek(0)
    st.session_state.upload_dir = str(upload_dir)
    resolved = str(path.resolve())
    log.info(f"Excel 已保存到临时路径: {resolved}")
    return resolved


def _render_tables(tables: dict[str, list[dict]], placeholders: dict[str, Any]) -> None:
    """把 history_tables 中的数据渲染到对应标签页。"""
    for target, keys in TABLE_TARGETS.items():
        placeholder = placeholders[target]
        table = next((tables[k] for k in keys if k in tables), None)
        if table is None:
            placeholder.info("暂无数据")
            continue
        df = pd.DataFrame(table)
        if df.empty:
            placeholder.info("该步骤返回了空表。")
        else:
            placeholder.dataframe(df, use_container_width=True)


def _apply_event(
    event: dict[str, Any],
    tables: dict[str, list[dict]],
    placeholders: dict[str, Any],
    live_ph: Any,
    run_state: dict[str, Any],
) -> None:
    """处理 LangGraph 的一个 update 事件。"""
    for node, output in (event or {}).items():
        if not isinstance(output, dict):
            continue

        # agent 节点：实时刷新当前回复气泡，并记住最后一段文字
        if node == "agent":
            for message in output.get("messages", []) or []:
                text = _message_text(message).strip()
                if text:
                    run_state["last_agent_text"] = text
                    live_ph.markdown(text)

        # tool_node 会把工具返回的表格放进 history_tables
        incoming = output.get("history_tables")
        if incoming:
            tables.update(incoming)
            st.session_state.history_tables = tables
            _render_tables(tables, placeholders)


async def _run_graph(
    payload: dict[str, Any] | None,
    thread_id: str,
    on_update: Any,
) -> None:
    """运行编译后的 LangGraph；每次工具节点中断后自动恢复。"""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        for _ in range(MAX_RESUMES):
            log.info(f"开始执行图，当前轮次: {_ + 1}")
            async for event in graph_app.astream(
                payload, config=config, stream_mode="updates"
            ):
                on_update(event)

            payload = None  # 续跑：从 checkpoint 恢复
            snapshot = await graph_app.aget_state(config)
            log.info(f"当前图状态 Next: {snapshot.next}")
            if not snapshot.next:
                return

        log.error("图执行超过最大恢复次数，强制终止。")
        raise RuntimeError(f"分析步骤超过 {MAX_RESUMES} 轮，已停止。")
    finally:
        # 每次 Streamlit 运行使用新的事件循环，结束前关闭 MCP 会话并清空工具缓存，
        # 避免下一次对话复用到已绑定旧事件循环的会话。
        await mcp_client.close_session()
        mcp_client.tools_cache.clear()


# =============================================================================
# 页面与会话初始化
# =============================================================================
st.set_page_config(page_title="Aptamer Agent", page_icon="🧬", layout="wide")
st.title("🧬 Aptamer Agent 智能分析")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages_history" not in st.session_state:
    st.session_state.messages_history = []
if "history_tables" not in st.session_state:
    st.session_state.history_tables = {}
if "sequences_sent" not in st.session_state:
    st.session_state.sequences_sent = False
if "uploader_nonce" not in st.session_state:
    st.session_state.uploader_nonce = 0  # 重置时递增，强制换一个新的 file_uploader
if "upload_dir" not in st.session_state:
    st.session_state.upload_dir = None
if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None

with st.sidebar:
    st.header("初始数据")
    st.caption("上传 Excel 或 粘贴序列（二选一）。上传 Excel 后，粘贴框会被忽略。")

    uploaded = st.file_uploader(
        "上传 Excel 文件（.xlsx / .xls）",
        type=["xlsx", "xls"],
        key=f"excel_upload_{st.session_state.uploader_nonce}",
    )

    seq_col: str | None = None
    count_col: str | None = None
    if uploaded is not None:
        columns = _read_excel_columns(uploaded)
        if columns:
            seq_guess = _guess_column(columns, SEQ_COL_KEYWORDS)
            seq_col = st.selectbox(
                "序列表头（seq_col）",
                columns,
                index=seq_guess if seq_guess is not None else 0,
            )
            count_options = ["(不使用)"] + columns
            count_guess = _guess_column(columns, COUNT_COL_KEYWORDS)
            count_choice = st.selectbox(
                "丰度表头（count_col，可选）",
                count_options,
                index=(count_guess + 1) if count_guess is not None else 0,
            )
            count_col = None if count_choice == "(不使用)" else count_choice
        else:
            st.error("无法读取该 Excel 的表头，请确认文件未加密且格式正确。")
            seq_col = st.text_input("序列表头（seq_col）").strip() or None
            count_col = st.text_input("丰度表头（count_col，可选）").strip() or None

    seq_text = st.text_area(
        "序列（每行一条）",
        height=240,
        placeholder="ATGCGTACGT...\nGGCTAACGTT...",
        disabled=uploaded is not None,
    )
    if uploaded is not None:
        st.caption("已启用 Excel 数据源，粘贴框中的序列不会发送。")
    else:
        st.caption("序列只在首轮对话时发送；修改序列后请重置对话。")

    if st.button("重置对话", use_container_width=True):
        _cleanup_upload()
        st.session_state.uploader_nonce += 1  # 换 key -> file_uploader 被清空
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages_history = []
        st.session_state.history_tables = {}
        st.session_state.sequences_sent = False
        log.info("用户点击重置对话，已清空历史、上传缓存并生成新的 thread_id。")
        st.rerun()

    st.caption(f"当前会话：{st.session_state.thread_id[:8]}")
    st.caption("运行前请确保 MCP Server 已启动。")

# =============================================================================
# 聊天区（先占位，便于后续把新消息写回历史下方）
# =============================================================================
chat_area = st.container()
with chat_area:
    if not st.session_state.messages_history:
        st.caption("请先在左侧上传 Excel 或粘贴序列，然后在底部输入分析指令。")
    for message in st.session_state.messages_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# =============================================================================
# 结果标签页
# =============================================================================
with st.container():
    tab_clean, tab_motif, tab_struct, tab_rank = st.tabs(
        ["清洗与聚类", "Motif 分析", "二级结构对比", "最终候选推荐"]
    )

    with tab_clean:
        st.caption("清洗结果")
        ph_clean = st.empty()
        st.caption("聚类结果")
        ph_cluster = st.empty()

    with tab_motif:
        st.caption("Motif 发现结果")
        ph_motif = st.empty()

    with tab_struct:
        st.caption("结构对比结果")
        ph_structure = st.empty()

    with tab_rank:
        st.caption("最终候选推荐")
        ph_rank = st.empty()

    placeholders = {
        "clean": ph_clean,
        "cluster": ph_cluster,
        "motif": ph_motif,
        "structure": ph_structure,
        "rank": ph_rank,
    }

    # 页面刷新后回显已有表格
    _render_tables(st.session_state.history_tables, placeholders)

# =============================================================================
# 对话交互
# =============================================================================
prompt = st.chat_input("请输入您的分析指令...")

if prompt:
    # 1) 记录并展示用户消息
    st.session_state.messages_history.append({"role": "user", "content": prompt})
    with chat_area:
        with st.chat_message("user"):
            st.markdown(prompt)

    use_excel = uploaded is not None
    sequences = [line.strip() for line in seq_text.splitlines() if line.strip()]
    first_turn = not st.session_state.sequences_sent

    # 首轮必须有一个可用的数据源：Excel 文件（且选好序列表头）或粘贴序列
    if first_turn and use_excel and not seq_col:
        message = "请先在左侧选择 Excel 的序列表头（seq_col），然后再开始分析。"
        log.warning("首轮使用 Excel 但未选择 seq_col，已提示用户补充。")
        with chat_area:
            with st.chat_message("assistant"):
                st.warning(message)
        st.session_state.messages_history.append(
            {"role": "assistant", "content": message}
        )
    elif first_turn and not use_excel and not sequences:
        message = "请先在左侧上传 Excel，或粘贴初始序列（每行一条），然后再开始分析。"
        log.warning("首轮对话缺少数据源，已提示用户补充。")
        with chat_area:
            with st.chat_message("assistant"):
                st.warning(message)
        st.session_state.messages_history.append(
            {"role": "assistant", "content": message}
        )
    else:
        source = f"Excel: {uploaded.name}" if use_excel else f"粘贴序列: {len(sequences)} 条"
        log.info(f"用户触发分析。数据源: {source}, 指令: {prompt}")
        log.info(f"当前 thread_id: {st.session_state.thread_id}, 首轮: {first_turn}")

        # 沿用已积累的表格；工具节点会返回合并后的 history_tables
        tables: dict[str, list[dict]] = dict(st.session_state.history_tables)
        run_state: dict[str, Any] = {"last_agent_text": ""}

        # 首轮才传原始数据；后续靠 checkpointer 中的状态续接
        payload: dict[str, Any] = {"messages": [HumanMessage(content=prompt)]}
        if first_turn:
            if use_excel:
                upload_path = _save_upload(uploaded)
                st.session_state.uploaded_path = upload_path
                payload["raw_input_path"] = upload_path
                payload["metadata"] = {
                    "seq_col": seq_col,
                    "count_col": count_col,
                }
                log.info(
                    f"首轮数据源为 Excel：path={upload_path}, "
                    f"seq_col={seq_col}, count_col={count_col}"
                )
            else:
                payload["raw_input_sequences"] = sequences
                log.info(f"首轮数据源为粘贴序列，共 {len(sequences)} 条。")

        with chat_area:
            with st.chat_message("assistant"):
                live_ph = st.empty()
                live_ph.markdown("_正在连接模型与 MCP Server..._")

                def _on_update(event: dict[str, Any]) -> None:
                    _apply_event(event, tables, placeholders, live_ph, run_state)

                try:
                    asyncio.run(
                        _run_graph(payload, st.session_state.thread_id, _on_update)
                    )
                    final_text = (
                        run_state["last_agent_text"].strip()
                        or "分析已完成，但模型没有返回文字回复。"
                    )
                    live_ph.markdown(final_text)
                    st.session_state.messages_history.append(
                        {"role": "assistant", "content": final_text}
                    )
                    st.session_state.history_tables = tables
                    st.session_state.sequences_sent = True
                except Exception as exc:  # 加载失败 / MCP 未启动 / LLM 报错等
                    log.error(f"前端执行分析失败: {exc}", exc_info=True)
                    st.session_state.history_tables = tables
                    st.error(f"分析失败：{exc}")
                    with st.expander("查看异常堆栈"):
                        st.exception(exc)
