"""极简日志模块：同步输出 + JSON 文件 + 彩色控制台 + contextvars 上下文。

设计目标（个人开源项目）：
- 只用标准库 logging，不引入队列、监听线程等异步机制，行为直观、易调试；
- 日志写入 logs/app.log（单行 JSON），控制台输出适合人眼阅读的彩色文本；
- trace_id / node 通过 contextvars 自动在同步函数、协程之间传递，业务代码
  不需要手动把 trace_id 一路当参数传下去。

contextvars 原理简述：
    每个线程 / 每个 asyncio 任务都持有一份独立的 Context。ContextVar.set()
    只修改“当前上下文”里的值，不会影响其他线程或其他协程。因此同一个进程里
    并发处理多个请求时，各自设置的 trace_id 互不串扰；而同一个调用链内部
    （包括被它 await 的协程）读取到的都是同一个值。这正是传统 thread-local
    无法兼顾协程场景的地方。
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import threading
import uuid
from datetime import datetime
from typing import Any

# =============================================================================
# 一、上下文变量：trace_id 与 node_name
# =============================================================================
# 两个 ContextVar 都是“当前上下文的局部变量”，默认值为 None。
# 读取时需要判断是否为空，再决定是否生成新的 trace_id。
_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_id", default=None
)
_node_name: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "node_name", default=None
)


def new_trace_id() -> str:
    """生成 16 位短 trace_id：取 uuid4 十六进制的前 16 位。

    长度足够区分一次请求，同时让日志行更短、更易读。
    """
    return uuid.uuid4().hex[:16]


def set_trace_id(trace_id: str | None = None) -> str:
    """设置当前上下文的 trace_id；不传则自动生成。返回最终生效的值。"""
    tid = trace_id or new_trace_id()
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    """读取当前上下文的 trace_id；缺失时自动生成并写回上下文。

    这样即使调用方忘了在入口处 set_trace_id，每条日志也一定带有 trace_id。
    """
    tid = _trace_id.get()
    if not tid:
        tid = set_trace_id()
    return tid


def set_node(node_name: str | None) -> None:
    """设置当前节点名（LangGraph 节点名或 MCP 工具名）。"""
    _node_name.set(node_name)


def get_node() -> str:
    """读取当前节点名；未设置时返回空串，便于直接写进日志字段。"""
    return _node_name.get() or ""


def snapshot() -> tuple[str | None, str | None]:
    """保存当前上下文的 (trace_id, node)。

    装饰器在进入被追踪函数前调用它，退出时用 restore() 还原，
    避免内层函数修改了上下文、返回后污染外层调用链。
    """
    return _trace_id.get(), _node_name.get()


def restore(state: tuple[str | None, str | None]) -> None:
    """把上下文恢复到 snapshot() 保存的状态。"""
    _trace_id.set(state[0])
    _node_name.set(state[1])


# =============================================================================
# 二、日志格式化器
# =============================================================================
# 各日志级别的控制台颜色（ANSI 转义码）。只在终端支持颜色时使用。
_LEVEL_COLORS = {
    "DEBUG": "\033[36m",      # 青色
    "INFO": "\033[32m",       # 绿色
    "WARNING": "\033[33m",    # 黄色
    "ERROR": "\033[31m",      # 红色
    "CRITICAL": "\033[1;41m", # 粗体 + 红底
}
_RESET = "\033[0m"


def _supports_color() -> bool:
    """判断当前标准错误流是否适合输出 ANSI 颜色。

    - 输出被重定向到文件 / IDE 非终端时禁用颜色，避免日志里出现转义码；
    - Windows 旧版控制台需要显式开启 VT 模式，失败则退回无色输出。
    """
    if not getattr(sys.stderr, "isatty", lambda: False)():
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            # -12 是 STD_ERROR_HANDLE；7 = ENABLE_PROCESSED_OUTPUT |
            # ENABLE_WRAP_AT_EOL_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-12), 7)
        except Exception:  # noqa: BLE001 - 开启失败不影响日志功能
            return False
    return True


class ContextFilter(logging.Filter):
    """在日志产生时把上下文里的 trace_id / node 注入 LogRecord。

    格式化器只负责“怎么显示”，过滤器负责“带上哪些字段”。这样业务代码写
    logger.info("...") 即可，无需每处都传 extra={"trace_id": ...}。
    如果调用方已通过 extra 显式指定了 trace_id / node，则尊重显式值。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "trace_id", None):
            record.trace_id = get_trace_id()
        if not getattr(record, "node", None):
            record.node = get_node()
        return True


class JsonFormatter(logging.Formatter):
    """单行 JSON 格式化器：适合写入文件、后续用脚本或日志平台采集。

    固定字段：timestamp / level / logger / message / trace_id / node / location。
    只有发生异常时才额外追加 exception（完整堆栈），保证正常日志足够精简。
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            # 带时区的本地时间，精确到毫秒，便于跨机器对齐
            "timestamp": datetime.fromtimestamp(record.created)
            .astimezone()
            .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", None) or "-",
            "node": getattr(record, "node", None) or "",
            # 精确到“哪个文件的哪一行”，排查问题时比只记模块名更有用
            "location": f"{record.pathname}:{record.lineno}",
        }
        if record.exc_info:
            # formatException 会输出完整 traceback；换行在 JSON 字符串里会被转义，
            # 所以整条日志仍然是“单行 JSON”。
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """控制台格式化器：时间 + 彩色级别 + logger 名 + 上下文 + 消息。

    输出示例：
        12:00:00 INFO     core.clean [trace=8f3a... node=clean_sequences] 清洗完成
    """

    def __init__(self, color: bool | None = None) -> None:
        super().__init__()
        # 传 None 表示自动判断；测试或特殊场景可以强制开 / 关
        self._color = _supports_color() if color is None else color

    def format(self, record: logging.LogRecord) -> str:
        # 级别左对齐到 8 个字符，让不同级别的日志起始列保持一致
        level = f"{record.levelname:<8}"
        if self._color:
            level = f"{_LEVEL_COLORS.get(record.levelname, '')}{level}{_RESET}"

        line = (
            f"{datetime.fromtimestamp(record.created):%H:%M:%S} {level} {record.name} "
            f"[trace={getattr(record, 'trace_id', '-')} "
            f"node={getattr(record, 'node', '')}] {record.getMessage()}"
        )
        # 异常堆栈单独换行追加，控制台可读性更好
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


# =============================================================================
# 三、全局初始化：控制台 + 文件双 handler（同步输出）
# =============================================================================
_CONFIGURED = False          # 是否已经配置过 root logger
_CONFIG_LOCK = threading.Lock()  # 保证多线程首次调用时只配置一次


def setup_logging(level: str | None = None, force: bool = False) -> None:
    """初始化根 logger（幂等）。

    参数：
        level : 日志级别，优先级：本参数 > 环境变量 LOG_LEVEL > INFO。
        force : 为 True 时先移除已有 handler 再重新配置，主要用于测试。
    """
    global _CONFIGURED
    with _CONFIG_LOCK:
        if _CONFIGURED and not force:
            return

        root = logging.getLogger()
        # 清理已有 handler，避免重复输出（例如测试中多次调用 setup_logging）
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

        # ---- 解析配置 ----
        level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
        level_no = logging.getLevelName(level_name)
        if not isinstance(level_no, int):  # 非法级别名时退回 INFO
            level_no = logging.INFO
        log_dir = os.getenv("LOG_DIR", "logs")
        log_file = os.path.join(log_dir, "app.log")

        # 日志目录可能不存在（全新克隆的项目），这里按需创建
        os.makedirs(log_dir, exist_ok=True)

        # ---- 控制台 handler：彩色文本，写给开发者在终端看 ----
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(ConsoleFormatter())
        console.addFilter(ContextFilter())
        console.setLevel(level_no)

        # ---- 文件 handler：单行 JSON，追加写入 logs/app.log ----
        # 使用普通 FileHandler 而非轮转/队列，保持实现最简单；
        # 每次 emit 后 logging 会自动 flush，异常退出最多丢失极小一段缓冲。
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(ContextFilter())
        file_handler.setLevel(level_no)

        root.addHandler(console)
        root.addHandler(file_handler)
        root.setLevel(level_no)

        # 压掉常见第三方库的噪音日志，避免 DEBUG 级别的网络细节淹没业务日志
        for name in ("httpx", "httpcore", "urllib3", "asyncio", "mcp"):
            logging.getLogger(name).setLevel(max(level_no, logging.WARNING))

        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 logger。

    首次调用时会自动完成日志系统初始化（幂等），因此各模块只需：
        log = get_logger(__name__)
    """
    setup_logging()
    return logging.getLogger(name)


__all__ = [
    "JsonFormatter",
    "ConsoleFormatter",
    "ContextFilter",
    "get_logger",
    "setup_logging",
    "new_trace_id",
    "set_trace_id",
    "get_trace_id",
    "set_node",
    "get_node",
    "snapshot",
    "restore",
]
