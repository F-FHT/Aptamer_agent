"""aptamer-agent 可观测性基础设施（极简版）。

包内只保留三个文件：
    logger.py      同步日志配置、JSON / 控制台格式化器、trace 上下文变量
    decorators.py  @log_node / @log_tool 函数级追踪
    __init__.py    对外统一导出

典型用法：
    from observability import get_logger, log_tool

    log = get_logger("core.clean")
    log.info("开始清洗")   # 自动带上 trace_id / node 字段
"""
from observability.decorators import log_node, log_tool
from observability.logger import (
    ConsoleFormatter,
    ContextFilter,
    JsonFormatter,
    get_logger,
    get_node,
    get_trace_id,
    new_trace_id,
    restore,
    set_node,
    set_trace_id,
    setup_logging,
    snapshot,
)

__version__ = "0.2.0"

__all__ = [
    # 日志
    "get_logger",
    "setup_logging",
    "JsonFormatter",
    "ConsoleFormatter",
    "ContextFilter",
    # 上下文
    "new_trace_id",
    "set_trace_id",
    "get_trace_id",
    "set_node",
    "get_node",
    "snapshot",
    "restore",
    # 装饰器
    "log_node",
    "log_tool",
]
