"""函数级追踪装饰器：@log_node 与 @log_tool。

两者共用同一个 `_instrument` 工厂，区别只在于记录的内容：
- @log_node：LangGraph 节点。记录开始/结束/耗时/异常（完整 traceback），
  并把节点名写入日志上下文，节点内部所有日志自动带 node 字段。
- @log_tool：MCP 工具。除上述信息外，额外记录截断后的入参和返回结果摘要，
  避免把整张 DataFrame 或长序列写进日志。

实现要点：
1. 同步函数与 async 函数都支持：装饰时用 inspect.iscoroutinefunction 判断，
   分别返回普通 wrapper 或 async wrapper。
2. 耗时用 time.perf_counter()：单调递增的高精度计时器，不受系统时间校准影响。
3. 上下文用 snapshot()/restore() 成对保护：被追踪函数内部可能再次 set_node，
   退出时恢复外层状态，避免污染调用链上层的 trace/node。
4. 参数截断：每个值最多保留 200 个字符；超大容器 / DataFrame 只记录长度或
   shape，防止日志体积失控，也避免 repr 大对象拖慢主流程。
"""
from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from observability.logger import (
    get_logger,
    get_node,
    restore,
    set_node,
    set_trace_id,
    snapshot,
)

F = TypeVar("F", bound=Callable[..., Any])

__all__ = ["log_node", "log_tool"]

# 单个参数值最多保留的字符数；超过后追加省略号
_PARAM_LIMIT = 200
# 容器元素超过该数量时只记录长度，不再展开内容
_CONTAINER_PREVIEW = 20


# =============================================================================
# 一、日志内容辅助函数
# =============================================================================
def _truncate(value: Any, limit: int = _PARAM_LIMIT) -> str:
    """把一个任意对象转成适合写日志的短字符串。

    处理顺序：
    1. DataFrame / ndarray 等带 shape 的对象：只记录类型和形状；
    2. 较大的 list / tuple / set / dict：只记录类型和长度；
    3. 其余对象用 str / repr，压缩空白后按 limit 截断。
    """
    # 带 shape 属性的对象通常是 pandas DataFrame / numpy ndarray，
    # repr 可能非常长，直接记录 shape 最经济。
    shape = getattr(value, "shape", None)
    if shape is not None and not isinstance(value, (str, bytes)):
        try:
            return f"<{type(value).__name__} shape={tuple(shape)}>"
        except TypeError:
            pass

    if isinstance(value, (list, tuple, set, frozenset, dict)):
        if len(value) > _CONTAINER_PREVIEW:
            return f"<{type(value).__name__} len={len(value)}>"
        text = repr(value)
    else:
        text = value if isinstance(value, str) else repr(value)

    # 把换行、连续空格压成单个空格，保证一条日志只占一行
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…(共 {len(text)} 字符)"


def _summarize(result: Any) -> str:
    """生成函数返回值的摘要：容器只报长度，标量截断后展示内容。"""
    if isinstance(result, (list, tuple, set, frozenset, dict)):
        return f"{type(result).__name__}(len={len(result)})"
    return _truncate(result)


def _bind_params(func: Callable[..., Any], args: tuple, kwargs: dict) -> dict[str, str]:
    """把实际传入的参数绑定到函数签名上，得到 {参数名: 截断后的值}。

    使用 bind_partial 而不是 bind：允许调用方省略有默认值的参数，也不会因为
    缺少参数而抛错（真正缺参数时会在执行函数时抛出，交给异常日志处理）。
    对于 C 扩展函数等拿不到签名的对象，直接返回空字典，不影响主流程。
    """
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return {}
    return {name: _truncate(value) for name, value in bound.arguments.items()}


def _inherit_trace_id(args: tuple, kwargs: dict) -> None:
    """尝试从调用参数中继承 trace_id。

    约定：如果入参里直接有 trace_id，或者某个参数是字典且包含 trace_id，
    就沿用这个值。这样 LangGraph 恢复执行、跨工具调用时仍是同一条链路。
    找不到时不设置，后续写日志时 ContextFilter 会自动生成一个新的。
    """
    candidates: list[Any] = [kwargs.get("trace_id")]
    for value in (*args, *kwargs.values()):
        if isinstance(value, Mapping):
            candidates.append(value.get("trace_id"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            set_trace_id(candidate)
            return


def _success_message(
    kind: str,
    name: str,
    elapsed_ms: float,
    params: dict[str, str] | None,
    result: Any,
) -> str:
    """拼装成功日志；params 为 None 表示该类型不记录入参（节点）。"""
    message = f"{kind} {name} 执行完成，耗时 {elapsed_ms:.2f}ms"
    if params is not None:
        message += f" | 入参: {params} | 返回: {_summarize(result)}"
    return message


def _error_message(
    kind: str,
    name: str,
    elapsed_ms: float,
    exc: BaseException,
    params: dict[str, str] | None,
) -> str:
    """拼装失败日志；堆栈通过 exc_info=True 由 logging 自动附加。"""
    message = (
        f"{kind} {name} 执行失败，耗时 {elapsed_ms:.2f}ms："
        f"{type(exc).__name__}: {exc}"
    )
    if params is not None:
        message += f" | 入参: {params}"
    return message


# =============================================================================
# 二、装饰器工厂
# =============================================================================
def _instrument(
    func: F,
    kind: str,
    name: str,
    with_params: bool,
) -> F:
    """为 func 套上“计时 + 上下文 + 成功/失败日志”的外壳。

    参数：
        func        : 被装饰的函数（同步或 async）。
        kind        : "node" 或 "tool"，只用于日志文案和 logger 名称。
        name        : 节点名 / 工具名。
        with_params : 是否记录入参与返回值摘要（工具为 True，节点为 False）。

    闭包说明：
        下面定义的 wrapper 会“记住”外层的 func / logger / kind / name，
        这些变量在装饰时确定，调用时无需再次计算。
    """
    logger = get_logger(f"{kind}.{name}")

    def _prepare(args: tuple, kwargs: dict) -> tuple[tuple, dict[str, str] | None]:
        """进入函数前的公共准备工作，返回 (上下文快照, 参数日志)。"""
        previous = snapshot()
        # 节点：node 就是节点名；
        # 工具：沿用外层节点名（这样工具日志能看出属于哪个流程节点），
        #      如果没有外层节点，则退化为工具名。
        set_node(name if kind == "node" else (get_node() or name))
        _inherit_trace_id(args, kwargs)
        # 工具记录入参，节点不记录（节点的 state 可能很大且包含完整对话历史）
        params = _bind_params(func, args, kwargs) if with_params else None
        return previous, params

    # ---- async 版本 ----
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            previous, params = _prepare(args, kwargs)
            started = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
            except Exception as exc:  # 记录完整堆栈后原样抛出，不吞异常
                elapsed_ms = (time.perf_counter() - started) * 1000
                logger.error(
                    _error_message(kind, name, elapsed_ms, exc, params),
                    exc_info=True,
                )
                raise
            else:
                elapsed_ms = (time.perf_counter() - started) * 1000
                logger.info(_success_message(kind, name, elapsed_ms, params, result))
                return result
            finally:
                # 无论成功失败都恢复外层上下文，避免 node / trace 泄漏
                restore(previous)

        return async_wrapper  # type: ignore[return-value]

    # ---- 同步版本 ----
    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        previous, params = _prepare(args, kwargs)
        started = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.error(
                _error_message(kind, name, elapsed_ms, exc, params),
                exc_info=True,
            )
            raise
        else:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(_success_message(kind, name, elapsed_ms, params, result))
            return result
        finally:
            restore(previous)

    return sync_wrapper  # type: ignore[return-value]


def log_node(node_name: str) -> Callable[[F], F]:
    """LangGraph 节点追踪装饰器。

    用法：
        @log_node("clean_node")
        async def clean_node(state): ...

    效果：节点内部的所有日志自动带上 node=clean_node；结束时输出耗时，
    异常时输出完整 traceback。
    """

    def decorator(func: F) -> F:
        return _instrument(func, "node", node_name, with_params=False)

    return decorator


def log_tool(tool_name: str) -> Callable[[F], F]:
    """MCP 工具追踪装饰器。

    用法：
        @log_tool("clean_sequences")
        def clean_sequences(...): ...

    效果：在节点日志的基础上，额外记录截断后的入参和返回值摘要。
    """

    def decorator(func: F) -> F:
        return _instrument(func, "tool", tool_name, with_params=True)

    return decorator
