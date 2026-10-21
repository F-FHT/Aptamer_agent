# =============================================================================
# core/loader.py
# 职责：读取 Excel 测序结果文件，按调用方显式指定的列名输出标准化 DataFrame。
# 设计原则：
#   1) 不做"猜列"逻辑——列名由调用方显式传入。列识别属于语义判断，
#      后续交给 LangGraph 的 schema_infer 节点用 LLM 完成。
#   2) 只负责"读进来、对齐列、做基本类型清洗"，业务清洗交给 clean.py。
#   3) 返回 (DataFrame, meta_dict)。meta 是普通 dict，直接并入 State.metadata，
#      避免自定义类在 Checkpointer 序列化时出问题。
# 适用范围：仅 .xlsx / .xls。
# =============================================================================
from __future__ import annotations
from pathlib import Path
import pandas as pd
from observability.logger import get_logger
from observability.decorators import log_tool
log = get_logger(__name__)

class ColumnError(ValueError):
    """列名不存在或列配置异常时抛出。
    调用方捕获后应提示用户检查表头，或触发 schema_infer 重新推断。
    """

# -----------------------------------------------------------------------------
# 读取 Excel：
#   - 默认第一个 sheet，可通过 sheet_name 指定
#   - dtype=str 强制字符串读入，避免 pandas 把纯数字序列（如 "000111"）
#     自动推断成整数 111，这是测序数据读 Excel 最常见的坑
#   - 列名统一 strip()，防止用户表头带隐形空格导致后续找不到列
# -----------------------------------------------------------------------------
def _load_excel(path: Path, sheet_name: str | int) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, dtype=str)
    if df.empty:
        raise ValueError(f"工作表为空：{path} (sheet={sheet_name})")
    df.columns = [str(c).strip() for c in df.columns]
    return df


# -----------------------------------------------------------------------------
# 列名校验：显式指定的列必须存在，且序列列与 count 列不能相同。
# 失败时回显所有可选列名，方便用户或上游 Agent 修正。
# -----------------------------------------------------------------------------
def _validate_columns(df: pd.DataFrame, seq_col: str,
                      count_col: str | None) -> None:
    if seq_col not in df.columns:
        raise ColumnError(
            f"序列列 '{seq_col}' 不存在。可选列：{list(df.columns)}"
        )
    if count_col is not None and count_col not in df.columns:
        raise ColumnError(
            f"count 列 '{count_col}' 不存在。可选列：{list(df.columns)}"
        )
    if count_col is not None and count_col == seq_col:
        raise ColumnError("seq_col 与 count_col 不能是同一列")


# -----------------------------------------------------------------------------
# 统一入口：读 Excel → 校验列 → 输出标准化 DataFrame + 元信息 dict。
#
# 参数：
#   path        : Excel 文件路径
#   seq_col     : 序列列名（必填）
#   count_col   : 丰度列名（可选；None 表示无 count 列，按 1 处理）
#   sheet_name  : 工作表名或索引，默认第一个
#
# 返回：
#   (DataFrame, meta)
#   DataFrame 固定两列：raw_seq（str）、count（int）
#   meta 为普通 dict，可直接合并进 LangGraph State.metadata
# -----------------------------------------------------------------------------
@log_tool("load_sequences")
def load_sequences(path: str,
                   seq_col: str,
                   count_col: str | None = None,
                   sheet_name: str | int = 0
                   ) -> tuple[pd.DataFrame, dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    if p.suffix.lower() not in (".xlsx", ".xls"):
        raise ValueError(f"仅支持 .xlsx / .xls，收到：{p.suffix}")

    df = _load_excel(p, sheet_name)
    n_raw = len(df)

    _validate_columns(df, seq_col, count_col)

    # ---- 构造标准化输出 ----
    # 先 fillna("")：read_excel 会把空单元格读成 NaN，直接 astype(str) 会变成 "nan"
    out = pd.DataFrame({"raw_seq": df[seq_col].fillna("").astype(str).str.strip()})

    if count_col is not None:
        # count 容错：非数值填 0，负值截断为 0，统一转 int
        cnt = pd.to_numeric(df[count_col], errors="coerce").fillna(0)
        out["count"] = cnt.clip(lower=0).astype(int)
    else:
        out["count"] = 1                          # 无 count 列时统一按 1

    # 剔除空序列行（Excel 里常见末尾空行或整行空白）
    before = len(out)
    out = out[out["raw_seq"].str.len() > 0].reset_index(drop=True)
    dropped = before - len(out)

    # ---- 元信息：普通 dict，直接并入 State.metadata ----
    meta = {
        "source": str(p),
        "sheet_name": sheet_name,
        "seq_col": seq_col,
        "count_col": count_col,
        "n_rows_raw": n_raw,
        "n_rows_kept": len(out),
        "dropped_empty": dropped,
    }
    log.info(f"加载完成：{meta}")
    return out, meta