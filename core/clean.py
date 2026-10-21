# 职责：对 loader.py 输出的标准化 DataFrame 做业务清洗：
#      引物去除 → 可变区截取 → 长度过滤 → 去重合并 → TopN 提取。
# 输入约定：DataFrame 必须含 raw_seq 和 count 两列（由 loader.py 保证）。
# 输出约定：保留 raw_seq、variable_region、count 三列，按 count 降序。
#           raw_seq 始终是“全长序列”：输入已是全长则原样保留；
#           输入是截断序列（找不到引物）时按 fwd + seq + rev 补齐全长。
# 本模块不依赖文件来源，可直接用构造的 DataFrame 做单元测试。
from __future__ import annotations
import re
import pandas as pd
from observability.logger import get_logger
from observability.decorators import log_tool

log = get_logger(__name__)

# -----------------------------------------------------------------------------
# 从单条序列中截取正反向引物之间的可变区。
# 逻辑：
#   1) 定位正向引物，跳过引物本身
#   2) 从正向引物之后定位反向引物
#   3) 返回两者之间的子串
# 任一引物未找到 → 返回 None（由调用方决定补齐策略）。
# 注意：此处假设反向引物以"正向序列形式"出现在 read 中；
#      若测序方向相反，需由调用方先做反向互补。
# -----------------------------------------------------------------------------
def extract_variable_region(seq: str, fwd: str, rev: str) -> str | None:
    start = seq.find(fwd)
    if start == -1:
        return None
    start += len(fwd)                         # 跳过正向引物
    end = seq.rfind(rev, start)               # 取最后一次出现，即读长末端的反向引物
    return None if end == -1 else seq[start:end]


def _as_text(value: object) -> str:
    """把单元格值安全转为字符串；缺失值统一视为空串。"""
    return "" if pd.isna(value) else str(value)


def _sanitize_seq(seq: str) -> str | None:
    """净化单条序列：大写化、去除空白与连字符，并校验字符集。

    只允许 A / C / G / T / U / N。小写字母会被统一转为大写；空格、制表符、
    换行和连字符（常见于测序导出或人工排版）会被直接删除。若清洗后为空串，
    或仍包含简并碱基（R/Y/M/K 等）及其他符号，则返回 None，由调用方剔除。
    """
    cleaned = re.sub(r"[\s-]+", "", _as_text(seq).upper())
    if not cleaned:
        return None
    # fullmatch 要求整个字符串都匹配，避免只校验开头而漏掉尾部非法字符
    return cleaned if re.fullmatch(r"[ACGTUN]+", cleaned) else None


def _prepare_sequence(seq: object, fwd: str, rev: str) -> tuple[str, str]:
    """返回 (raw_seq, variable_region)，实现“防御性补齐”契约。

    - 未提供任一引物：原序列同时作为 raw_seq 与 variable_region；
    - 引物齐全且能正常截取：raw_seq 保留原序列，variable_region 为中间区；
    - 找不到引物（说明用户传的是截断序列）：variable_region 用原序列，
      raw_seq 构造为 fwd_primer + 原序列 + rev_primer。
    """
    seq = _as_text(seq)
    if not fwd or not rev:
        return seq, seq

    region = extract_variable_region(seq, fwd, rev)
    if region is None:
        return fwd + seq + rev, seq
    return seq, region


def _infer_target_len(regions: pd.Series) -> int:
    """按可变区长度中位数推断目标长度；无法推断时给出明确错误。

    不同数据源（手动粘贴的短序列 / Excel 长表）长度差异很大，若沿用固定的
    40 nt，很容易把全部序列过滤掉。这里以实际数据的中位数作为基准。
    """
    lengths = regions.map(lambda s: len(_as_text(s)))
    lengths = lengths[lengths > 0]
    if lengths.empty:
        raise ValueError("无法自动推断 target_len：没有有效的序列长度。")
    inferred = int(round(float(lengths.median())))
    return max(inferred, 1)


# -----------------------------------------------------------------------------
# 清洗主流程：
#   0) 序列净化：大写化、去空白/连字符，只保留 ACGTUN，非法序列直接剔除
#   1) 防御性补齐：区分“全长读长”与“用户传入的截断序列”，保证 raw_seq 为全长
#   2) 确定目标长度：target_len <= 0 或 None 时按 variable_region 长度的中位数
#      自动推断；否则使用用户显式传入的值
#   3) 长度过滤：针对 variable_region，target_len ± tolerance（如 80 → 64~96 nt）
#   4) 相同可变区合并，count 累加（去重），raw_seq 保留首次出现的值
#   5) 按 count 降序取前 top_n 条
# 实际使用的目标长度写入 df.attrs["target_len"]，由上层组装进 meta。
# 每步写日志，便于在 LangGraph trace 中查看过滤损耗。
# -----------------------------------------------------------------------------
@log_tool("clean_sequences")
def clean_sequences(df: pd.DataFrame,
                    fwd_primer: str,
                    rev_primer: str,
                    target_len: int | None = 40,
                    tolerance: float = 0.2,
                    top_n: int = 50) -> pd.DataFrame:
    if df.empty:
        raise ValueError("输入 DataFrame 为空，无可清洗的序列")

    df = df.copy()

    # ---- 0) 序列净化：必须在引物截取和长度过滤之前完成 ----
    n_before_sanitize = len(df)
    df["raw_seq"] = df["raw_seq"].map(_sanitize_seq)
    df = df.dropna(subset=["raw_seq"]).reset_index(drop=True)
    n_dropped = n_before_sanitize - len(df)
    log.warning(f"剔除了 {n_dropped} 条包含非法字符的序列")
    if df.empty:
        raise ValueError(
            "序列净化后 DataFrame 为空：所有序列均包含非法字符或被清洗为空串"
        )

    fwd, rev = fwd_primer or "", rev_primer or ""

    # ---- 1) 防御性补齐 + 截取可变区 ----
    originals = list(df["raw_seq"])
    prepared = [_prepare_sequence(s, fwd, rev) for s in originals]
    df["raw_seq"] = [p[0] for p in prepared]          # 全长（必要时补齐）
    df["variable_region"] = [p[1] for p in prepared]  # 去引物后的核心区

    if fwd and rev:
        n_padded = sum(
            1 for orig, (raw, _) in zip(originals, prepared)
            if raw != _as_text(orig)
        )
        log.info(f"引物截取：正常截取 {len(df) - n_padded} 条，"
                 f"补齐全长 {n_padded} 条")
    else:
        log.info("未提供完整引物对：raw_seq 与 variable_region 均使用原序列")

    # ---- 2) 确定目标长度：未指定时按实际数据自动推断 ----
    # 以 variable_region 为准：长度过滤作用在这一列上，用 raw_seq（可能含引物）
    # 推断会把基准抬高 len(fwd) + len(rev)。
    if target_len is None or target_len <= 0:
        real_target_len = _infer_target_len(df["variable_region"])
        target_len_source = "auto"
        log.info(
            f"未指定 target_len（{target_len}），按可变区长度中位数自动推断为 "
            f"{real_target_len} nt"
        )
    else:
        real_target_len = int(target_len)
        target_len_source = "user"
        log.info(f"使用用户指定的 target_len = {real_target_len} nt")

    # ---- 3) 长度过滤（针对 variable_region）----
    lo = int(real_target_len * (1 - tolerance))
    hi = int(real_target_len * (1 + tolerance))
    n_before = len(df)
    df = df[df["variable_region"].str.len().between(lo, hi)]
    log.info(f"长度过滤({lo}-{hi}nt)：保留 {len(df)}/{n_before} 条")

    # ---- 4) 去重合并 + 5) 排序取 TopN ----
    # 相同可变区可能来自不同读长（两侧 flank 不同），这里保留首次出现的 raw_seq 作为代表
    df = (df.groupby("variable_region", as_index=False)
            .agg(raw_seq=("raw_seq", "first"), count=("count", "sum"))
            .sort_values("count", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
            .loc[:, ["raw_seq", "variable_region", "count"]])

    # 把本次实际使用的长度参数带给上层（MCP 工具会写进 meta）
    df.attrs["target_len"] = real_target_len
    df.attrs["target_len_source"] = target_len_source
    df.attrs["length_range"] = (lo, hi)

    log.info(f"清洗后保留 {len(df)} 条，read count 范围 "
             f"{df['count'].min()}–{df['count'].max()}")
    return df
