# core/rank.py
# 职责：对候选序列做多维度加权评分，并在分组维度上择优，
#      保证最终推荐结果具备多样性（每个家族/分组最多 1 条）。
# 设计原则：
#   1) 自适应降级：缺失的特征列或 df_pairs 直接剔除，权重重新归一化，不报错；
#   2) 分组择优：存在 group_id / cluster_id 时按组取最高分，避免 TopN 同源；
#   3) 输出保留原始全部列，并附加 raw_score（初步得分）与 final_score（最终得分）；
#   4) group_stability 无论是否参与打分，都会明确出现在返回表中。
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from observability.decorators import log_tool
from observability.logger import get_logger

log = get_logger(__name__)

# 默认权重池：用户未提供 weights 时使用
DEFAULT_WEIGHTS: dict[str, float] = {
    "count": 0.25,            # 丰度
    "mfe": 0.25,              # 稳定性（MFE 越负越稳定）
    "self_similarity": 0.20,  # 截短后结构保持度（方便后续截短实验）
    "group_stability": 0.20,  # 组内平均结构相似度
    "motif": 0.10,            # 含有 Motif 加分
}

# 分组列查找优先级
GROUP_COLUMNS = ("group_id", "cluster_id")


def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """把指定列安全转成数值，无法转换的位置置为 NaN。"""
    return pd.to_numeric(df[col], errors="coerce")


def _compute_group_stability(
    df: pd.DataFrame,
    df_pairs: pd.DataFrame | None = None,
) -> pd.Series | None:
    """由两两结构比较结果计算每条序列的组内平均相似度。

    计算步骤：
        a) 取 seq_A + 相似度列，按 seq_A 求均值；
        b) 取 seq_B + 相似度列（把 seq_B 当作 key），按 seq_B 求均值；
        c) 合并 a、b 两份均值，逐序列求总体平均（同一序列两边都出现时取两者均值）；
        d) 映射回 df，未在 df_pairs 中出现过的序列填 0.0
           （表示没有任何组内稳定性证据）。

    相似度列优先使用 trimmed_sim，缺失时退回 full_sim。

    返回与 df 同索引的 Series；df_pairs 不可用 / 缺少必要列 / 完全无有效
    比较记录时返回 None（调用方会补一列 NaN，保证输出列存在）。
    """
    if df_pairs is None or df_pairs.empty:
        return None
    if "seq_A" not in df_pairs.columns or "seq_B" not in df_pairs.columns:
        return None
    if "sequence" not in df.columns:
        return None

    sim_col = next(
        (c for c in ("trimmed_sim", "full_sim") if c in df_pairs.columns), None
    )
    if sim_col is None:
        return None

    pairs = df_pairs[["seq_A", "seq_B", sim_col]].copy()
    pairs[sim_col] = pd.to_numeric(pairs[sim_col], errors="coerce")
    pairs = pairs.dropna(subset=["seq_A", "seq_B", sim_col])
    if pairs.empty:
        return None

    # a) 每个 seq_A 的平均相似度
    mean_a = pairs.groupby("seq_A")[sim_col].mean().rename("mean_a")
    # b) 每个 seq_B 的平均相似度（seq_B 同样是一条参与比较的序列）
    mean_b = pairs.groupby("seq_B")[sim_col].mean().rename("mean_b")

    # c) 合并两份均值，得到每条序列的总体平均相似度
    combined = pd.concat([mean_a, mean_b], axis=1)
    combined.index.name = "sequence"
    per_seq = combined.mean(axis=1, skipna=True)

    # d) 合并回 df_seq；没有比较记录的序列记 0
    mapped = df["sequence"].map(per_seq)
    if not mapped.notna().any():
        return None
    return mapped.fillna(0.0).clip(0.0, 1.0)


def _build_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """从输入表提取所有可用特征，返回 {特征名: 与 df 同索引的 Series}。

    只返回实际可计算的列；缺失的特征不会出现在结果中，由调用方剔除权重。
    group_stability 只从 df 的同名列读取（调用方已提前合并好）。
    """
    feats: dict[str, pd.Series] = {}

    # ---- 丰度 ----
    if "count" in df.columns:
        feats["count"] = _to_numeric(df, "count").fillna(0.0).clip(lower=0.0)

    # ---- 稳定性：MFE 取绝对值，越负（绝对值越大）越好 ----
    mfe_col = next((c for c in ("mfe_full", "mfe") if c in df.columns), None)
    if mfe_col is not None:
        feats["mfe"] = _to_numeric(df, mfe_col).abs().fillna(0.0)

    # ---- 截短后结构保持度 ----
    if "self_similarity" in df.columns:
        feats["self_similarity"] = (
            _to_numeric(df, "self_similarity").fillna(0.0).clip(0.0, 1.0)
        )

    # ---- Motif 命中 ----
    # 优先查 motif_seq，其次查 hit_motifs / motif_hit / has_motif
    hit_col = next(
        (c for c in ("motif_seq", "hit_motifs", "motif_hit", "has_motif") if c in df.columns),
        None
    )
    if hit_col is not None:
        # 如果是字符串（如逗号分隔的 motif），只要不是空串就算命中
        # 如果是数值（0/1），大于 0 就算命中
        col_data = df[hit_col]
        if pd.api.types.is_string_dtype(col_data) or col_data.dtype == object:
            hit = col_data.fillna("").astype(str).str.strip().ne("")
        else:
            hit = pd.to_numeric(col_data, errors="coerce").fillna(0) > 0
        feats["motif"] = hit.astype("float64")

    # ---- 组内结构稳定性：读取已合并好的列 ----
    if "group_stability" in df.columns:
        stability = _to_numeric(df, "group_stability")
        if stability.notna().any():      # 全为 NaN（未传 df_pairs）时视为特征不可用
            feats["group_stability"] = stability.fillna(0.0).clip(0.0, 1.0)

    return feats


def _normalize_weights(
    weights: dict,
    available: set[str],
) -> dict[str, float]:
    """剔除不可用/非法权重，并把剩余权重缩放到总和为 1。

    若提供的权重全部非法但仍有可用特征，则在可用特征上均分。
    """
    used: dict[str, float] = {}
    for key, value in weights.items():
        if key not in available:
            continue
        try:
            w = float(value)
        except (TypeError, ValueError):
            continue
        if w > 0:
            used[key] = w

    total = sum(used.values())
    if total > 0:
        return {k: w / total for k, w in used.items()}

    # 权重全为 0 / 非数值：退化为均分
    keys = [k for k in weights if k in available]
    if keys:
        return {k: 1.0 / len(keys) for k in keys}
    return {}


def _minmax(series: pd.Series) -> pd.Series:
    """把一维特征缩放到 [0, 1]；常量列会被缩放为全 0。"""
    values = series.to_numpy(dtype="float64").reshape(-1, 1)
    scaled = MinMaxScaler().fit_transform(values).ravel()
    return pd.Series(scaled, index=series.index)


@log_tool("rank_candidates")
def rank_candidates(
    df_seq: pd.DataFrame,
    df_pairs: pd.DataFrame | None = None,
    weights: dict | None = None,
    top_n: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """多维度加权评分 + 分组择优，返回 (推荐结果表, meta)。

    参数：
        df_seq  : 候选序列表，可含 sequence / count / mfe_full /
                  self_similarity / motif_seq / group_id / cluster_id 等列
        df_pairs: 两两结构比较表（结构模块输出），用于计算 group_stability；
                  为 None 或缺少必要列时自动跳过该特征
        weights : 特征权重字典；为 None 时使用 DEFAULT_WEIGHTS
        top_n   : 最终推荐的序列条数

    返回：
        (df_out, meta)
        df_out  : 原始全部列 + group_stability + raw_score + final_score，
                  按得分降序；
        meta    : n_input / n_returned / weights_used / missing_features / strategy
    """
    # ---- 0) 先把 group_stability 合并进 df_seq，保证该列始终存在 ----
    df_seq = df_seq.copy()
    stability = _compute_group_stability(df_seq, df_pairs)
    if stability is not None:
        df_seq["group_stability"] = stability
    elif "group_stability" not in df_seq.columns:
        # 未传 df_pairs（或数据不可用）：补一列 NaN，保持输出结构稳定
        df_seq["group_stability"] = np.nan

    # ---- 1) 权重与特征准备 ----
    requested = dict(DEFAULT_WEIGHTS if weights is None else weights)
    feats = _build_features(df_seq)
    available = set(feats)

    weights_used = _normalize_weights(requested, available)  # 权重清洗与归一化
    missing_features = [k for k in requested if k not in available]  # 缺失特征，供日志/前端提示
    group_col = next((c for c in GROUP_COLUMNS if c in df_seq.columns), None)  # 动态寻找分组列
    strategy = "stratified" if group_col is not None else "global"  # 制定排序/打分策略

    if missing_features:
        log.warning(f"以下特征不可用，已剔除对应权重：{missing_features}")
    if not weights_used:
        log.warning("没有任何可用特征，所有候选得分将记为 0")

    # ---- 2) 空表直接返回，避免 MinMaxScaler 处理空数组 ----
    if df_seq.empty:
        df_out = df_seq.copy()
        df_out["raw_score"] = pd.Series(dtype="float64")
        df_out["final_score"] = pd.Series(dtype="float64")
        meta = {
            "n_input": 0,
            "n_returned": 0,
            "weights_used": weights_used,
            "missing_features": missing_features,
            "strategy": strategy,
        }
        log.info("输入为空，返回空推荐结果")
        return df_out, meta

    # ---- 3) 特征归一化 + 加权得分 ----
    work = df_seq.copy()
    raw_values = np.zeros(len(work), dtype=np.float64)
    for key in sorted(weights_used):
        raw_values += _minmax(feats[key]).to_numpy() * weights_used[key]
    work["raw_score"] = raw_values

    ranked = work.sort_values("raw_score", ascending=False, kind="stable")

    # ---- 4) 分组择优：每个 group 只保留最高分的一条 ----
    if group_col is not None:
        # 先整体降序，再每组取第一条，等价于组内取最高分；
        # dropna=False 保证分组列为空的行也作为一个分组参与
        selected = ranked.groupby(group_col, sort=False, dropna=False).head(1)
    else:
        selected = ranked

    # ---- 5) 在分组结果上取全局 TopN ----
    limit = max(int(top_n), 0)
    df_out = (
        selected.sort_values("raw_score", ascending=False, kind="stable")
        .head(limit)
        .reset_index(drop=True)
    )
    df_out["final_score"] = df_out["raw_score"]

    meta = {
        "n_input": int(len(df_seq)),
        "n_returned": int(len(df_out)),
        "weights_used": weights_used,
        "missing_features": missing_features,
        "strategy": strategy,
    }
    log.info(f"打分完成：输入 {meta['n_input']} 条 -> 返回 {meta['n_returned']} 条"
             f"（strategy={strategy}, 使用特征={sorted(weights_used)}）")
    return df_out, meta
