# 职责：对 clean.py 输出的序列做小规模贪心聚类（约 50 条、每条 40nt，无需 CD-HIT）。
# 输入约定：DataFrame 含 variable_region、count 两列（由 clean.py 保证）。
# 算法：按 count 降序，依次取未聚类序列作为代表（representative），与所有未聚类
#      序列计算相似度，>= identity 的归入同一簇（以代表序列为中心的星形簇）。
# 相似度：Biopython 全局比对（PairwiseAligner）得分 / (较短序列长度 * match_score)；
#        可容忍测序数据中的插入/缺失，避免逐位比对错位导致相似度被低估。
# =============================================================================
from __future__ import annotations
import pandas as pd
from Bio import Align
from observability.decorators import log_tool
from observability.logger import get_logger

log = get_logger(__name__)

# 模块级全局比对器：只建一次，供所有序列对比复用
aligner = Align.PairwiseAligner()  # (性能优化：把比对器定义在模块级别，避免每次调用比对函数时都重新创建对象，节省内存和时间)
aligner.mode = "global"  # 设置比对模式为“全局比对（Global Alignment）”。
aligner.match_score = 1.0  # 匹配得分（Match Score）：当两条序列在某个位置的字符完全相同时，加 1.0 分。
aligner.mismatch_score = -1.0  # 错配得分（Mismatch Score）：当两条序列在某个位置的字符不同时，扣 1.0 分。
aligner.open_gap_score = -2.0  # 开启空位罚分（Gap Open Penalty）：当需要在序列中插入一个空位（"-"）以对齐后续序列时，扣 2.0 分。
aligner.extend_gap_score = -0.5  # 延伸空位罚分（Gap Extend Penalty）：如果空位已经开启，并且在它后面紧接着又要插入一个空位（即连续的空位），那么每延伸一个空位，只扣 0.5 分。


def calc_identity(seq1: str, seq2: str) -> float:
    """用全局比对计算两条序列的相似度，返回 0.0 ~ 1.0。
    相似度 = 比对得分 / (较短序列长度 * match_score)；
    得分 <= 0 表示没有可比对的匹配，直接返回 0.0。
    """
    n = min(len(seq1), len(seq2))
    if n == 0:                               # 空序列保护，避免除零
        return 1.0 if seq1 == seq2 else 0.0
    score = aligner.score(seq1, seq2)
    if score <= 0:
        return 0.0
    return min(1.0, score / (n * aligner.match_score))


@log_tool("cluster_sequences")
def cluster_sequences(
    df: pd.DataFrame,
    identity: float = 0.8,
    min_cluster_size: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """对序列做贪心聚类，返回带聚类信息的 DataFrame 与 meta。

    参数：
        df               : 含 variable_region、count 两列的 DataFrame（clean.py 输出）
        identity         : 相似度阈值，>= 该值归入同一簇（0 < identity <= 1）
        min_cluster_size : 过滤掉成员数小于该值的簇

    返回：
        (clustered_df, meta)
        clustered_df 在原列基础上新增 cluster_id / is_representative /
        cluster_size / similarity_to_rep，行顺序与输入一致，cluster_id 连续编号；
        meta 含 n_clusters、n_representatives、cluster_sizes、params 及过滤统计。
    """
    if df.empty:
        raise ValueError("输入 DataFrame 为空，无可聚类的序列")
    missing = {"variable_region", "count"} - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列：{sorted(missing)}")
    if not 0 < identity <= 1:
        raise ValueError(f"identity 需在 (0, 1] 之间，收到：{identity}")
    if min_cluster_size < 1:
        raise ValueError(f"min_cluster_size 需 >= 1，收到：{min_cluster_size}")

    # 按 count 降序（稳定排序，count 相同时保持输入顺序）；保留原 index 以便还原行序
    work = df.sort_values("count", ascending=False, kind="stable")
    seqs = work["variable_region"].astype(str).tolist()
    n = len(seqs)

    cluster_ids = [-1] * n      # -1 表示尚未归类
    is_rep = [False] * n
    sims = [0.0] * n
    sizes: list[int] = []       # 下标即簇编号

    for i, rep in enumerate(seqs):
        if cluster_ids[i] != -1:        # 已被更早的代表序列收走
            continue
        cluster_ids[i] = len(sizes)     # 新簇编号
        is_rep[i] = True
        sims[i] = 1.0                   # 代表序列与自身相似度为 1
        members = 1
        for j in range(i + 1, n):       # 更早的行必然已归类，只需向后看
            if cluster_ids[j] != -1:
                continue
            sim = calc_identity(rep, seqs[j])
            if sim >= identity:         # 只与代表比较，不做传递合并
                cluster_ids[j] = cluster_ids[i]
                sims[j] = sim
                members += 1
        sizes.append(members)

    # 合并回原 DataFrame：按 work 的 index 对齐，再恢复输入行顺序
    clustered = work.assign(
        cluster_id=cluster_ids,
        is_representative=is_rep,
        similarity_to_rep=sims,
        cluster_size=[sizes[c] for c in cluster_ids],
    ).sort_index(kind="stable")

    # 过滤小簇，并把保留下来的 cluster_id 重编号为 0..k-1
    clustered = clustered[clustered["cluster_size"] >= min_cluster_size].copy()
    kept_ids = sorted(clustered["cluster_id"].unique())
    remap = {old: new for new, old in enumerate(kept_ids)}
    clustered["cluster_id"] = clustered["cluster_id"].map(remap).astype("int64")
    clustered = clustered.reset_index(drop=True)
    cluster_sizes = [sizes[old] for old in kept_ids]

    n_clusters = len(kept_ids)
    meta = {
        "n_clusters": n_clusters,# 最终保留下来的簇的总数
        "n_representatives": int(clustered["is_representative"].sum()), # 最终包含的“代表序列”数量。
        "cluster_sizes": cluster_sizes, # 每个簇的成员数量列表
        "params": {"identity": identity, "min_cluster_size": min_cluster_size},
        # 附加统计：便于在 trace 中查看过滤损耗
        "n_sequences": len(df),
        "n_clusters_raw": len(sizes),
        "n_dropped_clusters": len(sizes) - n_clusters,
        "n_dropped_sequences": len(df) - len(clustered),
    }
    log.info(f"聚类完成：{len(df)} 条序列 -> {n_clusters} 个簇"
             f"（过滤前 {len(sizes)} 个，min_cluster_size={min_cluster_size}），"
             f"簇大小 {cluster_sizes}")
    return clustered, meta
