# core/structure.py
# 职责：用 ViennaRNA 计算 aptamer 候选序列的二级结构，并做两类比较：
#   1) 自比较：同一序列“全长”与“去掉引物”后的结构相似度；
#   2) 组内比较：同组序列两两之间的结构相似度（全长 / 去引物）。
# 设计原则：
#   1) 作为 Agent 独立工具，支持 list[str] 或 DataFrame；缺少
#      cluster_id / motif_seq / count / 引物时自动降级，不报错。
#      若 DataFrame 含 raw_seq + variable_region（clean 契约），直接折叠两列，
#      不再重复做引物截取。
#   2) 单条序列折叠失败只跳过该条，两两比对失败只跳过该对，整体流程不中断。
# =============================================================================
from __future__ import annotations
from collections.abc import Callable
import pandas as pd
import RNA

from observability.decorators import log_tool
from observability.logger import get_logger

log = get_logger(__name__)

TOP_N_PER_GROUP = 3
PAIR_COLUMNS = ["group_id", "seq_A", "seq_B", "full_sim", "trimmed_sim"]

def _empty_seq_frame() -> pd.DataFrame:
    """带表头与数值类型的空 df_seq。"""
    return pd.DataFrame({
        "group_id": pd.Series(dtype="object"),# 分组 ID，通常是字符串或类别型，使用 object
        "sequence": pd.Series(dtype="object"),# 原始核酸序列，字符串
        "mfe_full": pd.Series(dtype="float64"), # 全长最小自由能 (MFE)，浮点数
        "structure_full": pd.Series(dtype="object"),# 全长二级结构（如点括号），字符串
        "mfe_trimmed": pd.Series(dtype="float64"),# 截断后的最小自由能，浮点数
        "structure_trimmed": pd.Series(dtype="object"),# 截断后的二级结构，字符串
        "self_similarity": pd.Series(dtype="float64"),# 自相似度分数，浮点数
    })


def _empty_pair_frame() -> pd.DataFrame:
    """带表头与数值类型的空 df_pairs。"""
    return pd.DataFrame({
        "group_id": pd.Series(dtype="object"),# 配对所属的分组 ID
        "seq_A": pd.Series(dtype="object"),# 参与对比的序列 A（通常是较短或作为基准的序列）
        "seq_B": pd.Series(dtype="object"),# 参与对比的序列 B（与序列 A 进行比对的另一条序列）
        "full_sim": pd.Series(dtype="float64"),# 两条序列在全长（full length）层面的相似度得分（如 identity、结构编辑距离等）
        "trimmed_sim": pd.Series(dtype="float64"), # 两条序列在截断/核心区域（trimmed）层面的相似度得分
    })


def _normalizer(dna_mode: bool) -> Callable[[str], str]:
    """返回序列归一化函数：统一大小写；DNA 模式把 U 归一到 T。"""
    def norm(s: str) -> str:
        s = str(s).strip().upper()
        return s.replace("U", "T") if dna_mode else s
    return norm


def _text(value: object) -> str:
    """安全取字符串：None / NaN 视为空串，其余转 str 并去除首尾空白。"""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _extract_groups(
    data: pd.DataFrame | list[str],
    source: str,  #数据来源/模式。通常会是 "direct"（直接输入）、"cluster"（聚类结果）或 "motif"（保守序列结果）。决定了函数内部走哪套处理逻辑。
    dna_mode: bool,
) -> list[tuple[str, list[dict]]]:
    """把任意输入统一成 [(group_id, 行字典列表), ...]。

    与旧版不同：这里保留完整行数据（raw_seq、variable_region、count、
    hit_motifs 等），而不是只取序列字符串。
    - list / source=direct：全部归入 "direct"；
    - source=cluster：按 cluster_id 分组，每组按 count 降序取前 3；
    - source=motif：每个 motif 找包含它的序列，按 count 降序取前 3；
    - 缺少对应列或 source 未知时退化为 direct，保证 Agent 流程不中断。
    """
    if isinstance(data, (list, tuple)):
        return [("direct", [{"sequence": s, "variable_region": s} for s in data])]

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"input_data 需为 DataFrame 或 list[str]，收到：{type(data).__name__}"
        )

    seq_col = next(
        (c for c in ("variable_region", "sequence", "raw_seq") if c in data.columns),
        None,
    )
    if seq_col is None:
        log.warning(f"DataFrame 缺少 variable_region / sequence / raw_seq 列"
                    f"（当前：{list(data.columns)}），返回空结果")
        return []

    source = (source or "direct").lower()

    # ---- 聚类分组：按 count 降序取每组前 3 ----
    if source == "cluster" and "cluster_id" in data.columns:
        groups: list[tuple[str, list[dict]]] = []  # [(簇ID, 行字典列表), ...]
        # 按簇ID分组遍历
        for cid, sub in data.groupby("cluster_id", sort=False, dropna=False):
            if "count" in sub.columns:
                sub = sub.sort_values("count", ascending=False, kind="stable")
            # 整行转字典，count / hit_motifs 等原始特征一并带入下游
            records = sub.head(TOP_N_PER_GROUP).to_dict("records")
            if records:  # 防空白名单：如果该组过滤完后还有序列，才记录到结果中
                groups.append((str(cid), records))
        return groups

    # ---- Motif 分组：找包含 motif 的序列，按 count 降序取前 3 ----
    if source == "motif" and "motif_seq" in data.columns:
        groups = []
        text = data[seq_col].fillna("").astype(str)  # 预处理序列文本，为后续的向量化搜索做准备
        motifs = data["motif_seq"].dropna().astype(str)  # 提取所有的 motif，并去重
        for motif in dict.fromkeys(motifs):
            if not motif.strip():  # 过滤掉纯空白的无效 motif（比如空格、制表符）
                continue
            mask = text.str.contains(motif, case=False, regex=False)  # Pandas 向量化字符串匹配
            sub = data[mask]
            if "count" in sub.columns:
                sub = sub.sort_values("count", ascending=False, kind="stable")
            records = sub.head(TOP_N_PER_GROUP).to_dict("records")
            if records:
                groups.append((motif, records))
        return groups

    # cluster/motif 所需的列缺失，或 source 未知：退化为 direct
    if source in ("cluster", "motif"):
        log.warning(f"source={source} 但缺少对应列，退化为 direct")
    return [("direct", data.to_dict("records"))]


def _fold(seq: str, dna_mode: bool) -> tuple[str, float]:
    """折叠单条序列，返回 (dot-bracket, MFE)。异常由调用方捕获。"""
    rna = seq.replace("T", "U") if dna_mode else seq
    struct, mfe = RNA.fold(rna)
    return struct, float(mfe)


def _trim_primers(seq: str, fwd: str, rev: str) -> tuple[str, int, int] | None:
    """截取两条引物之间的区域，返回 (中间序列, 起点, 终点)；未匹配返回 None。"""
    start = seq.find(fwd)
    if start == -1:
        return None
    start += len(fwd)                     # 跳过正向引物
    end = seq.rfind(rev, start)           # 取末端最后一次出现的反向引物
    if end == -1 or end <= start:
        return None
    return seq[start:end], start, end


def _bp_similarity(struct_a: str, struct_b: str, n_bases: int) -> float | None:
    """结构相似度 = 1 - bp_distance / (n_bases / 2)，裁剪到 [0, 1]。

    两条结构长度不一致时右侧补 '.' 对齐；n_bases < 2 或计算失败返回 None。
    """
    if n_bases < 2:
        return None
    length = max(len(struct_a), len(struct_b))
    try:
        dist = RNA.bp_distance(struct_a.ljust(length, "."),
                               struct_b.ljust(length, "."))
    except Exception as exc:              # ViennaRNA 遇到异常结构/长度会抛错
        log.warning(f"bp_distance 计算失败：{exc}")
        return None
    sim = 1.0 - dist / (n_bases / 2.0)
    return max(0.0, min(1.0, sim))


def _process_group(
    group_id: str,
    seqs: list[dict],
    fwd: str,
    rev: str,
    dna_mode: bool,
    do_self_compare: bool, # 自比对开关：决定是否计算序列“全长”与“截断后”结构的自我相似度
    do_pair_compare: bool, # 两两比对开关：决定是否计算该 group 内序列 A 与序列 B 之间的相似度
) -> tuple[list[dict], list[dict]]:
    """折叠一个 group 的所有行，返回 (df_seq 行, df_pairs 行)。

    入参是带原始列的行字典；输出行在原行基础上追加结构相关字段，
    因此 count、hit_motifs、cluster_id 等上游列不会丢失。

    序列来源分两种：
    - 契约模式：行中含 raw_seq（clean 模块产物）时，raw_seq 直接作为全长折叠，
      variable_region 直接作为去引物区折叠，本模块不再做引物定位或拼接；
    - 兼容模式：list[str] 或 DataFrame 无 raw_seq 时，用 _trim_primers 本地截取。
    """
    seq_rows: list[dict] = []  # 存储序列级别数据的行列表
    pair_rows: list[dict] = []  # 存储配对级别数据的行列表
    records: list[dict] = []              # 供两两比对复用的折叠结果
    norm = _normalizer(dna_mode)

    for item in seqs:
        # ---- 提取序列 ----
        # 契约模式（clean 产物）：raw_seq 是全长相，variable_region 是去引物核心区。
        # 此时直接使用两列，本模块不再做任何 find/rfind 截取或字符串拼接。
        raw_text = _text(item.get("raw_seq"))
        if raw_text:
            contract_mode = True
            full_seq = norm(raw_text)
            region_text = _text(item.get("variable_region"))
            trimmed_seq = norm(region_text) if region_text else ""
            if trimmed_seq == full_seq:   # 无独立核心区（如 clean 未给引物）
                trimmed_seq = ""
        else:
            # 非契约输入（list[str] 或 DataFrame 无 raw_seq）：仍需本地截取
            contract_mode = False
            seq_text = (_text(item.get("variable_region"))
                        or _text(item.get("sequence")))
            if not seq_text:
                log.debug(f"空序列，跳过 {group_id} 中的一行")
                continue
            full_seq = norm(seq_text)
            trimmed_seq = ""

        if not full_seq:
            continue

        # ---- 全长折叠；失败则跳过该序列 ----
        try:
            struct_full, mfe_full = _fold(full_seq, dna_mode)
        except Exception as exc:
            log.warning(f"全长折叠失败，跳过 {group_id}/{full_seq[:20]}…：{exc}")
            continue

        # ---- 去引物折叠与自比较；任一环节失败都退回全长 ----
        struct_trimmed, mfe_trimmed, self_sim = struct_full, mfe_full, 1.0
        trimmed_ok, trimmed_len = False, 0
        if do_self_compare:
            if contract_mode and trimmed_seq:
                # clean 已截取好：直接折叠 variable_region，不再定位引物
                try:
                    tstruct, tmfe = _fold(trimmed_seq, dna_mode)
                except Exception as exc:
                    log.warning(f"去引物折叠失败，退回全长："
                                f"{group_id}/{trimmed_seq[:20]}…：{exc}")
                    tstruct = None
                if tstruct is not None:
                    # 无需截取偏移：_bp_similarity 会把较短结构右侧补 '.' 对齐
                    sim = _bp_similarity(struct_full, tstruct, len(full_seq))
                    if sim is not None:
                        struct_trimmed, mfe_trimmed = tstruct, tmfe
                        self_sim, trimmed_ok, trimmed_len = sim, True, len(trimmed_seq)
            elif not contract_mode and fwd and rev:
                # 兼容 list[str] / 无 raw_seq 的 DataFrame：本地截取
                trimmed = _trim_primers(full_seq, fwd, rev)
                if trimmed is None:
                    log.debug(f"引物未匹配，{group_id}/{full_seq[:20]}… 按全长处理")
                else:
                    tseq, start, end = trimmed
                    try:
                        tstruct, tmfe = _fold(tseq, dna_mode)
                    except Exception as exc:
                        log.warning(f"去引物折叠失败，退回全长："
                                    f"{group_id}/{full_seq[:20]}…：{exc}")
                        tstruct = None
                    if tstruct is not None:
                        padded = "." * start + tstruct + "." * (len(full_seq) - end)
                        sim = _bp_similarity(struct_full, padded, len(full_seq))
                        if sim is not None:
                            struct_trimmed, mfe_trimmed = tstruct, tmfe
                            self_sim, trimmed_ok, trimmed_len = sim, True, len(tseq)

        # 动态字典合并：先复制原始行以保留 count / hit_motifs 等全部列，
        # 再覆盖/补充本模块计算出的字段
        result_dict = item.copy()
        result_dict.update({
            "group_id": group_id,
            "sequence": full_seq,
            "mfe_full": mfe_full,
            "structure_full": struct_full,
            "mfe_trimmed": mfe_trimmed,
            "structure_trimmed": struct_trimmed,
            "self_similarity": self_sim,
        })
        seq_rows.append(result_dict)
        records.append({
            "sequence": full_seq,
            "struct_full": struct_full,
            "struct_trimmed": struct_trimmed,
            "trimmed_ok": trimmed_ok,
            "trimmed_len": trimmed_len,
        })

    # ---- 同组两两比较；只有 1 条序列时自然跳过 ----
    if do_pair_compare:
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                a, b = records[i], records[j]
                full_sim = _bp_similarity(
                    a["struct_full"], b["struct_full"], len(a["sequence"])
                )
                if full_sim is None:      # 该对无法计算，跳过
                    continue
                trimmed_sim = full_sim    # 没引物 / 未开自比较 / 截取失败时沿用全长
                if a["trimmed_ok"] and b["trimmed_ok"]:
                    ts = _bp_similarity(
                        a["struct_trimmed"], b["struct_trimmed"],
                        min(a["trimmed_len"], b["trimmed_len"]),
                    )
                    if ts is not None:
                        trimmed_sim = ts
                pair_rows.append({
                    "group_id": group_id,
                    "seq_A": a["sequence"],
                    "seq_B": b["sequence"],
                    "full_sim": full_sim,
                    "trimmed_sim": trimmed_sim,
                })

    return seq_rows, pair_rows


@log_tool("compare_structures")
def compare_structures(
    input_data: pd.DataFrame | list[str],
    source: str = "direct",            # "direct", "cluster", "motif"
    fwd_primer: str = "",
    rev_primer: str = "",
    temp: float = 37.0,
    na_conc: float = 1.0,
    dna_mode: bool = True,
    do_self_compare: bool = True,      # 是否比较“全长”与“去引物”
    do_pair_compare: bool = True       # 是否进行“同组两两比对”
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """计算二级结构并比较，返回 (df_seq, df_pairs, meta)。

    参数：
        input_data     : 序列列表，或含 variable_region / sequence 列的 DataFrame；
                         若含 raw_seq（clean 模块输出），则直接用 raw_seq 作全长、
                         variable_region 作去引物区，不再本地截取
        source         : direct（全部一组）/ cluster（按 cluster_id）/ motif（按 motif_seq）
        fwd_primer     : 正向引物；仅在兼容模式（list / 无 raw_seq）下用于本地截取
        rev_primer     : 反向引物；同上
        temp           : 折叠温度（摄氏度）
        na_conc        : 盐浓度（mol/L）
        dna_mode       : True 时把 T 换成 U 再折叠
        do_self_compare: 是否计算全长 vs 去引物的结构相似度
        do_pair_compare: 是否计算同组序列两两结构相似度
    """
    # ---- ViennaRNA 全局参数 ----
    try:
        RNA.cvar.temperature = float(temp)
        RNA.cvar.salt = float(na_conc)
    except Exception as exc:
        log.warning(f"设置 ViennaRNA 全局参数失败（temperature/salt）：{exc}")

    # 引物与序列统一字母表，便于匹配
    norm = _normalizer(dna_mode)
    fwd = norm(fwd_primer or "")
    rev = norm(rev_primer or "")

    groups = _extract_groups(input_data, source, dna_mode)

    # ---- 逐组折叠 + 比对 ----
    seq_rows: list[dict] = [] # 准备两个大桶，用来收集所有组落下来的数据（避免用 Pandas 逐行追加导致极其缓慢）
    pair_rows: list[dict] = []
    for group_id, seqs in groups:
        s_rows, p_rows = _process_group(
            group_id, seqs, fwd, rev, dna_mode,
            do_self_compare, do_pair_compare,
        )
        seq_rows.extend(s_rows)
        pair_rows.extend(p_rows)

    # seq_rows 中每行都是「原始行字典 + 结构字段」，直接转表即可自动保留全部列
    df_seq = pd.DataFrame(seq_rows) if seq_rows else _empty_seq_frame()
    df_pairs = (pd.DataFrame(pair_rows, columns=PAIR_COLUMNS)
                if pair_rows else _empty_pair_frame())

    meta = {
        "source": (source or "direct").lower(),
        "n_sequences": len(df_seq),
        "n_groups": int(df_seq["group_id"].nunique()) if not df_seq.empty else 0,
        "do_self_compare": do_self_compare,
        "do_pair_compare": do_pair_compare,
        "params": {
            "temp": temp,
            "na_conc": na_conc,
            "dna_mode": dna_mode,
            "fwd_primer": fwd_primer,
            "rev_primer": rev_primer,
        },
    }
    log.info(f"结构分析完成：{len(groups)} 组 -> 折叠 {len(df_seq)} 条序列，"
             f"生成 {len(df_pairs)} 对结构比较"
             f"（self_compare={do_self_compare}, pair_compare={do_pair_compare}）")
    return df_seq, df_pairs, meta
