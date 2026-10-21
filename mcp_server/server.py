# mcp_server/server.py
# 职责：用 FastMCP 把 core 下的分析函数封装成 MCP 工具，通过 streamable-http 暴露。
# 架构说明：
#   LLM 无法直接传递 DataFrame，因此服务端维护进程内全局缓存 DATA_STORE：
#   {dataset_id: pd.DataFrame}。工具之间只传 dataset_id 等标量，真实数据留在内存。
# 注意：DATA_STORE 仅在本进程生命周期内有效，服务重启后所有 dataset_id 失效。
# =============================================================================
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from mcp.server import MCPServer

# 支持 `python mcp_server/server.py` 直接启动：把项目根目录加入 sys.path，
# 否则脚本所在目录（mcp_server/）会被当作导入根，导致找不到 core 包。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 读取项目根目录的 .env（已存在的系统环境变量优先）
load_dotenv(PROJECT_ROOT / ".env")

from core.clean import clean_sequences
from core.cluster import cluster_sequences
from core.loader import load_sequences
from core.motif import discover_motifs
from core.rank import rank_candidates
from core.structure import compare_structures
from observability.logger import get_logger

log = get_logger(__name__)

# ---- MCP 服务与内存缓存 ----
mcp = MCPServer("AptamerAnalysis",)

# dataset_id -> DataFrame / dict
DATA_STORE: dict[str, Any] = {}


def _new_dataset_id() -> str:
    """生成 8 位短 ID，作为 DATA_STORE 的键返回给 LLM。"""
    return str(uuid.uuid4())[:8]


def _save(data: Any) -> str:
    """把数据写入内存缓存，返回其 dataset_id。"""
    dataset_id = _new_dataset_id()
    DATA_STORE[dataset_id] = data
    log.info(f"数据已缓存：dataset_id={dataset_id}, type={type(data).__name__}")
    return dataset_id


def _get(dataset_id: str) -> Any:
    """按 dataset_id 取数据；不存在时给出可操作的错误信息。"""
    if dataset_id not in DATA_STORE:
        raise ValueError(
            f"dataset_id 不存在：{dataset_id}。"
            f"请先调用上一步工具生成数据（当前缓存 {len(DATA_STORE)} 份）。"
        )
    return DATA_STORE[dataset_id]


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON 安全的 records（NaN 转 null，numpy 标量转原生类型）。"""
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", force_ascii=False))


# =============================================================================
# 1) 读取
# =============================================================================
@mcp.tool()
def tool_inspect_excel(path: str) -> dict:
    """读取 Excel 的表头和前 3 行数据，帮助 LLM 推断哪一列是序列，哪一列是 count。
    参数：
        path: Excel 文件的绝对路径。
    返回：
        包含所有列名和前3行数据的字典。
    """
    path = path.strip('"').strip("'").strip()
    path = str(Path(path).resolve())
    try:
        # 只读前 3 行，避免大文件卡死
        df = pd.read_excel(path, nrows=3, dtype=str)
        # 提取列名和前 3 行数据
        columns = list(df.columns)
        sample_data = df.to_dict(orient="records")

        return {
            "columns": columns,
            "sample_rows": sample_data
        }
    except Exception as e:
        raise RuntimeError(f"无法读取 Excel 文件: {e}")


# =============================================================================
# 2) 加载：读取 Excel 或接收手动输入序列，统一输出 raw_seq / count
# =============================================================================
@mcp.tool()
def tool_load_sequences(
    path: str | None = None,
    sequences: list[str] | None = None,
    seq_col: str | None = None,
    count_col: str | None = None,
) -> dict:
    """加载测序数据：既支持读取 Excel 文件，也支持直接接收手动输入的序列列表。

    模式 A（Excel 文件）：
        用户提供 Excel 文件路径时，传 path 和 seq_col。
        如果不知道列名，请先调用 tool_inspect_excel 查看表头，再回来传 seq_col。

    模式 B（手动输入序列）：
        用户直接粘贴了几条短序列时，直接传 sequences 列表即可，
        无需传 path 和 seq_col；每条序列的 count 记为 1。

    参数：
        path      : Excel 文件绝对路径（支持 .xlsx / .xls）；与 sequences 二选一。
        sequences : 手动输入的序列列表；与 path 二选一。
        seq_col   : Excel 中序列表的列名，仅模式 A 需要。
        count_col : Excel 中丰度列的列名；不传时 count 记为 1，仅模式 A 使用。

    返回：
        {"dataset_id": 缓存 ID, "meta": 数据来源与统计信息}
        后续工具（清洗/聚类等）都需要把这个 dataset_id 作为输入。
    """
    if not path and not sequences:
        raise ValueError("必须提供 path 或 sequences 之一")

    if path:
        # ---- 模式 A：读取 Excel 文件 ----
        path = path.strip('"').strip("'").strip()
        path = str(Path(path).resolve())
        if not seq_col:
            raise ValueError(
                "缺失 seq_col 参数！请先调用 tool_inspect_excel 查看列名，"
                "然后再传入 seq_col 重新调用本工具。"
            )
        df, meta = load_sequences(path, seq_col, count_col)
    else:
        # ---- 模式 B：直接输入序列，统一按 count=1 构造 ----
        df = pd.DataFrame({"raw_seq": sequences, "count": 1})
        meta = {
            "source": "user_input",
            "n_rows_raw": len(sequences),
            "n_rows_kept": len(sequences),
        }

    dataset_id = _save(df)
    return {"dataset_id": dataset_id, "meta": meta}


# =============================================================================
# 3) 清洗：去引物、长度过滤、去重合并、TopN
# =============================================================================
@mcp.tool()
def tool_clean_sequences(
    dataset_id: str,
    fwd_primer: str = "" ,
    rev_primer: str = "" ,
    target_len: int = 0,
    tolerance: float = 0.2,
    top_n: int = 50,
) -> dict:
    """截取引物之间的可变区，并按长度过滤、去重合并、取丰度 TopN。

    参数：
        dataset_id : tool_load_sequences 返回的 dataset_id。
        fwd_primer : 正向引物序列（5'→3'），需以正向形式出现在读长中。
        rev_primer : 反向引物序列（同样以正向形式匹配）。
        target_len : 可变区目标长度（nt）。默认 0 表示自动推断：以输入序列
                     可变区长度的中位数作为基准；用户明确给出长度时才传正数
                     （如 40 表示核心区 40 nt）。
        tolerance  : 长度允许的相对偏差，默认 0.2（即基准长度 ±20%）。
        top_n      : 去重合并后保留的高丰度序列条数，默认 50。

    返回：
        {"dataset_id": 新缓存 ID, "meta": 清洗统计, "records": 清洗后全表}
        meta.params.target_len 为本次实际使用的长度，target_len_source 标明
        它是自动推断（auto）还是用户指定（user）。
        records 为 JSON 安全的逐行记录，供前端直接展示。
    """
    df = _get(dataset_id)
    cleaned = clean_sequences(
        df,
        fwd_primer=fwd_primer,
        rev_primer=rev_primer,
        target_len=target_len,
        tolerance=tolerance,
        top_n=top_n,
    )
    new_id = _save(cleaned)
    # clean_sequences 会把实际使用的长度写进 attrs（自动推断或用户指定）
    resolved_target_len = int(cleaned.attrs.get("target_len", target_len))
    target_len_source = cleaned.attrs.get("target_len_source", "user")
    length_range = list(cleaned.attrs.get("length_range", ()))
    meta = {
        "n_sequences": int(len(cleaned)),
        "params": {
            "fwd_primer": fwd_primer,
            "rev_primer": rev_primer,
            "target_len": resolved_target_len,
            "target_len_source": target_len_source,
            "length_range": length_range,
            "tolerance": tolerance,
            "top_n": top_n,
        },
    }
    return {
        "dataset_id": new_id,
        "meta": meta,
        "records": _records(cleaned),
    }


# =============================================================================
# 4) 聚类
# =============================================================================
@mcp.tool()
def tool_cluster_sequences(
    dataset_id: str,
    identity: float = 0.8,
    min_cluster_size: int = 3,
) -> dict:
    """对清洗后的可变区做贪心同源聚类，为每条序列分配 cluster_id。

    参数：
        dataset_id      : 清洗工具返回的 dataset_id（需含 variable_region、count）。
        identity        : 相似度阈值，>= 该值归入同一簇，取值 (0, 1]，默认 0.8。
        min_cluster_size: 过滤成员数小于该值的簇，默认 1（不过滤）。

    返回：
        {"dataset_id": 新缓存 ID, "meta": 簇数量与参数等统计,
         "records": 聚类结果全表}
        新表在原有列基础上新增 cluster_id / is_representative /
        cluster_size / similarity_to_rep。
    """
    df = _get(dataset_id)
    clustered, meta = cluster_sequences(
        df, identity=identity, min_cluster_size=min_cluster_size
    )
    new_id = _save(clustered)
    return {
        "dataset_id": new_id,
        "meta": meta,
        "records": _records(clustered),
    }


# =============================================================================
# 5) Motif 发现（Docker + STREME）
# =============================================================================
@mcp.tool()
def tool_discover_motifs(
    dataset_id: str,
    k: int = 10,
    top_n: int = 5,
    min_support: int = 3,
) -> dict:
    """用 MEME Suite STREME（Docker）发现保守基序。

    参数：
        dataset_id  : 含 variable_region 的 dataset_id；若含 cluster_id，
                      结果会额外统计每个 motif 命中的簇。
        k           : motif 宽度（nt），范围 [3, 30]，默认 6。
        top_n       : STREME 最多输出的 motif 数量，默认 5。
        min_support : 至少被多少条序列支持才保留，默认 3。

    返回：
        {"dataset_id": 带注释的原始序列表缓存 ID,
         "motifs_dataset_id": 纯 Motif 结果表缓存 ID,
         "meta": motif 分析统计}
        dataset_id 对应的表保留全部原始列，并新增 hit_motifs 列
        （逗号分隔的命中 motif），供下游结构预测等工具使用；
        motifs_dataset_id 对应的表为 motif_seq / support / cluster_ids 三列，
        供单独查看 Motif 汇总使用。
        另附 "records"（带注释的序列全表）与 "motif_records"（Motif 汇总表），
        供前端直接展示。
        注意：本工具依赖本机 Docker 与 STREME 镜像。
    """
    df = _get(dataset_id)
    motifs_df, enriched_df, meta = discover_motifs(
        df, k=k, top_n=top_n, min_support=min_support
    )
    enriched_id = _save(enriched_df)     # 原始序列 + hit_motifs，供下游使用
    motifs_id = _save(motifs_df)         # 纯 Motif 汇总表
    return {
        "dataset_id": enriched_id,
        "motifs_dataset_id": motifs_id,
        "meta": meta,
        "records": _records(enriched_df),
        "motif_records": _records(motifs_df),
    }


# =============================================================================
# 6) 二级结构计算与比较（ViennaRNA）
# =============================================================================
@mcp.tool()
def tool_compare_structures(
    dataset_id: str,
    source: str = "direct",
    fwd_primer: str = "",
    rev_primer: str = "",
    temp: float = 37.0,
    na_conc: float = 1.0,
    dna_mode: bool = True,
) -> dict:
    """用 ViennaRNA 计算二级结构，并做全长 vs 去引物、组内两两结构比较。

    参数：
        dataset_id : 待分析的 dataset_id（通常为聚类结果）。
        source     : 分组方式，"direct" 全部一组 / "cluster" 按 cluster_id /
                     "motif" 按 motif_seq；缺少对应列时自动退化为 direct。
        fwd_primer : 正向引物；与 rev_primer 任一为空则不做去引物比较。
        rev_primer : 反向引物。
        temp       : 折叠温度（摄氏度），默认 37。
        na_conc    : 盐浓度（mol/L），默认 1.0。
        dna_mode   : true 表示输入是 DNA（折叠前自动 T→U）。

    返回：
        {"dataset_id": 结构结果表 ID,
         "pairs_dataset_id": 两两比较结果表 ID,
         "meta": 参数与统计}
        dataset_id 对应的表含 sequence / mfe_full / structure_full /
        mfe_trimmed / structure_trimmed / self_similarity 等列，并保留
        count / hit_motifs / cluster_id 等全部原始列；
        pairs 表含 seq_A / seq_B / full_sim / trimmed_sim 等列。
        另附 "df_seq" / "df_pairs" 两份 records，供前端直接展示。
    """
    df = _get(dataset_id)
    df_seq, df_pairs, meta = compare_structures(
        df,
        source=source,
        fwd_primer=fwd_primer,
        rev_primer=rev_primer,
        temp=temp,
        na_conc=na_conc,
        dna_mode=dna_mode,
    )
    seq_id = _save(df_seq)
    pairs_id = _save(df_pairs)
    return {
        "dataset_id": seq_id,
        "pairs_dataset_id": pairs_id,
        "meta": meta,
        "df_seq": _records(df_seq),
        "df_pairs": _records(df_pairs),
    }


# =============================================================================
# 7) 多维度加权评分 + 分组择优
# =============================================================================
@mcp.tool()
def tool_rank_candidates(
    dataset_id: str,
    pairs_dataset_id: str | None = None,
    weights: dict | None = None,
    top_n: int = 5,
) -> dict:
    """对候选序列做多维度加权评分，并保证每个分组最多推荐 1 条。

    参数：
        dataset_id       : tool_compare_structures 返回的 dataset_id。
        pairs_dataset_id : 同一工具返回的 pairs_dataset_id，用于计算组内平均
                           结构相似度；不传时自动跳过该特征。
        weights          : 自定义权重，键可含 count / mfe / self_similarity /
                           group_stability / motif。不传则使用默认权重：
                           0.25 / 0.25 / 0.20 / 0.20 / 0.10。缺失特征会被
                           剔除并重新归一化。
        top_n            : 最终推荐的序列条数，默认 5。

    返回：
        {"candidates": [逐条候选的完整字段], "meta": 权重与策略信息}
        存在 group_id / cluster_id 时采用分层择优（每组最多 1 条）。
        结果直接以 records 返回，不再写入缓存。
    """
    df_seq = _get(dataset_id)
    df_pairs = _get(pairs_dataset_id) if pairs_dataset_id else None

    ranked, meta = rank_candidates(
        df_seq, df_pairs=df_pairs, weights=weights, top_n=top_n
    )
    return {"candidates": _records(ranked), "meta": meta}


if __name__ == "__main__":
    # streamable-http：监听地址与端口可通过 .env / 环境变量覆盖
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))
    log.info(f"启动 MCP Server: http://{host}:{port}/mcp")
    mcp.run(transport="streamable-http", host=host, port=port)
