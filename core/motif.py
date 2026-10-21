# core/motif.py
# 职责：通过 subprocess 调用 Docker 运行 MEME Suite 的 STREME，进行保守基序发现。
# 输入兼容两种形态：
#   1) 仅含 variable_region（独立调用）
#   2) 含 variable_region + cluster_id（聚类后调用，额外统计命中的簇）
# 流程：variable_region 写入临时 FASTA → Docker 挂载运行 STREME → 解析 streme.txt
#      得到各 motif 的 consensus → 统计 support 与命中 cluster。
# 说明：本模块依赖本机 Docker 与 STREME 镜像；Docker 未启动或镜像不存在会抛 RuntimeError。
# =============================================================================
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from observability.decorators import log_tool
from observability.logger import get_logger

log = get_logger(__name__)

# 固定输出列：即使无结果也保留表头
COLUMNS = ["motif_seq", "support", "cluster_ids"]

# Docker 运行超时（秒）：防止 Docker Desktop 卡死时 Agent 无限等待
DOCKER_TIMEOUT = 300


def _format_docker_path(path_str: str) -> str:
    """把 Windows 盘符路径转成 Docker/WSL 风格的 POSIX 路径。

    例如 C:/Users/me/AppData/Local/Temp/xxx -> /c/Users/me/AppData/Local/Temp/xxx。
    非盘符路径（Linux/macOS 的 /tmp/...）原样返回。
    """
    p = Path(path_str).resolve()
    posix_path = p.as_posix()
    # 如果是 C:/Users/... 这种格式，转换为 /c/Users/...（部分 Docker Desktop 版本
    # 在 WSL2 后端下需要这种写法）
    if len(posix_path) > 1 and posix_path[1] == ":":
        return "/" + posix_path[0].lower() + posix_path[2:]
    return posix_path


def _docker_mount_path(path_str: str) -> str:
    """返回用于 docker -v 的主机路径。

    默认使用原生 C:/...（Docker Desktop 通常直接支持）；若挂载报错，可设置
    环境变量 APTAMER_DOCKER_PATH_STYLE=wsl 改用 /c/... 形式重试。
    """
    posix_path = Path(path_str).resolve().as_posix()
    if os.name != "nt":
        return posix_path

    style = os.getenv("APTAMER_DOCKER_PATH_STYLE", "native").strip().lower()
    if style in {"wsl", "wsl2", "msys"}:
        converted = _format_docker_path(path_str)
        log.info(f"Docker 挂载路径（WSL 风格）：{posix_path} -> {converted}")
        return converted
    return posix_path


def _decode_output(value: object) -> str:
    """TimeoutExpired.stdout/stderr 可能是 bytes 或 None，统一转成字符串。"""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _empty_frame() -> pd.DataFrame:
    """构造带表头的空结果表。"""
    return pd.DataFrame({
        "motif_seq": pd.Series(dtype="object"),
        "support": pd.Series(dtype="int64"),
        "cluster_ids": pd.Series(dtype="object"),
    })


def _write_fasta(seqs: list[str], path: str) -> None:
    """把序列写成 FASTA，ID 用 seq_{index}（index 与原 DataFrame 行号一致）。"""
    lines: list[str] = []
    for i, seq in enumerate(seqs):
        if not seq:                     # 跳过空序列，避免 STREME 解析失败
            continue
        lines.append(f">seq_{i}")
        lines.append(seq.upper())
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


def _parse_streme_pwm(text: str) -> list[dict]:
    """从 STREME 文本输出解析每个 motif 的 consensus + PWM 矩阵。

    返回：[{"id": str, "consensus": str, "pwm": list[list[float]], "alphabet": str}, ...]
    解析方式：对每个 motif 取 letter-probability matrix，逐列 argmax 拼 consensus，
    同时把原始概率矩阵保留下来，供后续按需使用（如 meta 中的 PWM 信息）。
    """
    m = re.search(r"ALPHABET=\s*(\S+)", text)
    alphabet = m.group(1).upper() if m else "ACGT"

    motifs: list[dict] = []
    lines = text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        if not lines[i].startswith("MOTIF"):
            i += 1
            continue
        # MOTIF 行形如 "MOTIF 1-CCCAAA STREME-1"，第二列即 sites.tsv 里的 motif_ID
        parts = lines[i].split()
        motif_id = parts[1] if len(parts) > 1 else ""
        j = i + 1
        while j < n and not lines[j].startswith("letter-probability matrix"):
            j += 1
        if j >= n:
            break
        m_al = re.search(r"alength=\s*(\d+)", lines[j])
        m_w = re.search(r"w=\s*(\d+)", lines[j])
        if not m_al or not m_w:
            i = j + 1
            continue
        alength = int(m_al.group(1))
        width = int(m_w.group(1))
        matrix: list[list[float]] = []
        for k in range(j + 1, min(j + 1 + width, n)):
            vals = [float(x) for x in lines[k].split()]
            if len(vals) >= alength:
                matrix.append(vals[:alength])
        if matrix:
            chars = []
            for row in matrix:
                idx = max(range(len(row)), key=row.__getitem__)
                chars.append(alphabet[idx] if idx < len(alphabet) else "N")
            motifs.append({
                "id": motif_id,
                "consensus": "".join(chars),
                "pwm": matrix,
                "alphabet": alphabet,
            })
        i = j + 1 + width
    return motifs


def _parse_sites_tsv(path: str) -> list[dict]:
    """解析 STREME 的 sites.tsv。

    返回：[{"motif_id": str, "seq_id": str, "matched_sequence": str,
           "score": float, "p_value": float}, ...]
    兼容 STREME 5.x 的实际列名（motif_ID / seq_ID / site_Sequence / site_Score），
    列名按大小写不敏感匹配；缺少 motif/seq/序列列时返回空列表。
    """
    rows: list[dict] = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = [h.strip().lower() for h in f.readline().rstrip("\n").split("\t")]
        col = {name: i for i, name in enumerate(header)}
        motif_col = col.get("motif_id")
        seq_col = col.get("seq_id", col.get("sequence_name"))
        seq_idx = col.get("matched_sequence",
                          col.get("site_sequence", col.get("sequence")))
        if motif_col is None or seq_col is None or seq_idx is None:
            return rows
        score_col = col.get("score", col.get("site_score"))
        p_col = col.get("p-value", col.get("p_value"))
        need_max = max(x for x in (motif_col, seq_col, seq_idx,
                                   score_col, p_col) if x is not None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= need_max:
                continue
            rows.append({
                "motif_id": parts[motif_col],
                "seq_id": parts[seq_col],
                "matched_sequence": parts[seq_idx].upper(),
                "score": float(parts[score_col]) if score_col is not None else 0.0,
                "p_value": float(parts[p_col]) if p_col is not None else 1.0,
            })
    return rows


def _lcs_len(a: str, b: str) -> int:
    """最长公共子序列长度（用 SequenceMatcher 汇总匹配块，容忍 indel/替换）。

    注意：这里统计所有匹配块的总长度，而非单段最长匹配——单段最长匹配会把
    “中间一个碱基替换”误判成大量错配，导致 support 仍被低估。
    """
    from difflib import SequenceMatcher
    sm = SequenceMatcher(None, a, b, autojunk=False)
    return sum(block.size for block in sm.get_matching_blocks())


def _annotate_hits(df: pd.DataFrame, motifs_df: pd.DataFrame) -> pd.DataFrame:
    """为原始序列表新增 hit_motifs 列：逗号分隔的命中 motif，无命中为空字符串。

    判断方式：motif_seq 是否是 variable_region 的子串（大小写不敏感）。
    motifs_df 已按 support 降序，因此命中列表也按支持度从高到低排列。
    """
    out = df.copy()
    if "variable_region" not in out.columns or motifs_df.empty:
        out["hit_motifs"] = ""
        return out

    motif_seqs = list(dict.fromkeys(
        str(m).strip().upper()
        for m in motifs_df["motif_seq"].tolist()
        if str(m).strip()
    ))
    if not motif_seqs:
        out["hit_motifs"] = ""
        return out

    def _hits(value: object) -> str:
        seq = str(value).upper()
        return ",".join(m for m in motif_seqs if m in seq)

    out["hit_motifs"] = out["variable_region"].fillna("").map(_hits)
    return out


@log_tool("discover_motifs")
def discover_motifs(
    df: pd.DataFrame,
    k: int = 10,
    top_n: int = 5,
    min_support: int = 3,
    docker_image: str = "memesuite/memesuite:latest",
    indel_tolerance: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """调用 STREME 发现保守基序，返回 motif 统计表、带注释的序列表与 meta。

    参数：
        df           : 含 variable_region 列的 DataFrame；若含 cluster_id 则额外统计
        k            : motif 宽度（--minw/--maxw 都设为 k）
        top_n        : STREME 输出的 motif 数量上限（--nmotifs）
        min_support  : support 低于该值的 motif 被过滤（结果后处理）
        docker_image : 使用的 STREME 镜像
        indel_tolerance: 复核 sites.tsv 时允许的 indel 容差（bp）；
                         0 表示要求 matched_sequence 与 consensus 完全一致。

    返回：
        (motifs_df, df_annotated, meta)
        motifs_df 固定三列：motif_seq / support / cluster_ids，按 support 降序；
        df_annotated 为原始序列表的副本（保留 sequence / count / cluster_id 等
        全部原始列），并新增 hit_motifs 列：多个 motif 用逗号分隔，无命中为空字符串；
        meta 含 k、n_motifs、has_cluster_info、params。
    """
    # ---- 1) 输入校验 ----
    if df.empty:
        log.info("输入为空，跳过 STREME 分析")
        return _empty_frame(), _annotate_hits(df, _empty_frame()), {"n_motifs": 0}
    if "variable_region" not in df.columns:
        raise ValueError(f"缺少必要列 variable_region，当前列：{list(df.columns)}")
    if not 3 <= k <= 30:
        raise ValueError(f"k 需在 [3, 30] 之间（STREME 的 --minw/--maxw 约束），收到：{k}")
    if top_n < 1:
        raise ValueError(f"top_n 需 >= 1，收到：{top_n}")
    if min_support < 1:
        raise ValueError(f"min_support 需 >= 1，收到：{min_support}")

    has_cluster = "cluster_id" in df.columns
    seqs = df["variable_region"].fillna("").astype(str).str.strip().tolist()
    cluster_vals = df["cluster_id"].tolist() if has_cluster else []

    if not any(seqs):                    # 序列全为空，无数据可分析
        log.info("variable_region 全为空，跳过 STREME 分析")
        return _empty_frame(), _annotate_hits(df, _empty_frame()), {"n_motifs": 0}

    # ---- 2) 准备临时目录并写入 FASTA ----
    # Windows 下 C 盘 AppData 可能没有 Docker 挂载权限，强制把临时文件放到
    # Docker Desktop 已完成文件共享白名单配置的 D 盘目录。
    share_dir = r"D:\DockerShare"
    os.makedirs(share_dir, exist_ok=True)
    log.info(f"STREME 临时目录根路径：{share_dir}")
    # ignore_cleanup_errors：Windows 下容器/杀毒软件/索引器短暂占用文件时，
    # TemporaryDirectory 退出清理会抛 PermissionError(WinError 32)；忽略清理失败不影响结果
    with tempfile.TemporaryDirectory(
        dir=share_dir, ignore_cleanup_errors=True
    ) as temp_dir:
        # 主机路径必须是绝对路径 + 正斜杠；Windows 默认用 C:/...，
        # 可通过 APTAMER_DOCKER_PATH_STYLE=wsl 切换为 /c/... 形式
        posix_path = _docker_mount_path(temp_dir)
        input_fasta = os.path.join(temp_dir, "input.fasta")
        _write_fasta(seqs, input_fasta)
        log.info(f"STREME 临时目录：{temp_dir}；Docker 挂载路径：{posix_path}")

        # ---- 3) 构造并运行 Docker 命令（列表形式，避免 shell 注入）----
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{posix_path}:/data",
            docker_image,
            "streme",
            "--p", "/data/input.fasta",
            "--oc", "/data/out",
            "--dna",
            "--minw", str(k),
            "--maxw", str(k),
            "--nmotifs", str(top_n),
            "--thresh", "0.05",
        ]
        log.info(f"运行 STREME：{' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,        # 必须捕获，否则失败时拿不到 stderr
                text=True,                  # 直接返回 str，便于写日志
                encoding="utf-8",           # Windows 默认按 GBK 解码 docker 输出，易乱码/抛解码错误
                errors="replace",
                timeout=DOCKER_TIMEOUT,
            )
        except FileNotFoundError:
            log.error("未找到 docker 命令，请确认已安装 Docker 并加入 PATH")
            raise RuntimeError("未找到 docker 命令，请确认已安装 Docker 并加入 PATH") from None
        except subprocess.TimeoutExpired as exc:
            # 超时时 partial 输出在异常对象上，必须打出来才能判断卡在哪一步
            timeout_stdout = _decode_output(exc.stdout)
            timeout_stderr = _decode_output(exc.stderr)
            log.error(
                f"Docker 运行 STREME 超时（>{DOCKER_TIMEOUT}s）。"
                f"stdout: {timeout_stdout}, stderr: {timeout_stderr}"
            )
            raise RuntimeError(
                f"Docker 运行 STREME 超时（>{DOCKER_TIMEOUT}s），请检查 Docker Desktop 是否卡死。"
                f"stdout: {timeout_stdout}\nstderr: {timeout_stderr}"
            ) from None
        except OSError as exc:
            log.error(f"无法启动 docker 命令：{exc}")
            raise RuntimeError(f"无法启动 docker 命令：{exc}") from exc

        # 成功/失败都先记录完整输出，失败时便于定位 Docker / 挂载 / 镜像问题
        log.info(f"STREME stdout：{proc.stdout.strip()[:1000]}")
        if proc.returncode != 0:
            log.error(f"STREME 执行失败，返回码 {proc.returncode}。stdout: {proc.stdout}, stderr: {proc.stderr}")
            raise RuntimeError(
                f"Docker 运行 STREME 失败（退出码 {proc.returncode}）。"
                f"请确认 Docker 已启动、镜像 {docker_image} 已拉取；"
                f"若报 mount 错误，请在 Docker Desktop → Settings → Resources → File sharing "
                f"中允许 {posix_path.split(':')[0] if ':' in posix_path else posix_path} 盘，"
                f"或设置 APTAMER_DOCKER_PATH_STYLE=wsl 改用 /c/... 挂载路径。"
                f"\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
            )
        log.info(f"STREME stderr：{proc.stderr.strip()[:1000]}")

        # ---- 4) 检查并解析 STREME 结果 ----
        out_dir = os.path.join(temp_dir, "out")
        streme_txt = os.path.join(out_dir, "streme.txt")
        if not os.path.isfile(streme_txt):
            listing = os.listdir(out_dir) if os.path.isdir(out_dir) else "输出目录不存在"
            log.error(
                f"STREME 未生成 streme.txt；输出目录内容：{listing}。"
                f"stdout: {proc.stdout}, stderr: {proc.stderr}"
            )
            raise RuntimeError(f"STREME 未生成结果文件：{streme_txt}")

        try:
            # with 结束时句柄立即关闭，避免 Windows 下占用文件导致临时目录清理失败
            with open(streme_txt, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            motifs_raw = _parse_streme_pwm(text)
        except Exception as exc:
            # 解析失败时打印文件前 50 行，便于排查 STREME 版本/格式差异
            try:
                with open(streme_txt, "r", encoding="utf-8", errors="replace") as f:
                    head = "".join(f.readlines()[:50])
            except OSError:
                head = "<streme.txt 读取失败>"
            log.error(f"解析 STREME 结果失败：{exc}\nstreme.txt 前 50 行：\n{head}")
            raise RuntimeError(f"解析 STREME 结果失败：{exc}") from exc
        log.info(f"STREME 解析出 {len(motifs_raw)} 个 motif："
                 f"{[m['consensus'] for m in motifs_raw]}")

        # ---- 5) 读 sites.tsv + LCS 复核，统计 support 与 cluster 信息 ----
        sites_tsv = os.path.join(out_dir, "sites.tsv")
        sites_tsv_found = os.path.isfile(sites_tsv)   # 临时目录退出后会被删，须提前记下
        sites = _parse_sites_tsv(sites_tsv)

        # consensus 查表：motif_ID（形如 "1-CCCAAA"）→ consensus
        consensus_by_id = {m["id"]: m["consensus"] for m in motifs_raw if m["id"]}

        # 按 motif_ID 聚合命中的 seq 行号（seq_ID 形如 "seq_5"）
        hits_by_motif: dict[str, set[int]] = {}
        for s in sites:
            c = consensus_by_id.get(s["motif_id"])
            if c is None:
                continue
            # LCS 复核：matched_sequence 与 consensus 差异不超过 indel_tolerance
            if _lcs_len(c, s["matched_sequence"]) < len(c) - indel_tolerance:
                continue
            try:
                idx = int(s["seq_id"].split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            hits_by_motif.setdefault(s["motif_id"], set()).add(idx)

        rows: list[dict] = []
        for m in motifs_raw:
            hits = sorted(hits_by_motif.get(m["id"], set()))
            if len(hits) < min_support:
                continue
            if has_cluster:
                values = {cluster_vals[i] for i in hits
                          if not pd.isna(cluster_vals[i])}
                cluster_ids = sorted(int(v) for v in values)
            else:
                cluster_ids = []
            rows.append({
                "motif_seq": m["consensus"],
                "support": len(hits),
                "cluster_ids": cluster_ids,
            })

        rows.sort(key=lambda r: -r["support"])
        motifs_df = pd.DataFrame(rows, columns=COLUMNS) if rows else _empty_frame()

    # ---- 6) 把命中信息合并回原始序列表 ----
    df_annotated = _annotate_hits(df, motifs_df)

    # ---- 7) meta ----
    meta = {
        "k": k,
        "n_motifs": len(motifs_df),
        "has_cluster_info": has_cluster,
        "params": {
            "k": k, "top_n": top_n, "min_support": min_support,
            "docker_image": docker_image,
            "indel_tolerance": indel_tolerance,
        },
        # 附加统计：便于在 trace 中查看 STREME 原始输出与过滤损耗
        "n_sequences": len(seqs),
        "n_motifs_raw": len(motifs_raw),
        "sites_tsv_found": sites_tsv_found,
    }
    log.info(f"motif 分析完成：{len(seqs)} 条序列 -> STREME 输出 {len(motifs_raw)} 个，"
             f"过滤后 {len(motifs_df)} 个（min_support={min_support}，"
             f"indel_tolerance={indel_tolerance}）"
             + ("，已统计 cluster 信息" if has_cluster else "，无 cluster 信息"))
    return motifs_df, df_annotated, meta
