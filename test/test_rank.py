# =============================================================================
# test/test_rank.py
# 目的：验证 rank.py 的多维度加权评分、缺失列兼容处理以及分组择优策略
# =============================================================================
import pandas as pd
import pytest
from core.rank import rank_candidates


@pytest.fixture
def sample_rank_data():
    """构造模拟的序列属性和两两比对数据"""
    df_seq = pd.DataFrame({
        "sequence": ["seq1", "seq2", "seq3", "seq4"],
        "group_id": ["A", "A", "B", "B"],  # A组两条，B组两条
        "count": [100, 80, 60, 40],
        "mfe_full": [-10.5, -15.0, -8.0, -12.0],
        "self_similarity": [0.8, 0.9, 0.6, 0.7],
        "motif_seq": ["Motif1", None, "Motif1", None]
    })

    df_pairs = pd.DataFrame({
        "group_id": ["A", "B"],
        "seq_A": ["seq1", "seq3"],
        "seq_B": ["seq2", "seq4"],
        "trimmed_sim": [0.85, 0.60]
    })

    return df_seq, df_pairs


def test_rank_basic(sample_rank_data):
    """测试基础评分和全局排序（不传 df_pairs，测试降级处理）"""
    df_seq, _ = sample_rank_data
    # 不传 df_pairs，代码应能自动忽略 group_stability 权重
    df_ranked, meta = rank_candidates(df_seq, df_pairs=None, top_n=2)

    assert len(df_ranked) <= 2
    assert "final_score" in df_ranked.columns
    # 验证权重自动归一化（缺失 group_stability 后，剩余权重总和应为 1）
    assert abs(sum(meta["weights_used"].values()) - 1.0) < 1e-6


def test_rank_stratified_sampling(sample_rank_data):
    """测试分层抽样：如果 top_n=2 且有 2 个组，结果必须每组各出 1 条，不能全来自 A 组"""
    df_seq, df_pairs = sample_rank_data
    df_ranked, meta = rank_candidates(df_seq, df_pairs=df_pairs, top_n=2)

    assert meta["strategy"] == "stratified"
    assert len(df_ranked) == 2
    # 关键断言：选出来的 2 条序列必定分别来自 A 组和 B 组
    assert set(df_ranked["group_id"]) == {"A", "B"}


def test_rank_group_stability_calc(sample_rank_data):
    """测试组内稳定性计算：验证 df_pairs 的相似度是否被成功合并进主表"""
    df_seq, df_pairs = sample_rank_data
    # 将 top_n 设大一点，看看选出来的到底是谁
    df_ranked, _ = rank_candidates(df_seq, df_pairs=df_pairs, top_n=4)

    # 打印出来看看最终表长什么样
    print("\n--- 最终排名结果 ---")
    print(df_ranked[["sequence", "group_id", "final_score", "group_stability"]])

    # 验证：因为 A 组只能留一条，seq2 得分大概率比 seq1 高。
    # 所以只要在结果里找到 A 组的胜利者，检查它的分数即可。
    a_group_winner = df_ranked[df_ranked["group_id"] == "A"].iloc[0]

    # 它的 group_stability 应该是 0.85（因为它和同组另一条比对的 trimmed_sim 是 0.85）
    assert abs(a_group_winner["group_stability"] - 0.85) < 1e-6
    print(f"✅ A组胜出的是: {a_group_winner['sequence']}，其 group_stability 为 0.85")