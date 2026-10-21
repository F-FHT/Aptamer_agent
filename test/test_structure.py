import pytest
import pandas as pd
from core.structure import compare_structures


def test_structure_direct_no_primers():
    """测试场景1：直接输入（无引物），不比较前后结构，只测全长两两比对"""
    seqs = ["GGGAGCTCAGAATAAACGCTCAA", "GGGAGCTCAGATAAACGCTCAA"]
    df_seq, df_pairs, meta = compare_structures(
        input_data=seqs,
        source="direct",
        fwd_primer="",
        rev_primer="",
        do_self_compare=False,
        do_pair_compare=True
    )
    assert len(df_seq) == 2
    assert "mfe_full" in df_seq.columns
    assert len(df_pairs) == 1  # 2条序列应该有1对比较
    assert df_pairs.iloc[0]["full_sim"] is not None


def test_structure_with_primers():
    """测试场景2：带引物，比较去引物前后的结构变化"""
    seqs = [
        "GGGAGCTCAGAATAAACGCTCAA",
        "GGGAGCTCAGATAAACGCTCAA"
    ]
    # 截取开头的GGGAG和结尾的TCAA
    df_seq, _, meta = compare_structures(
        input_data=seqs,
        source="direct",
        fwd_primer="GGGAG",
        rev_primer="TCAA",
        do_self_compare=True,
        do_pair_compare=False
    )
    assert len(df_seq) == 2
    # 检查结构是否被截短
    assert df_seq.iloc[0]["structure_trimmed"] != df_seq.iloc[0]["structure_full"]
    # self_similarity 应该被计算出来
    assert df_seq.iloc[0]["self_similarity"] is not None


def test_structure_from_cluster():
    """测试场景3：输入聚类后的 DataFrame（取前3）"""
    df_cluster = pd.DataFrame({
        "variable_region": ["GGGAGCTCAGAATAAACGCTCAA", "GGGAGCTCAGATAAACGCTCAA", "GGGAGCTCAGAATAAACGC",
                            "TTCGACAGGAGGCTCACAACAGGC"],
        "cluster_id": [0, 0, 0, 1],
        "count": [100, 90, 80, 50]
    })

    df_seq, df_pairs, meta = compare_structures(
        input_data=df_cluster,
        source="cluster",
        do_self_compare=False,
        do_pair_compare=True
    )
    # cluster 0 有三条，取前3，所以 df_seq 应该有 3 + 1 = 4 条
    assert len(df_seq) == 4
    assert meta["n_groups"] == 2