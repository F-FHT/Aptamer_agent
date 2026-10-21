import pandas as pd
from core.cluster import cluster_sequences


def test_cluster_logic():
    """测试基础聚类和 Indel 容忍度"""
    test_data = {
        "variable_region": [
            "GGGAGCTCAGAATAAACGCTCAA",  # 代表
            "GGGAGCTCAGAATAAACGCTCAA",  # 完全一致
            "GGGAGCTCAGAAAAACGCTCAC",  # 错配
            "GGGAGCTCAGATAAACGCTCAA",  # 插入 (多了一个T)
            "TTCGACAGGAGGCTCACAACAGGC"  # 完全不同
        ],
        "count": [100, 80, 60, 50, 20]
    }
    df = pd.DataFrame(test_data)

    # 调用聚类函数：阈值设为 0.8，并且要求最小簇大小至少为 3
    clustered_df, meta = cluster_sequences(df, identity=0.8, min_cluster_size=3)

    # ---- 断言 1：只有大簇被保留 ----
    # 因为单条序列的簇被过滤了，所以现在应该只有 1 个簇
    assert meta['n_clusters'] == 1, f"预期分1个簇，实际分了 {meta['n_clusters']} 个簇"

    # ---- 断言 2：验证过滤统计（损耗数据） ----
    # 被丢掉了 1 个簇（原本的独立簇）
    assert meta['n_dropped_clusters'] == 1, f"预期丢弃1个簇，实际丢弃 {meta['n_dropped_clusters']} 个"
    # 被丢掉了 1 条序列（那个独立序列）
    assert meta['n_dropped_sequences'] == 1, f"预期丢弃1条序列，实际丢弃 {meta['n_dropped_sequences']} 条"

    # ---- 断言 3：验证输出的行数 ----
    # 原本 5 行，过滤掉 1 行，应该只剩 4 行
    assert len(clustered_df) == 4, f"预期输出4条序列，实际输出 {len(clustered_df)} 条"

    # ---- 断言 4：带有 Indel 的序列依然在 Cluster 0 里 ----
    indel_seq = "GGGAGCTCAGAATAAACGCTCAA"
    # 注意：因为你刚才过滤后，代码里做了重编号，原来的0号簇还是0号，所以这里还是取 0
    indel_cluster = clustered_df[clustered_df['variable_region'] == indel_seq]['cluster_id'].values[0]
    assert indel_cluster == 0, "Indel 处理失败或重编号错误"

    # ---- 断言 5：代表序列的 count 依然是最高的 ----
    rep_seq = clustered_df[clustered_df['is_representative'] == True]['variable_region'].values[0]
    assert rep_seq == "GGGAGCTCAGAATAAACGCTCAA", "代表序列选择错误"

    print("\n✅ 所有测试通过！小簇过滤和重编号完全正确。")