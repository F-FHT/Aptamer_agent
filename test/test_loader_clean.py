# core/tests/test_clean_flow.py
# 目的：用构造的 DataFrame 和临时 Excel 验证 loader → clean 能端到端跑通。
# 运行：pytest core/tests/test_clean_flow.py -v
import pandas as pd
import pytest

from core.loader import load_sequences, ColumnError
from core.clean import clean_sequences

@pytest.fixture
def sample_excel(tmp_path):
    """构造一个模拟测序结果 Excel：含 3 条有效序列 + 1 条空行 + 1 条短序列。"""
    data = {
        "Sequence": [
            "AAAACCCCGGGGTTTT",   # 16nt，会被长度过滤掉
            "ACGTTGCAACGT",       # 12nt，会被长度过滤掉
            "AAAACCCCGGGGTTTTACGT" * 2,  # 40nt，有效
            "GGGGCCCCAAAATTTTACGT" * 2,  # 40nt，有效（与上不同）
            "",                    # 空行，应被剔除
        ],
        "Reads": ["10", "5", "100", "80", "0"],
    }
    path = tmp_path / "test.xlsx"
    pd.DataFrame(data).to_excel(path, index=False)
    return str(path)


def test_load_basic(sample_excel):
    df, report = load_sequences(sample_excel, seq_col="Sequence", count_col="Reads")
    assert list(df.columns) == ["raw_seq", "count"]
    assert report["n_rows_raw"] == 5
    assert report["n_rows_kept"] == 4          # 空行被剔除


def test_load_wrong_column(sample_excel):
    with pytest.raises(ColumnError, match="不存在"):
        load_sequences(sample_excel, seq_col="WrongCol")


def test_clean_filters_by_length(sample_excel):
    df, _ = load_sequences(sample_excel, seq_col="Sequence", count_col="Reads")
    # 40nt 序列，引物用头尾各 4nt
    cleaned = clean_sequences(
        df,
        fwd_primer="AAAA", rev_primer="ACGT",
        target_len=40, tolerance=0.2, top_n=50,
    )
    assert len(cleaned) > 0
    assert (cleaned["variable_region"].str.len().between(32, 48)).all()
    assert cleaned["count"].is_monotonic_decreasing


def test_clean_top_n(sample_excel):
    df, _ = load_sequences(sample_excel, seq_col="Sequence", count_col="Reads")
    cleaned = clean_sequences(
        df, fwd_primer="AAAA", rev_primer="ACGT", top_n=1,
    )
    assert len(cleaned) <= 1