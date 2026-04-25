import sys
import types

import pandas as pd
import pytest


fake_s3_module = types.ModuleType("scripts.s3_scripts.read_write_to_s3")
fake_s3_module.read_csv_from_s3 = lambda *args, **kwargs: None
fake_s3_module.write_df_to_s3 = lambda *args, **kwargs: None
sys.modules["scripts.s3_scripts.read_write_to_s3"] = fake_s3_module

from scripts.vn_index_scripts import vn_index_preprocessing


def test_process_data_ignores_empty_ancillary_columns(monkeypatch):
    written = {}

    def fake_write_df_to_s3(df, bucket, key, **kwargs):
        written["df"] = df.copy()
        written["bucket"] = bucket
        written["key"] = key

    monkeypatch.setattr(vn_index_preprocessing, "write_df_to_s3", fake_write_df_to_s3)

    raw_df = pd.DataFrame(
        {
            "Date": ["04/24/2026", "04/25/2026"],
            "VN-INDEX": ["1,234.56", "1,240.00"],
            "Total Volume": ["123,456", "130,000"],
            "Total Value": ["1.50 bil", "1.75 bil"],
            "Foreign Buy Volume": [pd.NA, pd.NA],
        }
    )

    cleaned_df = vn_index_preprocessing.process_data(raw_df)

    assert len(cleaned_df) == 2
    assert written["bucket"] == "vn-index"
    assert written["key"] == "ready_data/vn_index_data/cleaned_vn_index_data.csv"
    assert list(cleaned_df.columns) == [
        "Date",
        "VN_Index_Close",
        "Total Volume",
        "Total Value",
    ]
    assert cleaned_df["VN_Index_Close"].tolist() == [1234.56, 1240.0]
    assert cleaned_df["Total Volume"].tolist() == [123456.0, 130000.0]
    assert cleaned_df["Total Value"].tolist() == [1.5e9, 1.75e9]


def test_process_data_refuses_to_write_empty_cleaned_data(monkeypatch):
    def fail_write_df_to_s3(*args, **kwargs):
        raise AssertionError("empty cleaned data should not be written")

    monkeypatch.setattr(vn_index_preprocessing, "write_df_to_s3", fail_write_df_to_s3)

    raw_df = pd.DataFrame(
        {
            "Date": [pd.NA],
            "VN-INDEX": [pd.NA],
            "Total Volume": [pd.NA],
            "Total Value": [pd.NA],
            "Foreign Buy Volume": [pd.NA],
        }
    )

    with pytest.raises(ValueError, match="No valid VN-Index rows remain"):
        vn_index_preprocessing.process_data(raw_df)
