"""
features.py — Feature engineering & tiền xử lý, hỗ trợ song song 2 "khẩu vị" model.

TRIẾT LÝ VẬN HÀNH:
  - Thiết lập tường lửa cô lập FEATURE độc lập khỏi ID/Treatment/Outcome chống rò rỉ dữ liệu (leakage).
  - Khuyết tật dữ liệu trong Consumer Finance mang biến số hành vi (Ví dụ: KH chưa từng có khoản vay).
    Hệ thống tạo lập thêm "Missing Flag" thay vì drop bỏ các cột này.
  - Tối giản hóa cấu trúc (No Over-engineering): Chỉ biến đổi log1p giảm lệch phải, băm tuổi theo nhóm cố định.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from . import data_io as io


# ============================================================================
# 1) CÁC BƯỚC ENGINEER (Tạo đặc trưng mới, bảo lưu cột gốc)
# ============================================================================

def high_missing_cols(df: pd.DataFrame, threshold: float = None) -> List[str]:
    """Xác định danh sách các cột tính năng có tỷ lệ khuyết vượt ngưỡng quy định trên tập TRAIN."""
    threshold = getattr(C, 'MISSING_FLAG_THRESHOLD', 0.3) if threshold is None else threshold
    
    # Sử dụng hàm bổ trợ split_feature_types vừa tạo để quét động danh sách biến
    num_cols, cat_cols = io.split_feature_types(df)
    feats = [c for c in num_cols + cat_cols if c not in getattr(C, 'NON_FEATURE_COLS', [])]
    
    miss = df[feats].isna().mean()
    return sorted(miss[miss >= threshold].index.tolist())


def add_missing_flags(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Thêm cột chỉ định <col>_ismiss (1 nếu thiếu, 0 nếu có) làm tín hiệu nghiệp vụ độc lập."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[f"{c}_ismiss"] = df[c].isna().astype(int)
    return df


def add_log_features(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    """Áp dụng log1p cho các cột phân bổ số tiền/khối lượng lệch phải mạnh để triệt tiêu nhiễu outlier."""
    cols = getattr(C, 'AMOUNT_FEATURES', []) if cols is None else cols
    df = df.copy()
    for c in cols:
        if c in df.columns:
            # Ép chặn dưới về 0 trước khi tính log1p phòng vệ dữ liệu bẩn dính số âm
            df[f"{c}_log"] = np.log1p(df[c].clip(lower=0))
    return df


def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    """Băm nhỏ cột tuổi thành các nhóm cố định để mô hình học các quan hệ phi tuyến dễ dàng."""
    df = df.copy()
    if "cli_age" in df.columns:
        df["cli_age_band"] = pd.cut(
            df["cli_age"],
            bins=[0, 25, 30, 35, 40, 50, 200],
            labels=["<=25", "26-30", "31-35", "36-40", "41-50", "50+"],
        ).astype("object")
    return df


def engineer_features(df: pd.DataFrame, miss_cols: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Pipeline tổng hợp thực thi toàn bộ các bước tạo lập tính năng mới.
    
    Lưu ý: Đối với tập TEST, bắt buộc phải truyền danh sách `miss_cols` thu được từ tập TRAIN 
    để đảm bảo cấu trúc cột đồng nhất tuyệt đối.
    """
    if miss_cols is None:
        miss_cols = high_missing_cols(df)
    out = add_missing_flags(df, miss_cols)
    out = add_log_features(out)
    out = add_age_band(out)
    return out, miss_cols


# ============================================================================
# 2) KHẨU VỊ MÔ HÌNH TREE-BASED: CATBOOST INPUTS
# ============================================================================

def get_catboost_inputs(df_engineered: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Trích xuất ma trận X và danh sách nhãn categorical dành riêng cho cấu trúc của CatBoost."""
    non_feat_cols = getattr(C, 'NON_FEATURE_COLS', [])
    feats = [c for c in df_engineered.columns if c not in non_feat_cols and not c.endswith("__strata")]
    X = df_engineered[feats].copy()

    # Nhận diện biến chữ linh hoạt dựa trên kiểm tra kiểu dữ liệu của Pandas API
    cat_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cat_features:
        # Ép chuỗi và gán giá trị sentinel hờ để Catboost không dính lỗi NaN biến phân loại
        X[c] = X[c].astype("object")
        X[c] = X[c].where(pd.notna(X[c]), "__NA__").astype(str)
    return X, cat_features


# ============================================================================
# 3) KHẨU VỊ MÔ HÌNH TUYẾN TÍNH / SKLEARN (MA TRẬN SỐ PHẲNG)
# ============================================================================

@dataclass
class SklearnPreprocessor:
    """Bộ xử lý trung gian thiết lập ma trận số học hoàn chỉnh cho các mô hình Sklearn."""
    scale: bool = True
    min_frequency: int = 50           # Gộp các nhóm xuất hiện ít hơn 50 lần vào biến chung 'infrequent'
    ct: ColumnTransformer = field(default=None, init=False, repr=False)
    num_cols: List[str] = field(default_factory=list, init=False)
    cat_cols: List[str] = field(default_factory=list, init=False)
    feature_names_: List[str] = field(default_factory=list, init=False)

    def _split_cols(self, df: pd.DataFrame):
        non_feat_cols = getattr(C, 'NON_FEATURE_COLS', [])
        feats = [c for c in df.columns if c not in non_feat_cols]
        cat = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
        num = [c for c in feats if c not in cat]
        return num, cat

    def fit(self, df: pd.DataFrame) -> SklearnPreprocessor:
        self.num_cols, self.cat_cols = self._split_cols(df)

        # Biến số liên tục: Điền khuyết bằng Trung vị + Thêm cờ báo khuyết -> Chuẩn hóa z-score
        num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if self.scale:
            num_steps.append(("scale", StandardScaler()))
        num_pipe = Pipeline(num_steps)

        # Biến phân loại: Điền khuyết bằng nhóm xuất hiện nhiều nhất -> Mã hóa One-Hot
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=self.min_frequency,
                sparse_output=False,
            )),
        ])

        self.ct = ColumnTransformer(
            [("num", num_pipe, self.num_cols),
             ("cat", cat_pipe, self.cat_cols)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.ct.fit(df)
        self.feature_names_ = list(self.ct.get_feature_names_out())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.ct is None:
            raise RuntimeError("Hệ thống lỗi: Cần gọi thực thi fit() trước khi transform().")
        return self.ct.transform(df)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


# ============================================================================
# 4) KHAI BÁO BÁO CÁO HỆ THỐNG
# ============================================================================

def feature_engineering_summary(df_raw: pd.DataFrame, df_eng: pd.DataFrame, miss_cols: List[str]) -> dict:
    """Tổng hợp số liệu các đặc trưng biến đổi cấu trúc phục vụ kết xuất báo cáo tổng."""
    non_feat_cols = getattr(C, 'NON_FEATURE_COLS', [])
    num_cols, cat_cols = io.split_feature_types(df_raw)
    raw_feats = [c for c in num_cols + cat_cols if c not in non_feat_cols]
    
    added = [c for c in df_eng.columns if c not in df_raw.columns]
    return {
        "n_raw_features": len(raw_feats),
        "n_missing_flags_added": sum(c.endswith("_ismiss") for c in added),
        "n_log_features_added": sum(c.endswith("_log") for c in added),
        "added_features": added,
        "high_missing_cols_used": miss_cols,
        "dropped_non_features": non_feat_cols,
    }


if __name__ == "__main__":
    df = io.load_train()
    eng, miss = engineer_features(df)
    Xcb, cats = get_catboost_inputs(eng)
    print("Engineered shape:", eng.shape)
    print("High-missing cols (flagged):", len(miss))
    print("CatBoost X shape:", Xcb.shape, "| #cat:", len(cats))
    pre = SklearnPreprocessor().fit(eng)
    Xsk = pre.transform(eng)
    print("Sklearn matrix shape:", Xsk.shape, "| #out features:", len(pre.feature_names_))