"""
data_io.py — Đọc/ghi dữ liệu một cách an toàn, nhất quán và tối ưu bộ nhớ.

Mục tiêu: Giữ nguyên cơ chế ép kiểu giảm RAM (Downcast) của bạn, đồng thời thiết lập
tường lửa cô lập FEATURE độc lập khỏi ID/Treatment/Outcome chống rò rỉ dữ liệu (leakage).
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Đọc cấu hình tập trung từ file config
from . import config as C

# ----------------------------------------------------------------------------
# LOGIC LÀM SẠCH & TỐI ƯU BỘ NHỚ RAM
# ----------------------------------------------------------------------------
def load_raw_data(is_train: bool = True) -> pd.DataFrame:
    """
    Đọc dữ liệu thô từ đường dẫn cấu hình trong config.py và thực hiện chuẩn hóa nền.
    """
    file_path = C.TRAIN_CSV if is_train else C.TEST_CSV
    
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {file_path}")
        
    mode = "TRAIN" if is_train else "TEST"
    print(f"--> Đang tải dữ liệu thô [{mode}] từ: {file_path.name}...")
    
    df = pd.read_csv(file_path)
    
    # 1) Bỏ cột index thừa nếu có (Unnamed: 0 hoặc trống)
    index_names = getattr(C, 'INDEX_COL_NAMES', ["Unnamed: 0", "index", ""])
    drop_idx = [c for c in df.columns if c in index_names or c.startswith("Unnamed")]
    if drop_idx:
        df = df.drop(columns=drop_idx)
        
    # 2) ĐỒNG BỘ QUAN TRỌNG: Chuẩn hóa cột phân bổ chiến dịch (Xóa khoảng trắng, ép chữ thường)
    if C.TREATMENT_COL in df.columns:
        df[C.TREATMENT_COL] = df[C.TREATMENT_COL].astype(str).str.strip().str.lower()
        
    print(f"    Kích thước dữ liệu {mode} sau tiền xử lý nền: {df.shape[0]} dòng, {df.shape[1]} cột.")
    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tối ưu dung lượng RAM an toàn mà KHÔNG tự ý điền khuyết (fillna).
    Sử dụng 'Int8' để giữ nguyên các giá trị rỗng (NaN/NA) cho bước sau.
    """
    df_optimized = df.copy()
    
    # Lấy danh sách cờ hoặc đích từ config an toàn
    flag_features = getattr(C, 'FLAG_FEATURES', [])
    target_cols_dict = getattr(C, 'TARGET_COLS', {})
    target_cols_list = list(target_cols_dict.values()) if isinstance(target_cols_dict, dict) else []
    
    # Thêm trường hợp nếu bạn đặt thẳng tên cột target trong config dạng chuỗi
    cash_out = getattr(C, 'CASH_OUTCOME', 'cash_sign_30d')
    card_out = getattr(C, 'CARD_OUTCOME', 'card_sign_30d')
    targets = target_cols_list + [cash_out, card_out]

    for col in df_optimized.columns:
        if col in flag_features or col in targets:
            df_optimized[col] = df_optimized[col].astype('Int8')
        elif df_optimized[col].dtype == 'int64':
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        elif df_optimized[col].dtype == 'float64':
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
            
    return df_optimized


# ----------------------------------------------------------------------------
# CÁC HÀM TIỆN ÍCH TRÍCH XUẤT (SHORTCUT FUNCTIONS)
# ----------------------------------------------------------------------------
def load_train() -> pd.DataFrame:
    """Hàm nạp và tối ưu tập dữ liệu TRAIN."""
    df = load_raw_data(is_train=True)
    return optimize_memory(df)


def load_test() -> pd.DataFrame:
    """Hàm nạp và tối ưu tập dữ liệu TEST."""
    df = load_raw_data(is_train=False)
    
    # Phòng vệ: Tạo nhãn giả lập NaN cho tập Test nếu file Test bị ẩn nhãn khi Scoring
    cash_out = getattr(C, 'CASH_OUTCOME', 'cash_sign_30d')
    card_out = getattr(C, 'CARD_OUTCOME', 'card_sign_30d')
    for ycol in (cash_out, card_out):
        if ycol not in df.columns:
            df[ycol] = np.nan
            
    return optimize_memory(df)


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Lưu dữ liệu sau khi đã làm sạch vào thư mục outputs."""
    output_path = C.OUTPUT_DIR / filename
    print(f"--> Đang lưu dữ liệu đã xử lý vào: {output_path}...")
    df.to_csv(output_path, index=False)
    print("    Lưu file thành công!")


# ----------------------------------------------------------------------------
# TƯỜNG LỬA PHÂN LOẠI CỘT (CRITICAL FOR LEAKAGE PREVENTION)
# ----------------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Trả về danh sách cột feature thực sự = mọi cột trừ id/treatment/outcome/index rác.
    Đây là hàng rào bảo vệ chặn đứng hoàn toàn Data Leakage.
    """
    non_feat_cols = getattr(C, 'NON_FEATURE_COLS', [])
    return [c for c in df.columns if c not in non_feat_cols and not c.endswith("__strata")]


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Phân tách tự động danh sách cột tính năng (chỉ lấy Feature) thành 2 nhóm độc lập:
    Biến số (Numeric) và Biến chữ/Phân loại (Categorical).
    """
    # Lấy danh sách biến tính năng sạch qua bộ lọc tường lửa trước
    feats = get_feature_columns(df)
    
    # Lọc cấu hình categorical nếu được định nghĩa trước
    cat_config = getattr(C, 'CATEGORICAL_FEATURES', [])
    
    num_cols = []
    cat_cols = []
    
    for c in feats:
        # Sử dụng API an toàn của Pandas kiểm tra bản chất số học (tương thích cả Pandas 3.0)
        is_num = pd.api.types.is_numeric_dtype(df[c])
        if c in cat_config or not is_num:
            cat_cols.append(c)
        else:
            num_cols.append(c)
            
    return num_cols, cat_cols


def make_treatment_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cặp cờ treatment nhị phân đối xứng phục vụ mô hình học máy Uplift."""
    df = df.copy()
    treat_cash = getattr(C, 'TREAT_CASH', 'cash')
    treat_card = getattr(C, 'TREAT_CARD', 'card')
    
    df["T_cash"] = (df[C.TREATMENT_COL] == treat_cash).astype(int)
    df["T_card"] = (df[C.TREATMENT_COL] == treat_card).astype(int)
    return df