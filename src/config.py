"""
config.py — Cấu hình trung tâm cho toàn bộ project DAZONE 2026 Vòng 3.1.

Vì sao có file này:
- Tránh hard-code path & tên cột rải rác trong notebook (dễ sai, khó sửa).
- Một chỗ duy nhất để đổi seed, đường dẫn, danh sách cột leakage.
- Mọi notebook/script đều `from src.config import *` để dùng chung 1 nguồn sự thật.

Người mới đọc lưu ý: file này KHÔNG chạy model, nó chỉ khai báo hằng số.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------------
# 1) SEED — để kết quả tái lập (chạy lại ra y hệt)
# ----------------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------------
# 2) PATHS — luôn tính từ gốc repo, giữ nguyên data/raw thực tế của bạn
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data/raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# File dữ liệu (đặt tên theo file BTC cung cấp)
TRAIN_CSV = DATA_DIR / "dazone_sample.csv"
TEST_CSV = DATA_DIR / "dazone_sample_test.csv"

# Tạo sẵn thư mục output nếu chưa có (chạy trực tiếp tuần tự cực nhẹ)
for _d in (OUTPUT_DIR, CHART_DIR, TABLE_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 3) ĐỊNH NGHĨA CỘT — treatment / outcome / id
# ----------------------------------------------------------------------------
ID_COL = "customer_id"
TREATMENT_COL = "treatment"          # giá trị: 'cash' hoặc 'card'
CASH_OUTCOME = "cash_sign_30d"       # KH ký Cash Loan trong 30 ngày
CARD_OUTCOME = "card_sign_30d"       # KH mở Credit Card trong 30 ngày

# Cột index thừa khi đọc CSV (cột '' đầu tiên: 0,1,2,...)
INDEX_COL_NAMES = ["Unnamed: 0", ""]

# Giá trị treatment đồng bộ chuỗi chữ thường
TREAT_CASH = "cash"
TREAT_CARD = "card"

# Cấu hình target dạng dict cũ của bạn (phục vụ multi-target pipeline)
TARGET_COLS = {
    "cash": "cash_sign_30d",
    "card": "card_sign_30d"
}

# ----------------------------------------------------------------------------
# 4) CỘT KHÔNG ĐƯỢC DÙNG LÀM FEATURE (leakage / id / target)
# ----------------------------------------------------------------------------
# - id: không có ý nghĩa dự đoán, chỉ định danh.
# - treatment: là biến can thiệp; trong S-Learner dùng có kiểm soát,
#   trong T-Learner dùng để CHIA nhóm, KHÔNG bỏ thẳng vào X.
# - outcome: là cái ta dự đoán; bỏ vào X là leakage trắng trợn.
NON_FEATURE_COLS = [
    ID_COL, 
    TREATMENT_COL, 
    CASH_OUTCOME, 
    CARD_OUTCOME,
    "target_cash", 
    "target_card", 
    "y_response"
] + INDEX_COL_NAMES

# ----------------------------------------------------------------------------
# 5) PHÂN LOẠI FEATURE (X) THEO ĐẶC TÍNH HÌNH HỌC DỮ LIỆU
# ----------------------------------------------------------------------------

# A. Categorical: kiểu chữ / nhóm hạng thuần túy. Sẽ encode hoặc để CatBoost xử lý native.
# (worst_credit_status KHÔNG nằm đây: data dictionary nói nó là ordinal 1..6
#  1=Unclassified ... 6=Write-off → để dạng numeric ordinal có ý nghĩa hơn.)
CATEGORICAL_FEATURES = [
    "cli_job",
    "cli_marital_status",
    "cli_contact_region_level",
    "last_process_channel",
    "risk_grp",                         # nhóm rủi ro (A..E), ordinal-ish dạng chữ
]

# B. Ordinal numeric: là số có thứ tự, KHÔNG one-hot. worst_credit_status 1..6.
ORDINAL_NUMERIC_FEATURES = [
    "worst_credit_status",
]

# C. Ordinal string: biến thứ bậc lưu ở dạng Chữ - Cần xử lý riêng tránh mất thứ tự business
ORDINAL_STRING_FEATURES = [
    "cli_education",                    # ordinal: ELEMENTARY<PTCS<PTTH<BACHELORS<MASTERS
    "cli_contact_region_population_grp", # ordinal: RURAL<SUB_URBAN<URBAN<KEY_URBAN
]

# D. Cờ nhị phân (0/1) — về bản chất là số nhưng ý nghĩa là Yes/No.
FLAG_FEATURES = [
    "f_approve_last_process",
    "f_appointment_3m",
    "f_ever_default",
]

# E. Cột tiền/khối lượng (lệch phải mạnh → cân nhắc log transform).
AMOUNT_FEATURES = [
    "min_sign_vol",
    "avg_sign_vol",
    "max_sign_vol",
    "last_offer_limit",
    "amt_terminated_loan_12m",
]

# F. Toàn bộ biến số (Numerical Features) - Giữ nguyên 100% danh sách biến đầy đủ của bạn
NUMERICAL_FEATURES = [
    "min_sign_vol", "avg_sign_vol", "max_sign_vol",
    "cnt_month_from_last_appl", 
    "cli_age", 
    "cli_emp_period", 
    "last_offer_limit", 
    "cnt_connected_tls", 
    "cnt_fail_tls_3m", 
    "cnt_total_payment", 
    "ratio_monthly_amt_l6m_vs_n6m", 
    "cnt_month_status1", 
    "ratio_vol_hcvn_cash_appl_12m", 
    "ratio_conn_tls_weekend_12m", 
    "ratio_conn_tls_endmonth_12m", 
    "cnt_bank_inst_appl_12m", 
    "cnt_fi_inst_appl_12m", 
    "cnt_report_bank_6m", 
    "cnt_report_institution_12m", 
    "cnt_lender_has2pay_last_24m", 
    "max_cntr_len_fi_12m", 
    "amt_terminated_loan_12m", 
    "cnt_terminated_loan_12m", 
    "cnt_month_over3_credit_card_6m", 
    "cnt_noinsurance_loan_24m", 
    "cnt_earlyterminated_cash", 
    "cnt_earlyterminted_pos", 
    "cnt_inbound_tls_12m", 
    "cnt_month_from_first_transaction", 
    "cnt_month_from_last_transaction"
]

# ----------------------------------------------------------------------------
# 6) NHÓM FEATURE THEO Ý NGHĨA BUSINESS (dùng cho EDA & report)
# ----------------------------------------------------------------------------
FEATURE_GROUPS = {
    "demographic": [
        "cli_age", "cli_job", "cli_marital_status", "cli_education",
    ],
    "region": [
        "cli_contact_region_level", "cli_contact_region_population_grp",
    ],
    "employment_offer": [
        "cli_emp_period", "last_offer_limit",
    ],
    "prev_application": [
        "cnt_month_from_last_appl", "f_approve_last_process",
        "last_process_channel",
    ],
    "telesales": [
        "cnt_connected_tls", "cnt_fail_tls_3m", "f_appointment_3m",
        "cnt_inbound_tls_12m", "ratio_conn_tls_weekend_12m",
        "ratio_conn_tls_endmonth_12m",
    ],
    "risk_credit": [
        "risk_grp", "worst_credit_status", "f_ever_default",
        "cnt_month_status1", "cnt_lender_has2pay_last_24m",
    ],
    "loan_payment_history": [
        "cnt_total_payment", "amt_terminated_loan_12m",
        "cnt_terminated_loan_12m", "cnt_noinsurance_loan_24m",
        "cnt_earlyterminated_cash", "cnt_earlyterminted_pos",
        "max_cntr_len_fi_12m",
    ],
    "bureau_signsvol": [
        "min_sign_vol", "avg_sign_vol", "max_sign_vol",
        "cnt_bank_inst_appl_12m", "cnt_fi_inst_appl_12m",
        "cnt_report_bank_6m", "cnt_report_institution_12m",
    ],
    "cross_sell_behavior": [
        "cnt_month_over3_credit_card_6m", "ratio_vol_hcvn_cash_appl_12m",
        "ratio_monthly_amt_l6m_vs_n6m",
        "cnt_month_from_first_transaction", "cnt_month_from_last_transaction",
    ],
}

# ----------------------------------------------------------------------------
# 7) NGƯỠNG & THAM SỐ MẶC ĐỊNH
# ----------------------------------------------------------------------------
# Tỷ lệ validation tách ra để chấm Qini offline.
VALID_SIZE = 0.25

# Các cột có tỷ lệ khuyết quá cao (> 80%), cân nhắc loại bỏ hoặc xử lý đặc biệt ở Pipeline
HIGH_MISSING_COLS = [
    "ratio_monthly_amt_l6m_vs_n6m",
    "ratio_vol_hcvn_cash_appl_12m",
    "cnt_month_from_first_transaction",
    "cnt_month_from_last_transaction"
]

# Cột missing > ngưỡng này thì BẮT BUỘC tạo missing-flag (vì missing có thể là
# tín hiệu business: KH chưa từng giao dịch/loan).
MISSING_FLAG_THRESHOLD = 0.10  # 10%

# Ngưỡng uplift tối thiểu để khuyến nghị target (sẽ tinh chỉnh theo dữ liệu).
DEFAULT_UPLIFT_THRESHOLD = 0.0

# Các mốc percentile để báo cáo uplift@topK.
TOPK_PERCENTILES = [0.10, 0.20, 0.30]


def ensure_dirs() -> None:
    """Tạo lại toàn bộ thư mục output (gọi đầu mỗi notebook cho chắc)."""
    for d in (OUTPUT_DIR, CHART_DIR, TABLE_DIR, MODEL_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Chạy `python src/config.py` để in nhanh cấu hình, tiện kiểm tra path.
    ensure_dirs()
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("TRAIN_CSV exists:", TRAIN_CSV.exists())
    print("TEST_CSV  exists:", TEST_CSV.exists())
    print("SEED:", SEED)
    print("#categorical:", len(CATEGORICAL_FEATURES))