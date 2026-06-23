from __future__ import annotations

import os
from pathlib import Path
import numpy as np
# ----------------------------------------------------------------------------
# 1) SEED — để kết quả tái lập (chạy lại ra y hệt)
# ----------------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------------
# PATHS — luôn tính từ gốc repo, không phụ thuộc chỗ chạy notebook
# ----------------------------------------------------------------------------
# config.py nằm trong src/, nên gốc repo là cha của src/
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

# Tạo sẵn thư mục output nếu chưa có (an toàn khi import nhiều lần)
for _d in (OUTPUT_DIR, CHART_DIR, TABLE_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. ĐỊNH DANH & BIẾN CAN THIỆP / BIẾN MỤC TIÊU
# ==========================================
ID_COL = "customer_id"
TREATMENT_COL = "treatment"  # Nhãn: 'Cash' hoặc 'Card'

# Bài toán Multi-Target (Cần xây dựng 2 pipeline uplift riêng biệt hoặc mô hình hóa đồng thời)
TARGET_COLS = {
    "cash": "cash_sign_30d",
    "card": "card_sign_30d"
}
# ==========================================
# 3. MA TRẬN TÍNH NĂNG X (Phân nhóm theo đặc tính hình học của dữ liệu)
# ==========================================

# A. Biến Phân Loại Định Danh (Categorical Features - Cần One-Hot Encoding)
CATEGORICAL_FEATURES = [
    "cli_job",                  # Vị trí công việc
    "cli_marital_status",       # Tình trạng hôn nhân
    "cli_contact_region_level", # Vùng miền từ địa chỉ liên hệ
    "cli_education",            # Trình độ giáo dục
    "last_process_channel",     # Kênh xử lý form yêu cầu gần nhất
    "cli_contact_region_population_grp" # Loại đô thị
]

# B. Biến Thứ Bậc / Đánh Giá Rủi Ro (Ordinal Features - Cần Label/Ordinal Encoding)
ORDINAL_FEATURES = [
    "risk_grp",                 # Nhóm rủi ro từ lần cuối được offer
    "worst_credit_status"       # Trạng thái tín dụng xấu nhất trong lịch sử CIC
]

# C. Biến Số Lượng, Số Tiền & Tỷ Lệ (Numerical Features - Cần Điền khuyết & Scaling)
NUMERICAL_FEATURES = [
    "min_sign_vol", "avg_sign_vol",      # Lịch sử số tiền vay
    "cnt_month_from_last_appl",          # Số tháng từ lần nộp form gần nhất
    "cli_age",                           # Tuổi khách hàng
    "cli_emp_period",                    # Thời gian tại vị trí công việc (tháng)
    "last_offer_limit",                  # Hạn mức gần nhất được offer
    "cnt_connected_tls",                 # Số cuộc gọi outbound kết nối được
    "cnt_fail_tls_3m",                   # Số cuộc gọi outbound thất bại 3 tháng qua
    "cnt_total_payment",                 # Số lần thanh toán trong lịch sử
    "ratio_monthly_amt_l6m_vs_n6m",      # Tỷ lệ số tiền phải trả (6 tháng qua vs 6 tháng tới)
    "cnt_month_status1",                 # Số tháng khách hàng có trạng thái xấu nhất = 1
    "ratio_vol_hcvn_cash_appl_12m",      # Tỷ lệ khoản tiền từ đơn vay mặt HC / tổng các TCTD
    "ratio_conn_tls_weekend_12m",        # Tỷ lệ cuộc gọi kết nối vào cuối tuần
    "ratio_conn_tls_endmonth_12m",       # Tỷ lệ cuộc gọi kết nối vào cuối tháng
    "cnt_bank_inst_appl_12m",            # Số ngân hàng yêu cầu vay trả góp
    "cnt_fi_inst_appl_12m",              # Số TCTC yêu cầu trả góp
    "cnt_report_bank_6m",                # Số ngân hàng có báo cáo trong 6 tháng
    "cnt_report_institution_12m",        # Số tổ chức có báo cáo trong 12 tháng
    "cnt_lender_has2pay_last_24m",       # Số tổ chức/ngân hàng phải trả nợ
    "max_cntr_len_fi_12m",               # Độ dài hợp đồng trả góp lớn nhất từ TCTC
    "amt_terminated_loan_12m",           # Tổng số tiền hợp đồng trả góp bị huỷ/kết thúc sớm
    "cnt_terminated_loan_12m",           # Số hợp đồng trả góp đã bị huỷ/kết thúc sớm
    "cnt_month_over3_credit_card_6m",    # Số tháng có hơn 3 thẻ tín dụng đồng thời
    "cnt_noinsurance_loan_24m",          # Số yêu cầu trả góp không bảo hiểm với HC
    "cnt_earlyterminated_cash",          # Số khoản vay tiền mặt với HC kết thúc sớm
    "cnt_earlyterminted_pos",            # Số khoản vay trả góp với HC kết thúc sớm
    "cnt_inbound_tls_12m",               # Số cuộc gọi inbound đến HC trong 12 tháng
    "cnt_month_from_first_transaction",  # Số tháng từ giao dịch chi tiêu đầu tiên với HC
    "cnt_month_from_last_transaction"    # Số tháng từ giao dịch chi tiêu gần nhất với HC
]

# D. Biến Cờ Hiệu / Nhị Phân (Flag/Binary Features - Chỉ cần điền khuyết bằng 0 nếu thiếu)
FLAG_FEATURES = [
    "f_approve_last_process",   # Flag nếu form yêu cầu gần nhất được phê duyệt
    "f_appointment_3m",         # Flag nếu khách hàng có đặt lịch hẹn trong 3 tháng
    "f_ever_default"            # Flag có từng nợ xấu với House Credit
]

# ==========================================
# 4. THAM SỐ VẬN HÀNH GIẢM THIỂU RỦI RO DỮ LIỆU KHUYẾT (MISSING RATIO)
# ==========================================
# Các cột có tỷ lệ khuyết quá cao (> 80%), cân nhắc loại bỏ hoặc xử lý đặc biệt ở Pipeline
HIGH_MISSING_COLS = [
    "ratio_monthly_amt_l6m_vs_n6m",
    "ratio_vol_hcvn_cash_appl_12m",
    "cnt_month_from_first_transaction",
    "cnt_month_from_last_transaction"
]
# Tỷ lệ khuyết dữ liệu để kích hoạt cơ chế gán nhãn Missing Flag độc lập
MISSING_FLAG_THRESHOLD = 0.3

# Danh sách các cột tính năng dạng dòng tiền dính phân bổ lệch phải mạnh cần tính log
AMOUNT_FEATURES = ["avg_sign_vol", "max_sign_vol", "total_debt"]  # Điều chỉnh tên cột theo đúng bộ data của bạn

# Danh sách các cột thuộc hàng rào bảo vệ, không dùng làm feature huấn luyện mô hình
NON_FEATURE_COLS = [ID_COL, TREATMENT_COL, "target_cash", "target_card", "y_response"]
RANDOM_SEED = 42
TEST_SIZE = 0.25