from __future__ import annotations
import pandas as pd
import numpy as np
from . import config as C

def check_data_health(df: pd.DataFrame, is_train: bool = True) -> dict[str, list[str]]:
    """
    Hệ thống tầm soát sức khỏe dữ liệu hợp nhất (Bác sĩ trưởng khoa).
    Quét từ cấu trúc hệ thống, nhân khẩu học cho đến logic tài chính chuyên sâu.
    """
    # Khởi tạo bệnh án tập trung
    medical_report = {
        "CRITICAL_ERRORS": [],     # Lỗi hệ thống, trùng lặp, thiếu cột hoặc sai nhãn
        "LOGIC_VIOLATIONS": [],    # Lỗi mâu thuẫn logic kinh doanh/toán học tài chính
        "DATA_QUALITY_NOTES": []   # Ghi chú về missing value cao hoặc điểm cần lưu ý
    }
    
    # -------------------------------------------------------------------------
    # 1. KIỂM TRA CẤU TRÚC & ĐỊNH DẠNG CỘT (Schema Validation)
    # -------------------------------------------------------------------------
    current_cols = set(df.columns)
    base_features = set(C.FLAG_FEATURES) 
    if hasattr(C, 'REAL_FEATURES'): base_features |= set(C.REAL_FEATURES)
    if hasattr(C, 'NUM_FEATURES'):  base_features |= set(C.NUM_FEATURES)
    if hasattr(C, 'CAT_FEATURES'):  base_features |= set(C.CAT_FEATURES)
    expected_features = base_features | {C.ID_COL, C.TREATMENT_COL}
    if is_train:
        expected_features |= set(C.TARGET_COLS.values())
        
    missing_cols = expected_features - current_cols
    if missing_cols:
        medical_report["CRITICAL_ERRORS"].append(f"Thiếu các cột bắt buộc trong file: {missing_cols}")
        return medical_report # Dừng lại ngay vì không đủ cột để kiểm tra các bước sau

    # -------------------------------------------------------------------------
    # 2. KIỂM TRA SƠ BỘ: TRÙNG LẶP ID & CỘT RỖNG 100%
    # -------------------------------------------------------------------------
    # Kiểm tra trùng lặp ID
    dup_count = df[C.ID_COL].duplicated().sum()
    if dup_count > 0:
        medical_report["CRITICAL_ERRORS"].append(f"Phát hiện {dup_count} dòng bị trùng lặp ID khách hàng ({C.ID_COL})!")

    # Kiểm tra cột rỗng hoàn toàn hoặc khuyết quá cao
    for col in df.columns:
        missing_pct = df[col].isnull().mean()
        if missing_pct == 1.0:
            medical_report["CRITICAL_ERRORS"].append(f"Cột [{col}] bị rỗng hoàn toàn 100%. Cần gỡ bỏ!")
        elif missing_pct > 0.8 and col not in ["ratio_monthly_amt_l6m_vs_n6m", "ratio_vol_hcvn_cash_appl_12m"]:
            # Né 2 cột đặc biệt ở Group 5 ra để tí xử lý riêng ở hàm logic
            medical_report["DATA_QUALITY_NOTES"].append(f"Cột [{col}] có tỷ lệ khuyết rất cao ({missing_pct*100:.2f}%).")

    # -------------------------------------------------------------------------
    # 3. KIỂM TRA NHÂN KHẨU HỌC & NHÃN CHIẾN DỊCH
    # -------------------------------------------------------------------------
    # Kiểm tra Tuổi (cli_age)
    if "cli_age" in df.columns:
        invalid_age = df[(df["cli_age"] < 18) | (df["cli_age"] > 100)]
        if len(invalid_age) > 0:
            medical_report["LOGIC_VIOLATIONS"].append(f"Nhân khẩu học: Có {len(invalid_age)} dòng có tuổi bất hợp lý (<18 hoặc >100).")

    # Kiểm tra tính hợp lệ của nhãn Treatment
    if C.TREATMENT_COL in df.columns:
        invalid_treatments = df[~df[C.TREATMENT_COL].isin(["Cash", "Card"])]
        if len(invalid_treatments) > 0:
            medical_report["CRITICAL_ERRORS"].append(f"Cột treatment chứa nhãn lạ ngoài 'Cash'/'Card': {invalid_treatments[C.TREATMENT_COL].unique()}")

    # -------------------------------------------------------------------------
    # 4. GỌI BỘ KIỂM TRA LOGIC TÀI CHÍNH CHUYÊN SÂU
    # -------------------------------------------------------------------------
    medical_report = _validate_financial_logic(df, medical_report)

    return medical_report


def _validate_financial_logic(df: pd.DataFrame, medical_report: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Hàm nội bộ (Private Helper): Quét 5 nhóm logic nghiệp vụ tài chính chuyên sâu.
    """
    # GROUP 1: Nhóm Logic về Thời gian và Lịch sử Giao dịch
    if "f_ever_default" in df.columns and "cnt_month_status1" in df.columns:
        cond_default = (df["f_ever_default"] == 1) & (df["cnt_month_status1"] == 0)
        if cond_default.sum() > 0:
            medical_report["LOGIC_VIOLATIONS"].append(
                f"Gãy Group 1.1: Có {cond_default.sum()} dòng ghi nhận 'Từng nợ xấu' (f_ever_default=1) nhưng số tháng trạng thái xấu nhất lại bằng 0!"
            )
            
    if all(c in df.columns for c in ["cnt_month_from_first_transaction", "cnt_total_payment"]):
        cond_timeline = df["cnt_month_from_first_transaction"].isna() & (df["cnt_total_payment"] > 0)
        if cond_timeline.sum() > 0:
            medical_report["LOGIC_VIOLATIONS"].append(
                f"Gãy Group 1.2: Có {cond_timeline.sum()} dòng chưa từng có giao dịch (Tháng rỗng) nhưng số lần thanh toán lại > 0!"
            )

    # GROUP 2: Nhóm Logic về Hành vi Cuộc gọi (Telesales)
    if all(c in df.columns for c in ["cnt_connected_tls", "cnt_fail_tls_3m"]):
        cond_call = (df["cnt_connected_tls"] == 0) & (df["cnt_fail_tls_3m"].fillna(0) == 0)
        if cond_call.sum() > 0:
            medical_report["LOGIC_VIOLATIONS"].append(
                f"Gãy Group 2: Có {cond_call.sum()} khách hàng có cả cuộc gọi thành công và thất bại đều bằng 0/NaN."
            )

    # GROUP 3: Nhóm Logic về Hạn mức và Số tiền (Volume & Limit)
    if "min_sign_vol" in df.columns and "avg_sign_vol" in df.columns:
        cond_vol = df["min_sign_vol"] > df["avg_sign_vol"]
        if cond_vol.sum() > 0:
            medical_report["LOGIC_VIOLATIONS"].append(
                f"Gãy Group 3.1: Có {cond_vol.sum()} dòng có số tiền vay nhỏ nhất (min_sign_vol) LỚN HƠN số tiền vay trung bình!"
            )
            
    money_cols = ["min_sign_vol", "avg_sign_vol", "last_offer_limit", "amt_terminated_loan_12m"]
    for col in money_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                medical_report["LOGIC_VIOLATIONS"].append(f"Gãy Group 3.1 (Số âm): Cột số tiền [{col}] chứa {neg_count} dòng giá trị âm.")
            
    if "f_approve_last_process" in df.columns and "last_offer_limit" in df.columns:
        cond_reject_limit = (df["f_approve_last_process"] == 0) & (df["last_offer_limit"] > 0)
        if cond_reject_limit.sum() > 0:
            medical_report["DATA_QUALITY_NOTES"].append(
                f"Lưu ý Group 3.2: Có {cond_reject_limit.sum()} dòng bị TỪ CHỐI đơn gần nhất nhưng vẫn có hạn mức offer > 0 (Có thể là Pre-approved)."
            )

    # GROUP 4: Nhóm Logic về CIC
    if all(c in df.columns for c in ["cnt_bank_inst_appl_12m", "cnt_fi_inst_appl_12m", "cnt_lender_has2pay_last_24m"]):
        total_appl_12m = df["cnt_bank_inst_appl_12m"].fillna(0) + df["cnt_fi_inst_appl_12m"].fillna(0)
        cond_cic = df["cnt_lender_has2pay_last_24m"] > total_appl_12m
        if cond_cic.sum() > 0:
            medical_report["DATA_QUALITY_NOTES"].append(
                f"Lưu ý Group 4: Có {cond_cic.sum()} khách hàng có số tổ chức đang trả nợ (24m) vượt quá số tổ chức nộp đơn (12m)."
            )

    # GROUP 5: Cảnh báo đặc biệt về Tỷ lệ Missing dữ liệu (Mẫu số bằng 0)
    for col in ["ratio_monthly_amt_l6m_vs_n6m", "ratio_vol_hcvn_cash_appl_12m"]:
        if col in df.columns:
            missing_pct = df[col].isnull().mean() * 100
            if missing_pct > 80:
                medical_report["DATA_QUALITY_NOTES"].append(
                    f"Cảnh báo Group 5: Cột [{col}] khuyết rất cao ({missing_pct:.2f}%). Do mẫu số bằng 0 (Sẽ xử lý ở bước sau)."
                )

    return medical_report


def print_health_report(report: dict[str, list[str]], title: str = "TRAIN") -> bool:
    """
    In báo cáo bệnh án sạch đẹp ra màn hình Notebook.
    """
    print(f"\n================ BÁO CÁO TẦM SOÁT SỨC KHỎE TẬP [{title.upper()}] ================")
    
    has_critical = len(report["CRITICAL_ERRORS"]) > 0
    has_violations = len(report["LOGIC_VIOLATIONS"]) > 0
    
    # 1. Lỗi nghiêm trọng
    if has_critical:
        print("\n LỖI NGHIÊM TRỌNG TRÊN HỆ THỐNG (Cần xử lý ngay):")
        for err in report["CRITICAL_ERRORS"]: print(f"  - {err}")
    else:
        print("\n Cấu trúc file & Hệ thống: OK")

    # 2. Vi phạm logic nghiệp vụ
    if has_violations:
        print("\n VI PHẠM LOGIC NGHIỆP VỤ & TOÁN HỌC TÀI CHÍNH:")
        for viol in report["LOGIC_VIOLATIONS"]: print(f"  - {viol}")
    else:
        print(" Logic nghiệp vụ & Toán học: OK")

    # 3. Ghi chú chất lượng
    if report["DATA_QUALITY_NOTES"]:
        print("\n GHI CHÚ CHẤT LƯỢNG DỮ LIỆU & PHÉP CHIA CHO 0:")
        for note in report["DATA_QUALITY_NOTES"]: print(f"  - {note}")
        
    print("\n=====================================================================\n")
    return not (has_critical or has_violations)