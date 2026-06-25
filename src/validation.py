from __future__ import annotations

import pandas as pd

from . import config as C


Report = dict[str, list[str]]


def _new_report() -> Report:
    return {
        "CRITICAL_ERRORS": [],
        "LOGIC_VIOLATIONS": [],
        "DATA_QUALITY_NOTES": [],
    }


def _target_columns() -> list[str]:
    targets = getattr(C, "TARGET_COLS", {})
    if isinstance(targets, dict):
        return list(targets.values())
    return list(targets)


def _configured_feature_columns() -> set[str]:
    """Lay danh sach feature tu schema hien tai trong config.py."""
    feature_cols: set[str] = set()

    for attr in (
        "NUMERICAL_FEATURES",
        "CATEGORICAL_FEATURES",
        "ORDINAL_NUMERIC_FEATURES",
        "ORDINAL_STRING_FEATURES",
        "FLAG_FEATURES",
    ):
        feature_cols.update(getattr(C, attr, []))

    for cols in getattr(C, "FEATURE_GROUPS", {}).values():
        feature_cols.update(cols)

    non_features = set(getattr(C, "NON_FEATURE_COLS", []))
    return feature_cols - non_features


def _valid_treatments() -> set[str]:
    return {
        str(getattr(C, "TREAT_CASH", "cash")).strip().lower(),
        str(getattr(C, "TREAT_CARD", "card")).strip().lower(),
    }


def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def check_data_health(df: pd.DataFrame, is_train: bool = True) -> Report:
    """
    Kiem tra suc khoe du lieu: schema, treatment, target, missing va logic nghiep vu.

    Ham nay chi bao cao loi, khong mutate dataframe dau vao.
    """
    report = _new_report()

    current_cols = set(df.columns)

    # Cot bat buoc de pipeline chay duoc.
    required_cols = {C.ID_COL, C.TREATMENT_COL}
    if is_train:
        required_cols.update(_target_columns())

    missing_required = sorted(required_cols - current_cols)
    if missing_required:
        report["CRITICAL_ERRORS"].append(
            f"Thieu cac cot bat buoc trong file: {missing_required}"
        )
        return report

    # Feature thieu trong config la canh bao schema, khong nen dung toan bo validation.
    missing_config_features = sorted(_configured_feature_columns() - current_cols)
    if missing_config_features:
        report["DATA_QUALITY_NOTES"].append(
            f"Config co {len(missing_config_features)} feature khong co trong file: {missing_config_features}"
        )

    _validate_ids(df, report)
    _validate_missingness(df, report, is_train=is_train)
    _validate_treatment_labels(df, report)
    _validate_targets(df, report, is_train=is_train)
    _validate_basic_ranges(df, report)
    _validate_financial_logic(df, report)

    return report


def _validate_ids(df: pd.DataFrame, report: Report) -> None:
    missing_ids = int(df[C.ID_COL].isna().sum())
    if missing_ids > 0:
        report["CRITICAL_ERRORS"].append(
            f"Cot {C.ID_COL} co {missing_ids} dong bi thieu ID."
        )

    dup_count = int(df[C.ID_COL].duplicated().sum())
    if dup_count > 0:
        report["CRITICAL_ERRORS"].append(
            f"Phat hien {dup_count} dong bi trung lap ID khach hang ({C.ID_COL})."
        )


def _validate_missingness(df: pd.DataFrame, report: Report, is_train: bool) -> None:
    optional_test_targets = set() if is_train else set(_target_columns())
    high_missing_allowed = set(getattr(C, "HIGH_MISSING_COLS", []))

    for col in df.columns:
        missing_pct = float(df[col].isna().mean())

        if col in optional_test_targets:
            if missing_pct == 1.0:
                report["DATA_QUALITY_NOTES"].append(
                    f"Cot [{col}] trong tap test rong 100%; neu day la file scoring thi hop le."
                )
            continue

        if missing_pct == 1.0:
            report["CRITICAL_ERRORS"].append(
                f"Cot [{col}] bi rong hoan toan 100%. Can kiem tra lai nguon du lieu."
            )
        elif missing_pct > 0.8 and col not in high_missing_allowed:
            report["DATA_QUALITY_NOTES"].append(
                f"Cot [{col}] co ty le khuyet rat cao ({missing_pct * 100:.2f}%)."
            )


def _validate_treatment_labels(df: pd.DataFrame, report: Report) -> None:
    treatment_norm = df[C.TREATMENT_COL].astype("string").str.strip().str.lower()
    invalid_mask = treatment_norm.isna() | ~treatment_norm.isin(_valid_treatments())
    invalid_count = int(invalid_mask.fillna(False).sum())

    if invalid_count > 0:
        invalid_values = (
            df.loc[invalid_mask, C.TREATMENT_COL]
            .drop_duplicates()
            .head(10)
            .astype(str)
            .tolist()
        )
        report["CRITICAL_ERRORS"].append(
            f"Cot treatment co {invalid_count} dong nhan la/khuyet. Vi du: {invalid_values}"
        )


def _validate_targets(df: pd.DataFrame, report: Report, is_train: bool) -> None:
    if not is_train:
        return

    for col in _target_columns():
        y = _to_numeric(df, col)

        missing = int(y.isna().sum())
        if missing > 0:
            report["CRITICAL_ERRORS"].append(
                f"Cot target [{col}] co {missing} dong bi khuyet trong tap train."
            )

        invalid_mask = y.notna() & ~y.isin([0, 1])
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            report["CRITICAL_ERRORS"].append(
                f"Cot target [{col}] co {invalid_count} gia tri khac 0/1."
            )


def _validate_basic_ranges(df: pd.DataFrame, report: Report) -> None:
    if "cli_age" in df.columns:
        age = _to_numeric(df, "cli_age")
        invalid_age = age.notna() & ((age < 18) | (age > 100))
        n_invalid = int(invalid_age.sum())
        if n_invalid > 0:
            report["LOGIC_VIOLATIONS"].append(
                f"Nhan khau hoc: Co {n_invalid} dong co tuoi bat hop ly (<18 hoac >100)."
            )

    for col in getattr(C, "FLAG_FEATURES", []):
        if col not in df.columns:
            continue

        values = _to_numeric(df, col)
        invalid_mask = values.notna() & ~values.isin([0, 1])
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            report["LOGIC_VIOLATIONS"].append(
                f"Cot co [{col}] co {invalid_count} gia tri khac 0/1."
            )


def _validate_financial_logic(df: pd.DataFrame, report: Report) -> None:
    """Quet cac nhom logic nghiep vu tai chinh chinh."""
    if {"f_ever_default", "cnt_month_status1"}.issubset(df.columns):
        ever_default = _to_numeric(df, "f_ever_default")
        status1_months = _to_numeric(df, "cnt_month_status1")
        cond_default = (ever_default == 1) & (status1_months == 0)
        n = int(cond_default.sum())
        if n > 0:
            report["LOGIC_VIOLATIONS"].append(
                f"Group 1.1: Co {n} dong f_ever_default=1 nhung cnt_month_status1=0."
            )

    if {"cnt_month_from_first_transaction", "cnt_total_payment"}.issubset(df.columns):
        first_txn_months = _to_numeric(df, "cnt_month_from_first_transaction")
        total_payment = _to_numeric(df, "cnt_total_payment")
        cond_timeline = first_txn_months.isna() & (total_payment > 0)
        n = int(cond_timeline.sum())
        if n > 0:
            report["LOGIC_VIOLATIONS"].append(
                f"Group 1.2: Co {n} dong chua co thang giao dich dau tien nhung cnt_total_payment > 0."
            )

    if {"cnt_connected_tls", "cnt_fail_tls_3m"}.issubset(df.columns):
        connected = _to_numeric(df, "cnt_connected_tls")
        failed_3m = _to_numeric(df, "cnt_fail_tls_3m")
        cond_call = (connected == 0) & (failed_3m.fillna(0) == 0)
        n = int(cond_call.sum())
        if n > 0:
            report["LOGIC_VIOLATIONS"].append(
                f"Group 2: Co {n} khach hang co ca so cuoc goi thanh cong va that bai bang 0/NaN."
            )

    if {"min_sign_vol", "avg_sign_vol"}.issubset(df.columns):
        min_sign = _to_numeric(df, "min_sign_vol")
        avg_sign = _to_numeric(df, "avg_sign_vol")
        cond_vol = min_sign > avg_sign
        n = int(cond_vol.sum())
        if n > 0:
            report["LOGIC_VIOLATIONS"].append(
                f"Group 3.1: Co {n} dong min_sign_vol lon hon avg_sign_vol."
            )

    money_cols = [
        "min_sign_vol",
        "avg_sign_vol",
        "max_sign_vol",
        "last_offer_limit",
        "amt_terminated_loan_12m",
    ]
    for col in money_cols:
        if col not in df.columns:
            continue

        neg_count = int((_to_numeric(df, col) < 0).sum())
        if neg_count > 0:
            report["LOGIC_VIOLATIONS"].append(
                f"Group 3.1: Cot tien [{col}] co {neg_count} dong gia tri am."
            )

    if {"f_approve_last_process", "last_offer_limit"}.issubset(df.columns):
        approved = _to_numeric(df, "f_approve_last_process")
        offer_limit = _to_numeric(df, "last_offer_limit")
        cond_reject_limit = (approved == 0) & (offer_limit > 0)
        n = int(cond_reject_limit.sum())
        if n > 0:
            report["DATA_QUALITY_NOTES"].append(
                f"Group 3.2: Co {n} dong bi reject lan gan nhat nhung van co last_offer_limit > 0."
            )

    if {
        "cnt_bank_inst_appl_12m",
        "cnt_fi_inst_appl_12m",
        "cnt_lender_has2pay_last_24m",
    }.issubset(df.columns):
        bank_appl = _to_numeric(df, "cnt_bank_inst_appl_12m").fillna(0)
        fi_appl = _to_numeric(df, "cnt_fi_inst_appl_12m").fillna(0)
        lenders_24m = _to_numeric(df, "cnt_lender_has2pay_last_24m")
        total_appl_12m = bank_appl + fi_appl
        cond_cic = lenders_24m > total_appl_12m
        n = int(cond_cic.sum())
        if n > 0:
            report["DATA_QUALITY_NOTES"].append(
                f"Group 4: Co {n} khach hang co so to chuc dang tra no 24m vuot so to chuc nop don 12m."
            )

    for col in getattr(C, "HIGH_MISSING_COLS", []):
        if col not in df.columns:
            continue

        missing_pct = float(df[col].isna().mean()) * 100
        if missing_pct > 80:
            report["DATA_QUALITY_NOTES"].append(
                f"Group 5: Cot [{col}] khuyet rat cao ({missing_pct:.2f}%). Can xu ly bang missing flag/imputation."
            )


def print_health_report(report: Report, title: str = "TRAIN") -> bool:
    """In bao cao validation va tra ve True neu khong co critical/logic violation."""
    print(f"\n================ BAO CAO TAM SOAT SUC KHOE TAP [{title.upper()}] ================")

    has_critical = len(report["CRITICAL_ERRORS"]) > 0
    has_violations = len(report["LOGIC_VIOLATIONS"]) > 0

    if has_critical:
        print("\nLOI NGHIEM TRONG:")
        for err in report["CRITICAL_ERRORS"]:
            print(f"  - {err}")
    else:
        print("\nCau truc file va he thong: OK")

    if has_violations:
        print("\nVI PHAM LOGIC NGHIEP VU:")
        for violation in report["LOGIC_VIOLATIONS"]:
            print(f"  - {violation}")
    else:
        print("Logic nghiep vu va toan hoc: OK")

    if report["DATA_QUALITY_NOTES"]:
        print("\nGHI CHU CHAT LUONG DU LIEU:")
        for note in report["DATA_QUALITY_NOTES"]:
            print(f"  - {note}")

    print("\n=====================================================================\n")
    return not (has_critical or has_violations)
