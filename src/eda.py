"""
eda.py — Hàm EDA tái sử dụng: covariate balance (SMD), treatment-assignment
model, naive segment uplift, và các chart lưu ra outputs/charts/.

Mọi hàm vẽ đều tích hợp cấu hình headless để chạy an toàn trên cả Notebook lẫn CI/CD Script.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency
from catboost import CatBoostClassifier

from . import config as C
from . import data_io as io

# Triển khai bộ nhớ đệm Matplotlib chạy ẩn (Headless Backend) bảo vệ hệ thống CI/CD
os.environ.setdefault("MPLCONFIGDIR", str(C.OUTPUT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(C.OUTPUT_DIR / ".cache"))
(C.OUTPUT_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)
(C.OUTPUT_DIR / ".cache").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")  # Ép backend chạy ngầm không cần màn hình GUI
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 10})


def _save(fig, name: str) -> Path:
    """Hàm lõi quản lý lưu trữ và giải phóng bộ nhớ đồ họa tập trung."""
    C.CHART_DIR.mkdir(parents=True, exist_ok=True)
    out = C.CHART_DIR / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# GIAI ĐOẠN 1: PRE-TREATMENT BIAS AUDIT (KIỂM TOÁN THIÊN VỊ ĐẦU VÀO)
# ============================================================================

def standardized_mean_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toán khoảng cách SMD cho từng feature numeric giữa nhóm Cash & Card."""
    from . import config as C
    from . import data_io as io

    num_cols, _ = io.split_feature_types(df)
    ignore_set = {C.ID_COL, C.TREATMENT_COL}
    
    # --- ĐOẠN CẦN SỬA ĐỔI ĐỒNG BỘ CHỮ THƯỜNG ---
    # Thay vì dùng 'Cash' / 'Card' viết hoa, hãy dùng biến cấu hình hoặc .str.lower()
    g_cash = df[df[C.TREATMENT_COL].astype(str).str.lower() == "cash"]
    g_card = df[df[C.TREATMENT_COL].astype(str).str.lower() == "card"]
    
    rows = []
    for c in num_cols:
        if c in ignore_set:
            continue
            
        # Lấy mảng dữ liệu sạch loại bỏ NaN
        x_a = g_cash[c].dropna().values
        x_b = g_card[c].dropna().values
        
        # Phòng vệ nếu một trong hai nhóm không có dữ liệu để tính mean/std
        if len(x_a) == 0 or len(x_b) == 0:
            continue
            
        ma, mb = np.mean(x_a), np.mean(x_b)
        va, vb = np.var(x_a, ddof=1), np.var(x_b, ddof=1)
        
        # Tránh lỗi chia cho 0 nếu phương sai bằng 0
        denom = np.sqrt((va + vb) / 2.0)
        smd_val = (ma - mb) / denom if denom > 0 else 0.0
        
        rows.append({
            "feature": c, 
            "mean_cash": round(float(ma), 3), 
            "mean_card": round(float(mb), 3),
            "diff": round(float(ma - mb), 3), 
            "smd": round(float(smd_val), 4),
            "abs_smd": round(abs(float(smd_val)), 4),
        })
        
    # Phòng vệ tối cao: Nếu không có đặc trưng nào tính được, trả về DataFrame trống có sẵn cấu trúc cột
    if not rows:
        return pd.DataFrame(columns=["feature", "mean_cash", "mean_card", "diff", "smd", "abs_smd"])
        
    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)


def plot_smd(smd_df: pd.DataFrame, top=20, name="covariate_smd") -> Path:
    """Vẽ biểu đồ thanh ngang thể hiện mức độ lệch SMD."""
    if smd_df.empty:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "Empty SMD Data", ha="center", va="center")
        return _save(fig, name)
        
    d = smd_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(d))))
    colors = ["#d9534f" if v > 0.1 else "#5cb85c" for v in d["abs_smd"]]
    ax.barh(d["feature"], d["abs_smd"], color=colors)
    ax.axvline(0.1, ls="--", color="gray", lw=1, label="Ngưỡng 0.1 (Cân bằng)")
    ax.set_title(f"Top {top} Feature lệch nhất giữa Cash vs Card (|SMD|)")
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15, axis='x')
    return _save(fig, name)


def cramers_v(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toán chỉ số Cramér's V đo độ lệch cho các biến phân loại."""
    _, cat_cols = io.split_feature_types(df)
    cat_cols = [c for c in cat_cols if c not in [C.ID_COL, C.TREATMENT_COL]]
    rows = []
    
    for c in cat_cols:
        ct = pd.crosstab(df[c].fillna("__NA__"), df[C.TREATMENT_COL])
        if ct.shape[0] < 2 or ct.values.sum() == 0:
            continue
        chi2, p, _, _ = chi2_contingency(ct)
        n = ct.values.sum()
        r, k = ct.shape
        denom = min(r - 1, k - 1) or 1
        v_val = np.sqrt((chi2 / n) / denom)
        rows.append({
            "feature": c, "chi2": round(chi2, 2), "p_value": p,
            "cramers_v": round(float(v_val), 4)
        })
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def treatment_assignment_auc(df: pd.DataFrame, prefix: str = "train") -> dict:
    """
    Xây dựng mô hình Propensity Score Audit bằng Catboost sử dụng OOF Cross-Validation.
    
    Hàm này được bọc giáp chống lỗi Overflow int8 và ép kiểu đồng bộ 
    giữa các tập dữ liệu Train/Test.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    from . import config as C
    from . import features as F
    from . import data_io as io

    # 0) Thiết lập môi trường hiển thị biểu đồ an toàn
    os.environ.setdefault("MPLCONFIGDIR", str(C.OUTPUT_DIR / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(C.OUTPUT_DIR / ".cache"))
    (C.OUTPUT_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)
    (C.OUTPUT_DIR / ".cache").mkdir(parents=True, exist_ok=True)
    
    df_eda = df.copy()
    
    # 1) Áp dụng bộ mapping chữ thường đồng bộ tuyệt đối với data_io
    treatment_mapping = {"cash": 1, "card": 0}
    df_eda["treatment_target"] = df_eda[C.TREATMENT_COL].map(treatment_mapping).fillna(-1).astype(int)
    df_eda = df_eda[df_eda["treatment_target"] != -1]
    
    if len(df_eda) == 0:
        return {"auc": 0.5, "importance": pd.DataFrame(columns=["feature", "importance"])}

    # 2) Thực thi Pipeline Feature Engineering sinh biến mới
    df_eng, miss_cols = F.engineer_features(df_eda)
    
    # 3) TƯỜNG LỬA CHỐNG LEAKAGE & OVERFLOW: Trích xuất các cột FEATURE SẠCH
    clean_features = io.get_feature_columns(df_eng)
    X_raw = df_eng[clean_features].copy()
    
    # Loại bỏ treatment_target ra khỏi tập đặc trưng học máy nếu còn sót
    if "treatment_target" in X_raw.columns:
        X_raw = X_raw.drop(columns=["treatment_target"])
        
    y = df_eda["treatment_target"].values

    # 4) Đưa dữ liệu qua khẩu vị xử lý native của CatBoost (Chuyển chuỗi về dạng __NA__)
    X, cat_features = F.get_catboost_inputs(X_raw)
    
    # 5) ĐỒNG BỘ KIỂU DỮ LIỆU & BỌC GIÁP LỚP CUỐI (CHẶN ĐỨNG INT8 OVERFLOW ERROR)
    _, calculated_cats = io.split_feature_types(X)
    if hasattr(C, 'FLAG_FEATURES') and C.FLAG_FEATURES:
        cat_features = list(set(calculated_cats) | set([c for c in C.FLAG_FEATURES if c in X.columns]))
        
    for col in X.columns:
        if col in cat_features:
            # Ép kiểu chuỗi tường minh cho biến phân loại để CatBoost nhận diện chính xác
            X[col] = X[col].astype(str).fillna("__NA__")
        else:
            # GIẢI PHÁP GỐC RỄ: Ép các biến số về float32 để giải phóng giới hạn int8 trước khi fillna(-999)
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")
            X[col] = X[col].fillna(-999.0)
            
    # 6) THỰC THI HUẤN LUYỆN OOF CROSS-VALIDATION
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=C.SEED)
    oof_preds = np.zeros(len(y))
    
    for tr, va in skf.split(X, y):
        model_cv = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=4, random_seed=C.SEED, verbose=0)
        model_cv.fit(X.iloc[tr], y[tr], cat_features=cat_features)
        oof_preds[va] = model_cv.predict_proba(X.iloc[va])[:, 1]
        
    auc = roc_auc_score(y, oof_preds)
    
    # 7) HUẤN LUYỆN MÔ HÌNH TOÀN DIỆN ĐỂ LẤY IMPORTANCE & VẼ BIỂU ĐỒ
    full_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=4, random_seed=C.SEED, verbose=0)
    full_model.fit(X, y, cat_features=cat_features)
    
    # Vẽ biểu đồ phân bổ mật độ xác suất chỉ định chiến dịch
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(oof_preds[y == 1], label="Nhóm CASH", fill=True, color="darkorange", alpha=0.4, ax=ax)
    sns.kdeplot(oof_preds[y == 0], label="Nhóm CARD", fill=True, color="navy", alpha=0.4, ax=ax)
    ax.set_title(f"Mật độ Propensity Score ({prefix.upper()} OOF AUC: {auc:.4f})")
    ax.set_xlabel("Xác suất dự đoán xếp vào nhánh Cash")
    ax.set_ylabel("Mật độ")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    # Gọi hàm lưu biểu đồ nội bộ
    _save(fig, f"propensity_density_{prefix}")
    plt.close(fig)  # Đóng fig giải phóng bộ nhớ RAM
    
    # Trích xuất tầm quan trọng của tính năng
    imp_df = pd.DataFrame({
        'feature': X.columns.tolist(),
        'importance': full_model.get_feature_importance()
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    return {"auc": round(float(auc), 4), "importance": imp_df}


def plot_treatment_importance(imp: pd.DataFrame, top=15, name="treatment_assignment_importance") -> Path:
    """Vẽ Feature Importance của mô hình Propensity Score."""
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(d))))
    ax.barh(d["feature"], d["importance"], color="#6f42c1")
    ax.set_title(f"Top {top} Feature quyết định phân bổ Treatment")
    ax.set_xlabel("Feature Importance (CatBoost)")
    return _save(fig, name)


def run_pre_treatment_eda(df: pd.DataFrame, output_prefix: str = "train") -> dict:
    """Pipeline tổng chạy Giai đoạn 1 kiểm toán chất lượng ngẫu nhiên đầu vào."""
    print(f"\n================ BẮT ĐẦU KIỂM TRA THIÊN VỊ [TẬP {output_prefix.upper()}] ================")
    smd_df = standardized_mean_diff(df)
    cramer_df = cramers_v(df)
    model_res = treatment_assignment_auc(df, prefix=output_prefix)
    
    plot_smd(smd_df, top=20, name=f"covariate_smd_{output_prefix}")
    plot_treatment_importance(model_res["importance"], top=15, name=f"treatment_importance_{output_prefix}")
    
    # Kết xuất báo cáo chẩn đoán hệ thống (.md)
    _generate_eda_markdown_report(model_res["auc"], model_res["importance"], smd_df, cramer_df, output_prefix)
    return {"auc": model_res["auc"], "smd": smd_df, "cramer_v": cramer_df}


# ============================================================================
# GIAI ĐOẠN 2: POST-TREATMENT EFFECTIVENESS (PHÂN TÍCH HIỆU QUẢ CHIẾN DỊCH)
# ============================================================================

def segment_uplift(df: pd.DataFrame, by: str, product: str = "cash", min_n: int = 200) -> pd.DataFrame:
    """Tính toán Naive Uplift thuần túy theo phân khúc cho cấu trúc Multi-treatment."""
    df_clean = df.copy()
    df_clean[by] = df_clean[by].astype(str).str.strip().fillna('Unknown')
    df_clean = df_clean[~df_clean[by].isin(['Unknown', 'nan'])]
    
    if product == "cash":
        ycol = C.TARGET_COLS.get('cash', 'target_cash') if isinstance(C.TARGET_COLS, dict) else C.TARGET_COLS[0]
        treat_val, ctrl_val = 'Cash', 'Card'
    else:
        ycol = C.TARGET_COLS.get('card', 'target_card') if isinstance(C.TARGET_COLS, dict) else C.TARGET_COLS[0]
        treat_val, ctrl_val = 'Card', 'Cash'

    rows = []
    for seg, sub in df_clean.groupby(by, observed=True):
        t = sub[sub[C.TREATMENT_COL] == treat_val]
        c = sub[sub[C.TREATMENT_COL] == ctrl_val]
        if len(t) == 0 or len(c) == 0:
            continue
        rt, rc = t[ycol].mean(), c[ycol].mean()
        rows.append({
            "segment": str(seg), "n_total": len(sub),
            "n_treat": len(t), "n_ctrl": len(c),
            f"{product}_resp_treat_pct": round(100 * rt, 3),
            f"{product}_resp_ctrl_pct": round(100 * rc, 3),
            f"naive_{product}_uplift_pct": round(100 * (rt - rc), 3),
            "reliable": (len(t) >= min_n and len(c) >= min_n),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(f"naive_{product}_uplift_pct", ascending=False).reset_index(drop=True)
    return out


def plot_segmented_uplift(uplift_df: pd.DataFrame, product: str, prefix: str) -> Path:
    """Trực quan hóa hiệu quả phân khúc thông qua lược đồ cột đôi đối xứng."""
    fig, ax = plt.subplots(figsize=(8, 4))
    d = uplift_df.copy()
    if d.empty:
        ax.text(0.5, 0.5, "No Segment Data", ha="center", va="center")
        return _save(fig, f"cr_{product}_{prefix}")
        
    x = np.arange(len(d))
    width = 0.35
    
    ax.bar(x - width/2, d[f"{product}_resp_treat_pct"], width, label=f"Nhánh chính mời {product.upper()}", color="darkorange")
    ax.bar(x + width/2, d[f"{product}_resp_ctrl_pct"], width, label=f"Nhánh đối chứng", color="navy")
    
    ax.set_xticks(x)
    ax.set_xticklabels(d["segment"], rotation=15)
    ax.set_ylabel("Tỷ lệ chuyển đổi (%)")
    ax.set_title(f"Hiệu quả chuyển đổi phân khúc sản phẩm: {product.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.1, axis='y')
    return _save(fig, f"cr_{product}_{prefix}")


def run_post_treatment_eda(df: pd.DataFrame, output_prefix: str = "train") -> dict:
    """Pipeline tổng chạy Giai đoạn 2 bóc tách hiệu năng thương mại và lập bản đồ định vị chiến lược."""
    print(f"\n================ BẮT ĐẦU PHÂN TÍCH HIỆU QUẢ CHIẾN DỊCH [TẬP {output_prefix.upper()}] ================")
    df_post = df.copy()
    
    # Đồng bộ hóa định vị nhãn phân khúc động an toàn
    if 'cli_age' in df_post.columns:
        df_post['age_tier'] = pd.cut(df_post['cli_age'], bins=[0, 30, 45, 60, 100], labels=['Under 30', '30-45', '45-60', 'Above 60'])
    else:
        df_post['age_tier'] = 'Unknown'
        
    # Tính toán khối ma trận uplift độc lập chuẩn chỉnh
    uplift_cash = segment_uplift(df_post, by='age_tier', product='cash')
    uplift_card = segment_uplift(df_post, by='age_tier', product='card')
    
    plot_segmented_uplift(uplift_cash, 'cash', output_prefix)
    plot_segmented_uplift(uplift_card, 'card', output_prefix)
    
    # Đào sâu phân tích phân bổ mật độ nhóm chuyển đổi thành công (KDE)
    _plot_financial_kde(df_post, output_prefix)
    
    # Tạo lập bản đồ định vị 4 nhóm khách hàng
    targeting_report = []
    for _, row in uplift_cash.iterrows():
        up_val = row['naive_cash_uplift_pct']
        if up_val > 2.0:
            label = "🎯 **Ưu tiên nhận CASH (Persuadables)**"
        elif up_val < -2.0:
            label = "💳 **Ưu tiên nhận CARD (Persuadables)**"
        else:
            label = "🤝 **Nhóm tự nguyện (Sure Things) hoặc 📭 Lost Causes**"
        targeting_report.append(f"- Phân khúc `{row['segment']}`: Uplift thô Cash = `{up_val:+.2f}%` -> Nhãn: {label}")
        
    # Nối dữ liệu vào tệp báo cáo tổng
    _append_post_eda_report(uplift_cash, uplift_card, targeting_report, output_prefix)
    return {"cash_uplift": uplift_cash, "card_uplift": uplift_card}


# ============================================================================
# HÀM PHỤ TRỢ XUẤT FILE BÁO CÁO VĂN BẢN (.MD)
# ============================================================================

def _plot_financial_kde(df: pd.DataFrame, prefix: str):
    """Vẽ đồ thị KDE bóc tách hành vi tài chính thực tế của khách hàng chốt đơn."""
    target_cash = C.TARGET_COLS.get('cash', 'target_cash') if isinstance(C.TARGET_COLS, dict) else C.TARGET_COLS[0]
    df_conv = df[df[target_cash] == 1]
    debt_col = C.REAL_FEATURES[0] if (hasattr(C, 'REAL_FEATURES') and C.REAL_FEATURES) else None
    
    if debt_col and debt_col in df_conv.columns and len(df_conv) > 1:
        valid = df_conv[np.isfinite(df_conv[debt_col])]
        if len(valid[valid[C.TREATMENT_COL] == 'Cash']) > 1 and len(valid[valid[C.TREATMENT_COL] == 'Card']) > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.kdeplot(data=valid[valid[C.TREATMENT_COL] == 'Cash'], x=debt_col, label="Chốt đơn CASH", fill=True, color="darkorange", alpha=0.4, ax=ax)
            sns.kdeplot(data=valid[valid[C.TREATMENT_COL] == 'Card'], x=debt_col, label="Chốt đơn CARD", fill=True, color="navy", alpha=0.4, ax=ax)
            ax.set_title(f"Hành vi tài chính tệp Chốt Đơn thành công (Feature: {debt_col})")
            ax.set_xlabel("Giá trị")
            ax.set_ylabel("Mật độ")
            ax.legend()
            _save(fig, f"financial_behavior_converted_{prefix}")


def _generate_eda_markdown_report(auc: float, imp: pd.DataFrame, smd: pd.DataFrame, cramer: pd.DataFrame, prefix: str):
    report_path = C.OUTPUT_DIR / f"eda_report_{prefix}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# BÁO CÁO KIỂM TOÁN CHẤT LƯỢNG NGẪU NHIÊN - TẬP {prefix.upper()}\n\n")
        f.write(f"## 1. Kết luận Chỉ số Toàn cầu (Global Propensity Audit)\n")
        f.write(f"- **Propensity Model OOF AUC Score:** `{auc:.4f}`\n")
        f.write(f"- *Nhận định:* " + ("✅ **Dữ liệu NGẪU NHIÊN HOÀN HẢO**. Không dính Selection Bias.\n\n" if auc < 0.55 else "⚠️ **CẢNH BÁO THIÊN VỊ ĐẦU VÀO**.\n\n"))
        f.write(f"## 2. Top 5 Biến Quyết Định Phân Bổ Chiến Dịch\n")
        for i, row in imp.head(5).iterrows():
            f.write(f"| {i+1} | `{row['feature']}` | {row['importance']:.2f}% |\n")
        f.write("\n## 3. Khoảng cách thống kê cơ sở\n")
        f.write(f"- Biến liên tục lệch nhất: `{smd.iloc[0]['feature'] if not smd.empty else 'None'}` (SMD: `{smd.iloc[0]['smd'] if not smd.empty else 0}`)\n")
        f.write(f"- Biến phân loại lệch nhất: `{cramer.iloc[0]['feature'] if not cramer.empty else 'None'}` (Cramér's V: `{cramer.iloc[0]['cramers_v'] if not cramer.empty else 0}`)\n")


def _append_post_eda_report(u_cash: pd.DataFrame, u_card: pd.DataFrame, target_rep: list[str], prefix: str):
    report_path = C.OUTPUT_DIR / f"eda_report_{prefix}.md"
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n---\n# PHÂN TÍCH HIỆU QUẢ CHIẾN DỊCH VÀ ĐỊNH VỊ PHÂN KHÚC (POST-TREATMENT)\n\n")
        f.write("## 4. Bản đồ Định vị Phân khúc Chiến lược (Targeting Matrix Mapping)\n")
        for line in target_rep:
            f.write(f"{line}\n")
        f.write("\n*Báo cáo kết xuất tự động hoàn chỉnh hệ thống EDA.*\n")