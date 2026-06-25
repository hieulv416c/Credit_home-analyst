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
    
    # 1. Tính toán các chỉ số kiểm toán khoảng cách thống kê cơ sở
    smd_df = standardized_mean_diff(df)
    cramer_df = cramers_v(df)
    
    # 2. Chạy Propensity Model để kiểm tra Selection Bias (Thiên vị phân bổ)
    model_res = treatment_assignment_auc(df, prefix=output_prefix)
    
    # 3. Trực quan hóa dữ liệu chẩn đoán hệ thống
    plot_smd(smd_df, top=20, name=f"covariate_smd_{output_prefix}")
    plot_treatment_importance(model_res["importance"], top=15, name=f"treatment_importance_{output_prefix}")
    
    # --- Đã loại bỏ hoàn toàn việc gọi hàm _generate_eda_markdown_report tại đây ---
    
    # 4. Trả về kết quả dạng dictionary để các công cụ hoặc hàm tổng khác có thể tái sử dụng
    return {
        "auc": model_res["auc"], 
        "smd": smd_df, 
        "cramer_v": cramer_df,
        "importance": model_res["importance"]
    }


# ============================================================================
# GIAI ĐOẠN 2: POST-TREATMENT EFFECTIVENESS (PHÂN TÍCH HIỆU QUẢ CHIẾN DỊCH)
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Đồng bộ hệ thống import chuẩn theo cấu trúc dự án của bạn
from . import config as C
from . import data_io as io

def _save(fig, name):
    """Hàm phụ trợ lưu đồ thị tự động vào thư mục đầu ra của hệ thống."""
    C.CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = C.CHART_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


# ============================================================================
# BƯỚC 5: OUTCOME / CONVERSION CHARTS (MA TRẬN CHUYỂN ĐỔI TOÀN CỤC)
# ============================================================================

def calculate_global_conversion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toán ma trận tỷ lệ chuyển đổi thô toàn cục (Global Signal)."""
    rows = []
    for treat in [C.TREAT_CASH, C.TREAT_CARD]:
        sub = df[df[C.TREATMENT_COL] == treat]
        if len(sub) == 0:
            continue
        rows.append({
            "treatment": treat,
            "cash_sign_rate_pct": round(100 * sub[C.CASH_OUTCOME].mean(), 2),
            "card_sign_rate_pct": round(100 * sub[C.CARD_OUTCOME].mean(), 2)
        })
    return pd.DataFrame(rows).set_index("treatment")


def plot_conversion_matrix(df: pd.DataFrame, name="conversion_by_treatment") -> Path:
    """Trực quan hóa tín hiệu thô toàn cục - 'Cùng một người, theo dõi 2 outcome'."""
    cm = calculate_global_conversion_matrix(df)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(2)
    w = 0.35
    
    # Phản ánh chính xác bảng số thật của bài mẫu lên biểu đồ
    ax.bar(x - w/2, [cm.loc[C.TREAT_CASH, "cash_sign_rate_pct"],
                     cm.loc[C.TREAT_CARD, "cash_sign_rate_pct"]],
           w, label="Tỷ lệ ký CASH", color="#2a7fff")
    ax.bar(x + w/2, [cm.loc[C.TREAT_CASH, "card_sign_rate_pct"],
                     cm.loc[C.TREAT_CARD, "card_sign_rate_pct"]],
           w, label="Tỷ lệ mở CARD", color="#ff8c42")
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"Được gọi {C.TREAT_CASH}", f"Được gọi {C.TREAT_CARD}"])
    ax.set_ylabel("Tỷ lệ chuyển đổi (%)")
    ax.set_title("Conversion theo Treatment (Toàn cục)")
    ax.legend()
    
    # Điền số liệu trực quan lên đầu các cột
    for i, treat in enumerate([C.TREAT_CASH, C.TREAT_CARD]):
        ax.text(i - w/2, cm.loc[treat, "cash_sign_rate_pct"],
                f'{cm.loc[treat, "cash_sign_rate_pct"]:.2f}%', ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, cm.loc[treat, "card_sign_rate_pct"],
                f'{cm.loc[treat, "card_sign_rate_pct"]:.2f}%', ha="center", va="bottom", fontsize=8)
                
    ax.margins(y=0.15)
    ax.grid(True, alpha=0.1, axis='y')
    return _save(fig, name)


# ============================================================================
# BƯỚC 6: NAIVE SEGMENT UPLIFT (TÍNH TOÁN HIỆU ỨNG TĂNG THÊM THÔ)
# ============================================================================

def segment_uplift(df: pd.DataFrame, by: str, product: str = "cash", min_n: int = 200) -> pd.DataFrame:
    """Tính toán Naive Uplift thuần túy theo phân khúc mở rộng dựa trên biến phân loại.
    
    product='cash': uplift = P(cash_sign|called_cash) - P(cash_sign|called_card)
    product='card': uplift = P(card_sign|called_card) - P(card_sign|called_cash)
    """
    df_clean = df.copy()
    df_clean[by] = df_clean[by].astype(str).str.strip().fillna('Unknown')
    df_clean = df_clean[~df_clean[by].isin(['Unknown', 'nan'])]
    
    if product == "cash":
        ycol, treat_val, ctrl_val = C.CASH_OUTCOME, C.TREAT_CASH, C.TREAT_CARD
    else:
        ycol, treat_val, ctrl_val = C.CARD_OUTCOME, C.TREAT_CARD, C.TREAT_CASH

    rows = []
    for seg, sub in df_clean.groupby(by, dropna=False):
        t = sub[sub[C.TREATMENT_COL] == treat_val]
        c = sub[sub[C.TREATMENT_COL] == ctrl_val]
        if len(t) == 0 or len(c) == 0:
            continue
        rt, rc = t[ycol].mean(), c[ycol].mean()
        rows.append({
            "segment": str(seg), 
            "n_total": len(sub),
            "n_treat": len(t), 
            "n_ctrl": len(c),
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
    """Trực quan hóa hiệu quả phân khúc thông qua lược đồ cột đơn đối xứng (Uplift)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    d = uplift_df.copy()
    if d.empty:
        ax.text(0.5, 0.5, "No Segment Data", ha="center", va="center")
        return _save(fig, f"uplift_{product}_{prefix}")
        
    x = np.arange(len(d))
    
    # 1. ĐỔI MÀU THÔNG MINH: Dương = Xanh (Hiệu quả), Âm = Đỏ (Tác dụng ngược)
    colors = ["#2a7fff" if val >= 0 else "#e63946" for val in d[f"naive_{product}_uplift_pct"]]
    
    # 2. VẼ CỘT ĐƠN TẠI VỊ TRÍ X (Không cần biến width tịnh tiến trái phải nữa)
    bars = ax.bar(x, d[f"naive_{product}_uplift_pct"], color=colors, alpha=0.85)
    
    # 3. THÊM ĐƯỜNG ranh giới 0 để nhìn rõ phân cực âm/dương
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    # 4. CẤU HÌNH TRỤC VÀ NHÃN HIỂN THỊ
    ax.set_xticks(x)
    ax.set_xticklabels(d["segment"], rotation=15)
    ax.set_ylabel("Mức độ tăng trưởng Uplift (điểm %)")
    ax.set_title(f"Hiệu quả chuyển đổi phân khúc sản phẩm: {product.upper()}")
    
    # 5. ĐIỀN SỐ TRỰC QUAN LÊN ĐẦU HOẶC DƯỚI CỘT
    for bar in bars:
        height = bar.get_height()
        # Nếu cột dương thì chữ nằm trên, cột âm thì chữ nằm dưới đồ thị
        va_dir = 'bottom' if height >= 0 else 'top'
        offset = 0.1 if height >= 0 else -0.3
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{height:+.2f}%', ha='center', va=va_dir, fontsize=8, fontweight='bold')
                
    ax.grid(True, alpha=0.1, axis='y')
    return _save(fig, f"uplift_{product}_{prefix}")


# ============================================================================
# PIPELINE ĐIỀU PHỐI CHÍNH (ĐỒNG BỘ ĐA NHÁNH MULTI-TREATMENT)
# ============================================================================

def run_post_treatment_eda(df: pd.DataFrame, by_column: str = "risk_grp", output_prefix: str = "train", uplift_threshold: float = 2.0) -> dict:
    """Pipeline tổng chạy Giai đoạn 2 bóc tách hiệu năng thương mại và lập bản đồ định vị chiến lược."""
    print(f"\n================ BẮT ĐẦU PHÂN TÍCH HIỆU QUẢ CHIẾN DỊCH [TẬP {output_prefix.upper()}] ================")
    df_post = df.copy()
    
    # ------------------------------------------------------------------------
    # LUỒNG KIỂM TOÁN VÀ DỰ PHÒNG CHO BY_COLUMN
    # ------------------------------------------------------------------------
    if by_column not in df_post.columns:
        print(f"⚠️ Cảnh báo: Không tìm thấy biến `{by_column}` trong dữ liệu.")
        
        # Nếu cột truyền vào bị lỗi, kiểm tra xem cột thực tế 'risk_grp' của bạn có sẵn không
        if 'risk_grp' in df_post.columns:
            by_column = 'risk_grp'
            print(f"🔄 Hệ thống tự động chuyển sang phân khúc mặc định: `{by_column}`.")
        else:
            # Phương án đường cùng nếu mất cả 'risk_grp'
            df_post['segment_dummy'] = 'All_Customers'
            by_column = 'segment_dummy'
            print("🚨 Không tìm thấy cột risk_grp! Hệ thống phân tích trên toàn bộ tệp (All_Customers).")
            
    # Bước 5: Trực quan hóa ma trận chuyển đổi tổng quan (Global Signal)
    plot_conversion_matrix(df_post, name=f"conversion_global_{output_prefix}")
    
    # Bước 6: Tính toán khối ma trận uplift phân khúc song song cho cả 2 sản phẩm
    uplift_cash = segment_uplift(df_post, by=by_column, product='cash')
    uplift_card = segment_uplift(df_post, by=by_column, product='card')
    
    # Vẽ biểu đồ cột mô tả tác động tăng thêm (Uplift)
    plot_segmented_uplift(uplift_cash, 'cash', output_prefix)
    plot_segmented_uplift(uplift_card, 'card', output_prefix)
    
    # Phân tích mật độ phân bổ tài chính (KDE) thông qua cấu hình REAL_FEATURES
    _plot_financial_kde(df_post, output_prefix)
    
    # Tạo lập bản đồ định vị phối hợp thông tin nhân quả của 2 sản phẩm
    targeting_report = []
    merged_uplift = pd.merge(uplift_cash, uplift_card, on="segment", suffixes=('_cash', '_card'))
    
    th = uplift_threshold # Đặt biến ngắn gọn để dễ viết logic dưới đây
    
    for _, row in merged_uplift.iterrows():
        up_cash = row['naive_cash_uplift_pct']
        up_card = row['naive_card_uplift_pct']
        
        # 2. SO SÁNH ĐỘNG VỚI BIẾN THRESHOLD THAY VÌ HARDCODE CON SỐ 2.0
        if up_cash > th and up_card <= th:
            label = " **Ưu tiên nhận CASH (Persuadables Cash)**"
        elif up_card > th and up_cash <= th:
            label = " **Ưu tiên nhận CARD (Persuadables Card)**"
        elif up_cash > th and up_card > th:
            label = " **Nhạy cảm cao với cả hai (Ưu tiên sản phẩm có biên lợi nhuận ròng cao hơn)**"
        else:
            label = " **Nhóm tự nguyện (Sure Things) hoặc 📭 Lost Causes (Không gọi đại trà)**"
            
        targeting_report.append(
            f"- Phân khúc `{row['segment']}`: Uplift Cash = `{up_cash:+.2f}%`, Uplift Card = `{up_card:+.2f}%` -> Nhãn: {label}"
        )
        
    # Ghi nhận kết quả phân tích tự động vào báo cáo Markdown của hệ thống thông qua io.
    _append_post_eda_report(by_column, uplift_cash, uplift_card, targeting_report, output_prefix)
    return {"cash_uplift": uplift_cash, "card_uplift": uplift_card}


# ============================================================================
# HÀM PHỤ TRỢ QUẢN LÝ FILE VÀ MẬT ĐỘ PHÂN BỔ TÀI CHÍNH
# ============================================================================

def _plot_financial_kde(df: pd.DataFrame, prefix: str):
    """Vẽ đồ thị mật độ phân bổ hành vi tài chính thực tế của nhóm chốt đơn thành công."""
    df_conv = df[df[C.CASH_OUTCOME] == 1]
    
    # Lấy feature liên tục lệch nhất từ danh sách REAL_FEATURES của module feature
    debt_col = C.REAL_FEATURES[0] if (hasattr(C, 'REAL_FEATURES') and C.REAL_FEATURES) else None
    
    if debt_col and debt_col in df_conv.columns and len(df_conv) > 1:
        valid = df_conv[np.isfinite(df_conv[debt_col])]
        if len(valid[valid[C.TREATMENT_COL] == C.TREAT_CASH]) > 1 and len(valid[valid[C.TREATMENT_COL] == C.TREAT_CARD]) > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.kdeplot(data=valid[valid[C.TREATMENT_COL] == C.TREAT_CASH], x=debt_col, label="Chốt đơn CASH", fill=True, color="darkorange", alpha=0.4, ax=ax)
            sns.kdeplot(data=valid[valid[C.TREATMENT_COL] == C.TREAT_CARD], x=debt_col, label="Chốt đơn CARD", fill=True, color="navy", alpha=0.4, ax=ax)
            ax.set_title(f"Hành vi tài chính tệp Chốt Đơn thành công (Feature: {debt_col})")
            ax.set_xlabel("Giá trị")
            ax.set_ylabel("Mật độ")
            ax.legend()
            _save(fig, f"financial_behavior_converted_{prefix}")


def _append_post_eda_report(by_col: str, u_cash: pd.DataFrame, u_card: pd.DataFrame, target_rep: list[str], prefix: str):
    """Sử dụng cơ chế quản lý tệp tin xuất báo cáo kiểm toán chất lượng chiến dịch."""
    report_path = C.OUTPUT_DIR / f"eda_report_{prefix}.md"
    
    if not report_path.exists():
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# BÁO CÁO KIỂM TOÁN CHẤT LƯỢNG NGẪU NHIÊN - TẬP {prefix.upper()}\n\n")

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n---\n# PHÂN TÍCH HIỆU QUẢ CHIẾN DỊCH VÀ ĐỊNH VỊ PHÂN KHÚC (POST-TREATMENT)\n\n")
        f.write(f"## 4. Bản đồ Định vị Phân khúc Chiến lược dựa trên `{by_col.upper()}` (Targeting Matrix Mapping)\n")
        f.write("- *Nhận định kinh doanh:* Tín hiệu thô này phản ánh trực diện hiệu ứng nhân quả sơ khởi trong tệp thử nghiệm.\n\n")
        for line in target_rep:
            f.write(f"{line}\n")
        f.write("\n*Báo cáo kết xuất tự động hoàn chỉnh hệ thống EDA sau Treatment.*\n")