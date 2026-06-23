# BÁO CÁO KIỂM TOÁN CHẤT LƯỢNG NGẪU NHIÊN - TẬP TRAIN

## 1. Kết luận Chỉ số Toàn cầu (Global Propensity Audit)
- **Propensity Model OOF AUC Score:** `0.5842`
- *Nhận định:* ⚠️ **CẢNH BÁO THIÊN VỊ ĐẦU VÀO**.

## 2. Top 5 Biến Quyết Định Phân Bổ Chiến Dịch
| 1 | `last_offer_limit` | 75.27% |
| 2 | `f_ever_default` | 10.23% |
| 3 | `last_process_channel` | 1.52% |
| 4 | `cli_marital_status` | 1.27% |
| 5 | `cnt_month_from_last_appl` | 1.27% |

## 3. Khoảng cách thống kê cơ sở
- Biến liên tục lệch nhất: `cnt_month_from_last_transaction` (SMD: `0.0659`)
- Biến phân loại lệch nhất: `cli_marital_status` (Cramér's V: `0.026`)

---
# PHÂN TÍCH HIỆU QUẢ CHIẾN DỊCH VÀ ĐỊNH VỊ PHÂN KHÚC (POST-TREATMENT)

## 4. Bản đồ Định vị Phân khúc Chiến lược (Targeting Matrix Mapping)

*Báo cáo kết xuất tự động hoàn chỉnh hệ thống EDA.*
