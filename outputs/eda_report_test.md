# BÁO CÁO KIỂM TOÁN CHẤT LƯỢNG NGẪU NHIÊN - TẬP TEST

## 1. Kết luận Chỉ số Toàn cầu (Global Propensity Audit)
- **Propensity Model OOF AUC Score:** `0.5820`
- *Nhận định:* ⚠️ **CẢNH BÁO THIÊN VỊ ĐẦU VÀO**.

## 2. Top 5 Biến Quyết Định Phân Bổ Chiến Dịch
| 1 | `last_offer_limit` | 61.49% |
| 2 | `f_ever_default` | 10.61% |
| 3 | `last_process_channel` | 3.99% |
| 4 | `cnt_connected_tls` | 1.65% |
| 5 | `cnt_lender_has2pay_last_24m` | 1.62% |

## 3. Khoảng cách thống kê cơ sở
- Biến liên tục lệch nhất: `cnt_month_from_first_transaction` (SMD: `-0.0724`)
- Biến phân loại lệch nhất: `last_process_channel` (Cramér's V: `0.047`)
