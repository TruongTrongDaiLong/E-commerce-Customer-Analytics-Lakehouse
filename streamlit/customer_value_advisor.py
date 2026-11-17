"""
Customer Value Advisor
======================
Ứng dụng demo 2 mô hình dự đoán hành vi khách hàng
- Mô hình 1: Customer_Spending_Prediction_XGB  → avg_spend_per_year
- Mô hình 2: XGBoost_Tweedie_Final             → avg_orders_per_year
"""

import streamlit as st
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_SPEND  = "models:/XGB_Model_V1/Production"
MODEL_ORDERS = "models:/XGBoost_Tweedie_Final/Production"

SPEND_THRESHOLD  = 2500
ORDERS_THRESHOLD = 55

SEGMENT_CONFIG = {
    "VIP":            {"color": "#4F46E5", "bg": "#EEF2FF", "label": "VIP"},
    "Tiềm năng":      {"color": "#0891B2", "bg": "#E0F7FA", "label": "Tiềm năng"},
    "Trung bình":     {"color": "#059669", "bg": "#ECFDF5", "label": "Trung bình"},
    "Rủi ro rời bỏ":  {"color": "#DC2626", "bg": "#FEF2F2", "label": "Rủi ro rời bỏ"},
}

CRM_ACTIONS = {
    "VIP": [
        "Tặng voucher sinh nhật cá nhân hoá",
        "Ưu tiên truy cập sản phẩm mới trước khi ra mắt đại trà",
        "Chăm sóc 1-1 qua account manager riêng",
        "Mời tham gia chương trình khách hàng thân thiết cao cấp",
    ],
    "Tiềm năng": [
        "Gửi email upsell sản phẩm liên quan đến đơn gần nhất",
        "Voucher giảm giá 10–15% để kích hoạt mua thêm",
        "Thông báo khi sản phẩm đã xem có khuyến mãi",
        "Gợi ý chương trình mua định kỳ",
    ],
    "Trung bình": [
        "Đưa vào luồng email newsletter định kỳ hàng tuần",
        "Gợi ý sản phẩm từ danh mục chưa mua",
        "Retargeting quảng cáo theo hành vi duyệt web",
        "Mời viết đánh giá sản phẩm để đổi điểm thưởng",
    ],
    "Rủi ro rời bỏ": [
        "Chiến dịch win-back với ưu đãi đặc biệt 20–30%",
        "Gửi khảo sát ngắn tìm hiểu lý do không quay lại",
        "Flash sale độc quyền gửi riêng cho nhóm này",
        "Liên hệ trực tiếp nếu từng là khách hàng lớn",
    ],
}

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang load mô hình từ MLflow...")
def load_models():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_spend  = mlflow.pyfunc.load_model(MODEL_SPEND)
    model_orders = mlflow.pyfunc.load_model(MODEL_ORDERS)
    return model_spend, model_orders

# ─────────────────────────────────────────────
# HÀM TIỆN ÍCH
# ─────────────────────────────────────────────
def classify_segment(spend, orders):
    if spend >= SPEND_THRESHOLD and orders >= ORDERS_THRESHOLD:
        return "VIP"
    elif spend >= SPEND_THRESHOLD and orders < ORDERS_THRESHOLD:
        return "Tiềm năng"
    elif spend < SPEND_THRESHOLD and orders >= ORDERS_THRESHOLD:
        return "Trung bình"
    else:
        return "Rủi ro rời bỏ"

def predict_customer(model_spend, model_orders, features: dict):
    df = pd.DataFrame([features])

    feat_spend = ["total_spend", "avg_order_value", "std_order_value",
                  "avg_items_per_order", "total_orders", "years_active"]
    pred_spend = float(model_spend.predict(df[feat_spend])[0])

    df["total_spend_log"]     = np.log1p(df["total_spend"])
    df["avg_order_value_log"] = np.log1p(df["avg_order_value"])
    df["recency_log"]         = np.log1p(df["recency"])
    feat_orders = ["total_spend_log", "category_diversity", "avg_order_value_log",
                   "recency_log", "years_active", "age_group"]
    pred_orders = float(model_orders.predict(df[feat_orders])[0])

    return max(pred_spend, 0), max(pred_orders, 0)

def clv_estimate(pred_spend, horizon_years=3):
    return pred_spend * horizon_years

# ─────────────────────────────────────────────
# PAGE CONFIG & CSS MINIMALIST
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tư vấn giá trị khách hàng",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Font Inter cho cảm giác SaaS hiện đại */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar tinh gọn */
[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #0F172A !important;
    font-weight: 800;
}
[data-testid="stSidebar"] label {
    color: #475569 !important;
    font-weight: 500;
}
[data-testid="stSidebar"] hr { border-color: #E2E8F0; margin: 1.5rem 0; }

/* Header Banner */
.header-banner {
    background: #FFFFFF;
    padding: 24px 32px;
    margin-bottom: 24px;
    border-radius: 16px;
    border: 1px solid #E2E8F0;
    border-left: 6px solid #4F46E5;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
}
.header-banner h2 {
    background: linear-gradient(90deg, #1E3A8A 0%, #4F46E5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 36px;
    letter-spacing: -0.5px;
    margin: 0 0 8px 0;
}
.header-banner p {
    color: #64748B !important;
    margin: 0;
    font-size: 15px;
    font-weight: 400;
}

/* Nút bấm */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    color: white;
}

#MainMenu, footer { visibility: hidden; }

/* Pill Status */
.status-pill {
    display: inline-flex;
    align-items: center;
    padding: 8px 16px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-top: 32px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.status-ok  { background-color: #ECFDF5; color: #059669; border: 1px solid #A7F3D0; }
.status-err { background-color: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }

/* Thẻ thông tin */
.result-card { border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; }
.result-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin-bottom: 4px; font-weight: 600;}
.result-value { font-size: 28px; font-weight: 800; line-height: 1.2; letter-spacing: -0.5px;}
.segment-tag { display: inline-block; padding: 6px 16px; border-radius: 999px; font-size: 14px; font-weight: 700; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}

/* Card hành động CRM */
.action-item {
    padding: 14px 18px;
    border-left: 4px solid;
    border-radius: 0 8px 8px 0;
    background: #FFFFFF;
    margin-bottom: 12px;
    font-size: 14px;
    font-weight: 500;
    color: #334155;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    border-top: 1px solid #F1F5F9;
    border-right: 1px solid #F1F5F9;
    border-bottom: 1px solid #F1F5F9;
    transition: transform 0.2s ease;
}
.action-item:hover {
    transform: translateX(4px);
}

/* Header nhóm trường */
.field-group {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #4F46E5;
    margin-top: 16px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #EEF2FF;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Phân tích Giá trị")
    st.markdown("<p style='color:#64748B; font-size:14px;'>Hệ thống dự đoán hành vi</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div style='font-weight:700; color:#0F172A; margin-bottom:12px;'>Cấu hình phân khúc</div>", unsafe_allow_html=True)
    spend_thr  = st.number_input("Chi tiêu tối thiểu nhóm VIP ($/năm)",
                                  value=SPEND_THRESHOLD,  step=100, min_value=0)
    orders_thr = st.number_input("Số đơn tối thiểu nhóm VIP (đơn/năm)",
                                  value=ORDERS_THRESHOLD, step=1,   min_value=1)
    SPEND_THRESHOLD  = spend_thr
    ORDERS_THRESHOLD = orders_thr

    st.markdown("---")
    st.markdown(
        "<div style='font-size:13px;color:#64748B;line-height:1.6; background: #FFFFFF; padding: 12px; border-radius: 8px; border: 1px solid #E2E8F0;'>"
        "<b style='color:#0F172A;'>Nhóm 24</b><br>Khóa luận tốt nghiệp<br>Kỹ thuật dữ liệu"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
try:
    model_spend, model_orders = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    _load_error = str(e)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown(
        "<div class='header-banner'>"
        "<h2>Tư vấn Giá trị Khách hàng</h2>"
        "<p>Phân tích chỉ số, dự đoán xu hướng chi tiêu và đề xuất kịch bản chăm sóc (CRM).</p>"
        "</div>",
        unsafe_allow_html=True,
    )
with col_status:
    if model_loaded:
        st.markdown(
            "<div style='text-align:right;'>"
            "<div class='status-pill status-ok'>Đã kết nối MLflow</div>"
            "</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='text-align:right;'>"
            "<div class='status-pill status-err'>Lỗi MLflow</div>"
            "</div>",
            unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2 TAB
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["Dự đoán khách hàng", "Mô phỏng kịch bản"])

# ══════════════════════════════════════════
# TAB 1 — DỰ ĐOÁN
# ══════════════════════════════════════════
with tab1:
    form_box = st.container(border=True)
    with form_box:
        st.markdown("<div class='field-group' style='margin-top:0;'>Thông tin giao dịch</div>", unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)
        total_spend         = f1.number_input("Tổng chi tiêu (USD)",          min_value=0.0,  value=3200.0, step=100.0,
                                               help="Tổng số tiền đã chi tiêu trong toàn bộ lịch sử")
        total_orders        = f2.number_input("Tổng số đơn hàng",             min_value=1,    value=24,     step=1)
        avg_order_value     = f3.number_input("Trung bình mỗi đơn (USD)", min_value=0.0, value=133.3, step=10.0,
                                               help="AOV = Tổng chi tiêu / Tổng số đơn")
        std_order_value     = f4.number_input("Độ biến động giá trị đơn",     min_value=0.0,  value=45.0,   step=5.0,
                                               help="Độ lệch chuẩn giá trị các đơn hàng")
        f5, f6, f7, f8 = st.columns(4)
        avg_items_per_order = f5.number_input("Sản phẩm trung bình/đơn", min_value=1.0, value=3.5,  step=0.5)
        category_diversity  = f6.number_input("Số danh mục đã mua",  min_value=1,    value=5,      step=1,
                                               help="Mức độ đa dạng danh mục sản phẩm")
        years_active         = f7.number_input("Số năm là khách hàng",          min_value=1, value=3,  step=1)
        recency               = f8.number_input("Số ngày từ lần mua cuối", min_value=0, value=45, step=1,
                                               help="Recency — càng thấp càng active")

        st.markdown("<div class='field-group'>Thông tin nhân khẩu học</div>", unsafe_allow_html=True)
        fg1, _, _, _ = st.columns(4)
        age_group = fg1.selectbox("Nhóm tuổi", ["18-24", "25-34", "35-44", "45-54", "55+"])

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("Phân tích dữ liệu", type="primary", use_container_width=False)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    if not predict_btn:
        st.info("Điền thông tin phía trên và bấm **Phân tích dữ liệu** để xem kết quả dự đoán.")
    else:
        if not model_loaded:
            st.error("Chưa kết nối được MLflow.")
        else:
            features = {
                "total_spend":          total_spend,
                "total_orders":         total_orders,
                "avg_order_value":      avg_order_value,
                "std_order_value":      std_order_value,
                "avg_items_per_order":  avg_items_per_order,
                "category_diversity":   category_diversity,
                "years_active":         years_active,
                "recency":              recency,
                "age_group":            age_group,
            }
            with st.spinner("Hệ thống đang xử lý mô hình..."):
                pred_spend, pred_orders = predict_customer(model_spend, model_orders, features)

            segment = classify_segment(pred_spend, pred_orders)
            clv     = clv_estimate(pred_spend)
            cfg     = SEGMENT_CONFIG[segment]

            result_box = st.container(border=True)
            with result_box:
                head_l, head_r = st.columns([1, 3])
                with head_l:
                    st.markdown(
                        f"<span class='segment-tag' style='background:{cfg['bg']};color:{cfg['color']}; border: 1px solid {cfg['color']}40;'>"
                        f"{segment}</span>",
                        unsafe_allow_html=True,
                    )
                with head_r:
                    r1, r2, r3 = st.columns(3)
                    r1.markdown(
                        f"<div class='result-label'>Dự phóng chi tiêu / năm</div>"
                        f"<div class='result-value' style='color:{cfg['color']};'>${pred_spend:,.0f}</div>",
                        unsafe_allow_html=True,
                    )
                    r2.markdown(
                        f"<div class='result-label'>Dự phóng số đơn / năm</div>"
                        f"<div class='result-value' style='color:{cfg['color']};'>{pred_orders:.1f}</div>",
                        unsafe_allow_html=True,
                    )
                    r3.markdown(
                        f"<div class='result-label'>Giá trị trọn đời (CLV 3 năm)</div>"
                        f"<div class='result-value' style='color:{cfg['color']};'>${clv:,.0f}</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

                col_chart, col_crm = st.columns([3, 2], gap="large")

                with col_chart:
                    st.markdown(
                        "<div style='font-size:15px;font-weight:700;color:#0F172A;margin-bottom:8px;'>"
                        "Vị thế phân khúc khách hàng</div>",
                        unsafe_allow_html=True,
                    )
                    
                    # Cấu hình kích thước biểu đồ mở rộng linh hoạt
                    max_orders_plot = max(ORDERS_THRESHOLD * 2.2, pred_orders * 1.3, 25)
                    max_spend_plot  = max(SPEND_THRESHOLD * 2.2, pred_spend * 1.3, 6000)

                    fig = go.Figure()

                    # Hàm vẽ vùng phân khúc và Nhãn Badge ở góc
                    def add_segment_region(x0, x1, y0, y1, seg_name):
                        c = SEGMENT_CONFIG[seg_name]
                        # Tô màu nền
                        fig.add_shape(
                            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                            fillcolor=c["bg"], line_width=0, layer="below"
                        )
                        # Tính tọa độ góc trên bên trái của vùng để đặt nhãn
                        label_x = x0 + (max_orders_plot * 0.02)
                        label_y = y1 - (max_spend_plot * 0.03)
                        
                        # Vẽ thẻ tên phân khúc dạng thẻ tinh gọn
                        fig.add_annotation(
                            x=label_x, y=label_y,
                            text=seg_name.upper(), showarrow=False,
                            xanchor="left", yanchor="top",
                            font=dict(size=11, color=c["color"], weight="bold"),
                            bgcolor="white", bordercolor=c["color"], borderwidth=1, borderpad=4
                        )

                    # Áp dụng hàm vẽ 4 vùng
                    add_segment_region(0, ORDERS_THRESHOLD, 0, SPEND_THRESHOLD, "Rủi ro rời bỏ")
                    add_segment_region(0, ORDERS_THRESHOLD, SPEND_THRESHOLD, max_spend_plot, "Tiềm năng")
                    add_segment_region(ORDERS_THRESHOLD, max_orders_plot, 0, SPEND_THRESHOLD, "Trung bình")
                    add_segment_region(ORDERS_THRESHOLD, max_orders_plot, SPEND_THRESHOLD, max_spend_plot, "VIP")

                    # Kẻ đường đứt nét ngưỡng (chỉ vẽ đường, bỏ text mặc định đi kèm)
                    fig.add_hline(y=SPEND_THRESHOLD, line_dash="dash", line_color="#94A3B8", opacity=0.8, line_width=2)
                    fig.add_vline(x=ORDERS_THRESHOLD, line_dash="dash", line_color="#94A3B8", opacity=0.8, line_width=2)

                    # TẠO CHÚ THÍCH NGƯỠNG ĐỘC LẬP - Chỉnh vị trí tránh bị đè
                    fig.add_annotation(
                        x=max_orders_plot, y=SPEND_THRESHOLD,
                        text=f"Ngưỡng chi tiêu: ${SPEND_THRESHOLD:,.0f}",
                        showarrow=False, xanchor="right", yanchor="bottom",
                        yshift=6, xshift=-6, # Dịch lên trên và sang trái
                        font=dict(size=11, color="#64748B", weight="bold"),
                        bgcolor="rgba(255,255,255,0.7)" # Nền trắng mờ
                    )

                    fig.add_annotation(
                        x=ORDERS_THRESHOLD, y=0,
                        text=f"Ngưỡng số đơn: {ORDERS_THRESHOLD}",
                        showarrow=False, xanchor="left", yanchor="bottom",
                        xshift=6, yshift=6, # Dịch sang phải và lên trên
                        font=dict(size=11, color="#64748B", weight="bold"),
                        bgcolor="rgba(255,255,255,0.7)" # Nền trắng mờ
                    )

                    # Đánh dấu khách hàng đang phân tích
                    fig.add_trace(go.Scatter(
                        x=[pred_orders], y=[pred_spend],
                        mode="markers+text",
                        marker=dict(size=22, color=cfg["color"],
                                    symbol="star", line=dict(width=3, color="white")),
                        text=["Khách hàng này"],
                        textposition="top center",
                        textfont=dict(size=13, color=cfg["color"], weight="bold"),
                        name="Khách hàng hiện tại",
                        hoverinfo="x+y"
                    ))

                    fig.update_layout(
                        height=420,
                        margin=dict(l=0, r=20, t=30, b=0),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        showlegend=False,
                        xaxis=dict(range=[0, max_orders_plot], title="Số đơn / năm", showgrid=False, zeroline=False),
                        yaxis=dict(range=[0, max_spend_plot], title="Chi tiêu / năm (USD)", showgrid=False, zeroline=False),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_crm:
                    st.markdown(
                        f"<div style='font-size:13px;font-weight:700;text-transform:uppercase;"
                        f"letter-spacing:0.05em;color:#4F46E5;margin-bottom:12px;'>"
                        f"Hành động CRM được gợi ý</div>",
                        unsafe_allow_html=True,
                    )
                    for action in CRM_ACTIONS[segment]:
                        st.markdown(
                            f"<div class='action-item' style='border-color:{cfg['color']};'>{action}</div>",
                            unsafe_allow_html=True,
                        )

            # Lưu session cho tab 2
            st.session_state["last_input"]  = features
            st.session_state["last_result"] = (pred_spend, pred_orders, segment)

# ══════════════════════════════════════════
# TAB 2 — MÔ PHỎNG KỊCH BẢN
# ══════════════════════════════════════════
with tab2:
    if "last_input" not in st.session_state:
        st.info("Hãy thực hiện phân tích ở tab **Dự đoán khách hàng** trước, sau đó quay lại đây để mô phỏng các kịch bản thay đổi.")
    else:
        base_input = st.session_state["last_input"]
        base_spend, base_orders, base_seg = st.session_state["last_result"]

        st.markdown(
            "<p style='color:#64748B;font-size:15px;margin-bottom:16px;'>"
            "Kéo thanh trượt để thay đổi từng yếu tố — kết quả dự đoán sẽ tự động tính toán lại ngay lập tức."
            "</p>",
            unsafe_allow_html=True,
        )

        slider_box = st.container(border=True)
        with slider_box:
            st.markdown("<div class='field-group' style='margin-top:0;'>Thông số giả lập</div>", unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            sim_recency   = s1.slider("Số ngày từ lần mua cuối",
                                      0, 365, base_input["recency"], step=5)
            sim_orders    = s2.slider("Tổng số đơn hàng",
                                      1, 200, base_input["total_orders"], step=1)
            sim_spend     = s3.slider("Tổng chi tiêu (USD)",
                                      0, 50000, int(base_input["total_spend"]), step=100)
            s4, s5, _ = st.columns(3)
            sim_diversity = s4.slider("Số danh mục đã mua",
                                      1, 20, base_input["category_diversity"], step=1)
            sim_years     = s5.slider("Số năm là khách hàng",
                                      1, 10, base_input["years_active"], step=1)

        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

        if not model_loaded:
            st.warning("Chưa kết nối MLflow.")
        else:
            sim_features = {
                **base_input,
                "recency":            sim_recency,
                "total_orders":       sim_orders,
                "total_spend":        sim_spend,
                "avg_order_value":    sim_spend / max(sim_orders, 1),
                "category_diversity": sim_diversity,
                "years_active":       sim_years,
            }
            sim_spend_pred, sim_orders_pred = predict_customer(model_spend, model_orders, sim_features)
            sim_segment = classify_segment(sim_spend_pred, sim_orders_pred)
            sim_clv     = clv_estimate(sim_spend_pred)
            sim_cfg     = SEGMENT_CONFIG[sim_segment]

            delta_spend  = sim_spend_pred  - base_spend
            delta_orders = sim_orders_pred - base_orders
            delta_clv    = sim_clv - clv_estimate(base_spend)
            
            # Khắc phục hiển thị mũi tên Streamlit (Dấu âm phải đứng trước dấu $)
            delta_spend_str = f"{'-' if delta_spend < 0 else '+'}${abs(delta_spend):,.0f}" if delta_spend != 0 else "$0"
            delta_clv_str   = f"{'-' if delta_clv < 0 else '+'}${abs(delta_clv):,.0f}" if delta_clv != 0 else "$0"

            sim_result_box = st.container(border=True)
            with sim_result_box:
                head_l, head_r = st.columns([1, 3])
                with head_l:
                    if sim_segment != base_seg:
                        st.markdown(
                            f"<div style='padding:8px 16px;border-radius:8px;background:{sim_cfg['bg']};"
                            f"color:{sim_cfg['color']};font-weight:700;font-size:14px;display:inline-block; border: 1px solid {sim_cfg['color']}40;'>"
                            f"{base_seg} → {sim_segment}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<span class='segment-tag' style='background:{sim_cfg['bg']};color:{sim_cfg['color']}; border: 1px solid {sim_cfg['color']}40;'>"
                            f"{sim_segment}</span>",
                            unsafe_allow_html=True,
                        )
                with head_r:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Chi tiêu / năm",  f"${sim_spend_pred:,.0f}",  delta_spend_str)
                    m2.metric("Số đơn / năm",    f"{sim_orders_pred:.1f}",   f"{delta_orders:+.1f}")
                    m3.metric("CLV 3 năm",       f"${sim_clv:,.0f}",         delta_clv_str)

                st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

                col_chart2, col_crm2 = st.columns([3, 2], gap="large")

                with col_chart2:
                    st.markdown(
                        "<div style='font-size:15px;font-weight:700;color:#0F172A;margin-bottom:8px;'>"
                        "Vị thế phân khúc khách hàng (Mô phỏng)</div>",
                        unsafe_allow_html=True,
                    )
                    
                    fig_cmp = go.Figure()

                    # Hàm vẽ vùng phân khúc và Nhãn Badge ở góc (Tương tự Tab 1)
                    def add_segment_region(x0, x1, y0, y1, seg_name):
                        c = SEGMENT_CONFIG[seg_name]
                        fig_cmp.add_shape(
                            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                            fillcolor=c["bg"], line_width=0, layer="below"
                        )
                        label_x = x0 + (max(ORDERS_THRESHOLD * 2.2, sim_orders_pred * 1.3, 25) * 0.02)
                        label_y = y1 - (max(SPEND_THRESHOLD * 2.2, sim_spend_pred * 1.3, 6000) * 0.03)
                        
                        fig_cmp.add_annotation(
                            x=label_x, y=label_y,
                            text=seg_name.upper(), showarrow=False,
                            xanchor="left", yanchor="top",
                            font=dict(size=11, color=c["color"], weight="bold"),
                            bgcolor="white", bordercolor=c["color"], borderwidth=1, borderpad=4
                        )

                    max_orders_plot_sim = max(ORDERS_THRESHOLD * 2.2, sim_orders_pred * 1.3, 25)
                    max_spend_plot_sim  = max(SPEND_THRESHOLD * 2.2, sim_spend_pred * 1.3, 6000)

                    add_segment_region(0, ORDERS_THRESHOLD, 0, SPEND_THRESHOLD, "Rủi ro rời bỏ")
                    add_segment_region(0, ORDERS_THRESHOLD, SPEND_THRESHOLD, max_spend_plot_sim, "Tiềm năng")
                    add_segment_region(ORDERS_THRESHOLD, max_orders_plot_sim, 0, SPEND_THRESHOLD, "Trung bình")
                    add_segment_region(ORDERS_THRESHOLD, max_orders_plot_sim, SPEND_THRESHOLD, max_spend_plot_sim, "VIP")

                    fig_cmp.add_hline(y=SPEND_THRESHOLD, line_dash="dash", line_color="#94A3B8", opacity=0.8, line_width=2)
                    fig_cmp.add_vline(x=ORDERS_THRESHOLD, line_dash="dash", line_color="#94A3B8", opacity=0.8, line_width=2)

                    fig_cmp.add_annotation(
                        x=max_orders_plot_sim, y=SPEND_THRESHOLD,
                        text=f"Ngưỡng chi tiêu: ${SPEND_THRESHOLD:,.0f}",
                        showarrow=False, xanchor="right", yanchor="bottom",
                        yshift=6, xshift=-6,
                        font=dict(size=11, color="#64748B", weight="bold"),
                        bgcolor="rgba(255,255,255,0.7)"
                    )

                    fig_cmp.add_annotation(
                        x=ORDERS_THRESHOLD, y=0,
                        text=f"Ngưỡng số đơn: {ORDERS_THRESHOLD}",
                        showarrow=False, xanchor="left", yanchor="bottom",
                        xshift=6, yshift=6, # Đưa xuống dưới trục X
                        font=dict(size=11, color="#64748B", weight="bold"),
                        bgcolor="rgba(255,255,255,0.7)"
                    )

                    # Marker cũ (Màu nhạt)
                    fig_cmp.add_trace(go.Scatter(
                        x=[base_orders], y=[base_spend],
                        mode="markers+text",
                        marker=dict(size=16, color="#CBD5E1", symbol="circle"),
                        text=["Hiện tại"],
                        textposition="bottom center",
                        textfont=dict(size=11, color="#94A3B8"),
                        name="Hiện tại",
                        hoverinfo="x+y"
                    ))

                    # Marker mới (Ngôi sao, màu đậm)
                    fig_cmp.add_trace(go.Scatter(
                        x=[sim_orders_pred], y=[sim_spend_pred],
                        mode="markers+lines+text",
                        marker=dict(size=22, color=sim_cfg["color"],
                                    symbol="star", line=dict(width=3, color="white")),
                        text=["Sau mô phỏng"],
                        textposition="top center",
                        textfont=dict(size=13, color=sim_cfg["color"], weight="bold"),
                        name="Sau mô phỏng",
                        hoverinfo="x+y"
                    ))
                    
                    # Vẽ đường nối giữa cũ và mới
                    fig_cmp.add_trace(go.Scatter(
                        x=[base_orders, sim_orders_pred], y=[base_spend, sim_spend_pred],
                        mode="lines",
                        line=dict(color=sim_cfg["color"], width=2, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip"
                    ))

                    fig_cmp.update_layout(
                        height=420,
                        margin=dict(l=0, r=20, t=30, b=0),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        showlegend=False,
                        xaxis=dict(range=[0, max_orders_plot_sim], title="Số đơn / năm", showgrid=False, zeroline=False),
                        yaxis=dict(range=[0, max_spend_plot_sim], title="Chi tiêu / năm (USD)", showgrid=False, zeroline=False),
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

                with col_crm2:
                    st.markdown(
                        f"<div style='font-size:13px;font-weight:700;text-transform:uppercase;"
                        f"letter-spacing:0.05em;color:#4F46E5;margin-bottom:12px;'>"
                        f"Hành động CRM thay đổi</div>",
                        unsafe_allow_html=True,
                    )
                    for action in CRM_ACTIONS[sim_segment]:
                        st.markdown(
                            f"<div class='action-item' style='border-color:{sim_cfg['color']};'>{action}</div>",
                            unsafe_allow_html=True,
                        )