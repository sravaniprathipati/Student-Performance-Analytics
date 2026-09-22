import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from io import BytesIO
import base64
import numpy as np

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="🎓 Student Performance Analytics & Early Academic Risk Identification",
    page_icon="🎯",
    layout="wide"
)


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


try:
    img_base64 = get_base64_image("background.png")

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: "Segoe UI", sans-serif;
    }}
    .block-container {{
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    .glass-card {{
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 28px;
        border: 1px solid rgba(255, 150, 120, 0.35);
        box-shadow: 0 12px 28px rgba(120, 55, 35, 0.12);
        margin-bottom: 24px;
    }}
    .hero-title {{
        font-size: 38px;
        font-weight: 850;
        color: #6b2e1f;
        margin-bottom: 8px;
    }}
    .hero-subtitle {{
        font-size: 17px;
        color: #8a3f2b;
        font-weight: 500;
    }}
    h1, h2, h3, h4, label, p, span {{
        color: #6b2e1f !important;
    }}
    [data-testid="stSidebar"] {{
        background: rgba(255, 238, 230, 0.82);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 150, 120, 0.35);
    }}
    [data-testid="stMetric"] {{
        background: rgba(255, 255, 255, 0.78);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 150, 120, 0.35);
        box-shadow: 0 8px 20px rgba(120, 55, 35, 0.10);
    }}
    .stButton > button {{
        background: linear-gradient(135deg, #ff9a8b, #f97316);
        color: white;
        border-radius: 14px;
        padding: 0.7rem 1.4rem;
        border: none;
        font-weight: 700;
    }}
    .stDownloadButton > button {{
        background: linear-gradient(135deg, #34d399, #10b981);
        color: white;
        border-radius: 14px;
        padding: 0.7rem 1.4rem;
        border: none;
        font-weight: 700;
    }}
    [data-testid="stPlotlyChart"] {{
        background: rgba(255, 255, 255, 0.82);
        border-radius: 20px;
        padding: 14px;
        border: 1px solid rgba(255, 150, 120, 0.25);
    }}
    [data-testid="stDataFrame"] {{
        border-radius: 18px;
        overflow: hidden;
    }}
    </style>
    """, unsafe_allow_html=True)

except FileNotFoundError:
    st.warning("⚠️ background.png not found. App will run without background image.")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip().str.upper()

    required_original_cols = [
        "CAT1", "CAT2", "QUIZ1", "QUIZ2", "QUIZ3",
        "ATTENDANCE PERCENTAGE", "FAT"
    ]

    missing_cols = [col for col in required_original_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data.csv: {missing_cols}")

    df = df[required_original_cols].copy()

    df.rename(columns={
        "CAT1": "mid1",
        "CAT2": "mid2",
        "QUIZ1": "quiz1",
        "QUIZ2": "quiz2",
        "QUIZ3": "quiz3",
        "ATTENDANCE PERCENTAGE": "attendance",
        "FAT": "final_exam"
    }, inplace=True)

    numeric_cols = ["mid1", "mid2", "quiz1", "quiz2", "quiz3", "attendance", "final_exam"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    df["Student_ID"] = ["Student " + str(i + 1) for i in range(len(df))]
    return df


try:
    predictor = load_model()
except Exception as e:
    st.error(f"❌ Could not load model.pkl: {e}")
    st.stop()

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Could not load data.csv: {e}")
    st.stop()

if df.empty:
    st.error("❌ Dataset is empty after cleaning. Check data.csv.")
    st.stop()


st.markdown("""
<div class="glass-card">
    <div class="hero-title">🎓 Student Performance Analytics & Early Academic Risk Identification</div>
    <div class="hero-subtitle">
    academic risk prediction and intervention support system
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.success("✅ Offline model loaded")
st.sidebar.info(f"📊 Students loaded: {len(df)}")


st.subheader("🎯 Select Student")
selected_student = st.selectbox("Choose a student", df["Student_ID"])

filtered_df = df[df["Student_ID"] == selected_student]

if filtered_df.empty:
    st.error("❌ Selected student not found.")
    st.stop()

student_row = filtered_df.iloc[0]

st.write("### 📊 Student Data")
display_row = student_row.drop(["final_exam"], errors="ignore")
st.dataframe(display_row.to_frame().T)


def risk_color(risk):
    if risk in ["VERY HIGH", "HIGH"]:
        return "red"
    elif risk == "MODERATE":
        return "orange"
    return "green"


def generate_pdf(student_name, cat2_result, fat_result, student_row, risk_df, lci_df, actions):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph(f"<b>Student Risk Report: {student_name}</b>", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Academic Details</b>", styles["Heading2"]))
    pdf_display_row = student_row.drop(["final_exam"], errors="ignore")

    for key, value in pdf_display_row.to_dict().items():
        content.append(Paragraph(f"{key}: {value}", styles["Normal"]))

    content.append(Spacer(1, 12))

    cat2_fail = cat2_result.get("fail_probability", 0)
    fat_fail = fat_result.get("fail_probability", 0)

    content.append(Paragraph("<b>Risk Summary</b>", styles["Heading2"]))
    content.append(Paragraph(
        f"<font color='{risk_color(cat2_result['risk_level'])}'>"
        f"Mid 2 Risk: {cat2_result['risk_level']} | P(Fail): {cat2_fail:.1%}"
        f"</font>",
        styles["Normal"]
    ))
    content.append(Paragraph(
        f"<font color='{risk_color(fat_result['risk_level'])}'>"
        f"Final Exam Risk: {fat_result['risk_level']} | P(Fail): {fat_fail:.1%}"
        f"</font>",
        styles["Normal"]
    ))

    content.append(Spacer(1, 12))

    chart_buffer = BytesIO()
    ax = risk_df.plot(
        kind="barh",
        x="Factor",
        y="Impact (%)",
        figsize=(6, 4),
        legend=False
    )
    ax.set_title("Actual Contribution to Failure Risk")
    ax.set_xlabel("Actual Risk Impact (%)")
    plt.tight_layout()
    plt.savefig(chart_buffer, format="png")
    plt.close()
    chart_buffer.seek(0)

    content.append(Paragraph("<b>Risk Decomposition</b>", styles["Heading2"]))
    content.append(Image(chart_buffer, width=400, height=250))
    content.append(Spacer(1, 12))

    lci_chart_buffer = BytesIO()
    ax2 = lci_df.plot(
        kind="bar",
        x="Component",
        y="Score (%)",
        figsize=(6, 4),
        legend=False
    )
    ax2.set_title("Learning Readiness Breakdown")
    ax2.set_ylabel("Readiness Score (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(lci_chart_buffer, format="png")
    plt.close()
    lci_chart_buffer.seek(0)

    content.append(Paragraph("<b>Learning Readiness Index</b>", styles["Heading2"]))
    content.append(Image(lci_chart_buffer, width=400, height=250))
    content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Smart Action Plan</b>", styles["Heading2"]))
    for action in actions:
        content.append(Paragraph(f"• {action}", styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer


def create_feature_importance_donut(fi_df, stage_key):
    fi_df = fi_df.copy()

    name_map = {
        "weighted_stage1": "Stage 1 Performance",
        "lci_stage1": "Learning Consistency",
        "mid1": "Mid1 Score",
        "mid1_n": "Mid1 Score",
        "quiz1": "Quiz 1",
        "quiz1_n": "Quiz 1",
        "gap_mid1_quiz1": "Mid1-Quiz Gap",
        "low_mid1_flag": "Low Mid1 Flag",
        "mid2": "Mid2 Score",
        "attendance": "Attendance",
        "quiz2": "Quiz 2",
        "quiz3": "Quiz 3",
        "quiz_avg": "Quiz Average"
    }

    fi_df["Display Feature"] = fi_df["Feature"].replace(name_map)
    fi_df = fi_df.sort_values("Importance", ascending=False)

    donut_df = fi_df.head(3).copy()

    donut_df["Importance (%)"] = (
        donut_df["Importance"] / donut_df["Importance"].sum() * 100
    )

    title = (
        "🚀 Top 3 Drivers of Mid2 Prediction"
        if stage_key == "mid2"
        else "🎯 Top 3 Drivers of Final Exam Prediction"
    )

    fig = px.pie(
        donut_df,
        names="Display Feature",
        values="Importance (%)",
        hole=0.6,
        title=title
    )

    fig.update_traces(
        textinfo="percent+label",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>Influence: %{value:.1f}%<extra></extra>"
    )

    fig.update_layout(
        height=430,
        showlegend=True,
        legend_title_text="Top Factors",
        title_x=0.5,
        annotations=[
            dict(
                text="Top 3",
                x=0.5,
                y=0.5,
                font_size=18,
                showarrow=False
            )
        ],
        margin=dict(l=20, r=20, t=70, b=20)
    )

    return fig, donut_df


if st.button("🔍 Predict Selected Student"):

    cat1 = student_row.get("mid1", 0)
    mid2 = student_row.get("mid2", 0)
    quiz1 = student_row.get("quiz1", 0)
    quiz2 = student_row.get("quiz2", 0)
    quiz3 = student_row.get("quiz3", 0)
    attendance = student_row.get("attendance", 75)

    cat2_result = predictor.predict_cat2_risk(cat1, quiz1)

    fat_result = predictor.predict_fat_risk(
        cat2_result,
        quiz2,
        quiz3,
        attendance,
        mid2=mid2
    )

    cat2_fail_prob = cat2_result.get("fail_probability", 0)
    fat_fail_prob = fat_result.get("fail_probability", 0)

    st.success(f"📌 {selected_student} Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.error(f"Mid 2 Risk: {cat2_result['risk_level']}")
        st.metric("P(Fail Mid 2)", f"{cat2_fail_prob:.1%}")

    with col2:
        st.warning(f"Final Exam Risk: {fat_result['risk_level']}")
        st.metric("P(Fail Final Exam)", f"{fat_fail_prob:.1%}")

    st.subheader("📊 Model Performance")

    metrics = predictor.get_main_metrics()

    st.markdown("### 🔁 Cross-Validation Performance")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.metric("Mid 2 F1 Score", f"{metrics.get('mid2_f1', 0):.2%}")
        st.caption(f"± {metrics.get('mid2_f1_cv_std', 0):.2f} CV stability")

    with col_m2:
        st.metric("Final Exam F1 Score", f"{metrics.get('final_exam_f1', 0):.2%}")
        st.caption(f"± {metrics.get('final_exam_f1_cv_std', 0):.2f} CV stability")

    st.caption("Primary evaluation method: Stratified K-Fold Cross Validation")

    if "test_size" in metrics:
        st.markdown("### 🧪 Holdout Test Performance")
        st.write(f"Test size: {metrics.get('test_size', 0)} students")
        st.write(f"Mid2 Test F1: {metrics.get('mid2_test_f1', 0):.2%}")
        st.write(f"Final Exam Test F1: {metrics.get('final_exam_test_f1', 0):.2%}")
        st.caption("Note: Test set is small, so results may vary.")

    st.subheader("📌 Global Feature Importance")

    fi_stage = st.selectbox(
        "Select model for feature importance",
        ["Mid 2 Model", "Final Exam Model"]
    )

    stage_key = "mid2" if fi_stage == "Mid 2 Model" else "final"
    fi_df = predictor.get_feature_importance(stage_key)

    fig_fi, donut_df = create_feature_importance_donut(fi_df, stage_key)

    st.plotly_chart(fig_fi, use_container_width=True)

    st.caption(
        "This donut chart shows only the top 3 most influential factors used by the model."
    )

    with st.expander("View feature importance values"):
        display_donut = donut_df.copy()
        display_donut["Importance (%)"] = display_donut["Importance (%)"].round(2)
        st.dataframe(display_donut)

    # -------------------------------
    # SHAP-STYLE WATERFALL EXPLANATION
    # -------------------------------
    st.subheader("🔍 SHAP-Style Waterfall Explanation")

    local_exp_df = predictor.get_local_explanation(
        mid2,
        quiz2,
        quiz3,
        attendance
    )

    waterfall_df = local_exp_df.copy()

    waterfall_df["Risk Contribution"] = pd.to_numeric(
        waterfall_df["Risk Contribution"],
        errors="coerce"
    ).fillna(0)

    waterfall_df["Signed Contribution"] = waterfall_df.apply(
        lambda row: -abs(row["Risk Contribution"])
        if str(row.get("Effect", "")).lower() in [
            "negative",
            "reduce",
            "reduces risk",
            "risk reducing",
            "decreases risk"
        ]
        else abs(row["Risk Contribution"]),
        axis=1
    )

    # shrink waterfall for low-risk students
    if fat_fail_prob < 0.25:
        contribution_scale = 0.30
        waterfall_height = 320
        st.success("🟢 Low risk – feature contributions are minimal.")
    elif fat_fail_prob < 0.50:
        contribution_scale = 0.70
        waterfall_height = 400
        st.warning("🟡 Moderate risk – some features influence the prediction.")
    else:
        contribution_scale = 1.00
        waterfall_height = 470
        st.error("🔴 High risk – strong feature contributions are detected.")

    waterfall_df["Signed Contribution"] = (
        waterfall_df["Signed Contribution"] * contribution_scale
    )

    waterfall_df = waterfall_df.sort_values(
        "Signed Contribution",
        key=abs,
        ascending=True
    )

    final_impact = waterfall_df["Signed Contribution"].sum()

    fig_waterfall = go.Figure(go.Waterfall(
        name="Risk Explanation",
        orientation="v",
        measure=["relative"] * len(waterfall_df) + ["total"],
        x=list(waterfall_df["Feature"]) + ["Final Risk Impact"],
        y=list(waterfall_df["Signed Contribution"]) + [final_impact],
        text=[f"{v:.3f}" for v in waterfall_df["Signed Contribution"]] + [f"{final_impact:.3f}"],
        textposition="outside",
        connector={"line": {"width": 1}}
    ))

    fig_waterfall.update_layout(
        title="Student-Level Feature Contribution to Final Exam Risk",
        xaxis_title="Feature",
        yaxis_title="Risk Contribution",
        height=waterfall_height,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=80)
    )

    if fat_fail_prob < 0.25:
        fig_waterfall.update_yaxes(range=[-0.10, 0.10])
    elif fat_fail_prob < 0.50:
        fig_waterfall.update_yaxes(range=[-0.30, 0.30])

    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.dataframe(local_exp_df)

    st.caption(
        "This adaptive waterfall shows smaller contributions for low-risk students and stronger contributions for higher-risk students."
    )

    st.subheader("📊 Risk Decomposition (Actual Impact)")

    breakdown = predictor.get_risk_decomposition(
        mid2,
        quiz2,
        quiz3,
        attendance
    )

    scaled_breakdown = {k: v * fat_fail_prob for k, v in breakdown.items()}

    risk_df = pd.DataFrame({
        "Factor": list(scaled_breakdown.keys()),
        "Contribution": list(scaled_breakdown.values())
    })

    risk_df["Impact (%)"] = risk_df["Contribution"] * 100

    risk_df["Factor"] = risk_df["Factor"].replace({
        "CAT2_Risk": "Mid 2 Risk",
        "Quiz_Risk": "Quiz Risk",
        "Attendance_Risk": "Attendance Risk"
    })

    risk_df["Label"] = risk_df["Impact (%)"].apply(lambda x: f"{x:.2f}%")

    if risk_df["Impact (%)"].sum() == 0:
        top_factor = "None"
    else:
        top_factor = risk_df.loc[risk_df["Impact (%)"].idxmax(), "Factor"]

    risk_df["Color"] = risk_df["Factor"].apply(
        lambda x: "Top Risk Driver" if x == top_factor else "Other Factor"
    )

    fig = px.bar(
        risk_df,
        x="Impact (%)",
        y="Factor",
        orientation="h",
        text="Label",
        color="Color",
        title="Actual Contribution to Failure Risk",
        height=250
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Actual Risk Impact (%)",
        yaxis_title="Risk Factor",
        showlegend=True,
        xaxis_range=[0, 100]
    )

    st.plotly_chart(fig, use_container_width=False)

    if fat_fail_prob < 0.20:
        st.success("🟢 Student is safe. Overall failure risk is very low.")
    elif fat_fail_prob < 0.40:
        st.warning("🟡 Moderate risk. Monitor performance and review the top risk driver.")
    else:
        st.error("🔴 High risk. Immediate intervention is recommended.")

    if top_factor != "None":
        st.info(f"🎯 Primary risk driver: {top_factor}")

    st.caption("Risk contributions are based on normalized academic performance factors.")

    st.subheader("📈 Learning Readiness Index (LRI)")

    cat2_strength = mid2 / 50
    quiz_strength = ((quiz2 / 20) + (quiz3 / 20)) / 2
    attendance_strength = attendance / 100

    base_readiness = (
        0.45 * cat2_strength +
        0.30 * quiz_strength +
        0.25 * attendance_strength
    )

    risk_penalty = fat_fail_prob * 0.30
    risk_adjusted_readiness = base_readiness * (1 - risk_penalty)

    if min(cat2_strength, quiz_strength, attendance_strength) >= 0.70:
        risk_adjusted_readiness = max(risk_adjusted_readiness, 0.70)

    risk_adjusted_readiness = float(np.clip(risk_adjusted_readiness, 0, 1))

    lci_df = pd.DataFrame({
        "Component": [
            "Mid 2 Strength",
            "Quiz Strength",
            "Attendance Strength",
            "Final Readiness"
        ],
        "Score (%)": [
            cat2_strength * 100,
            quiz_strength * 100,
            attendance_strength * 100,
            risk_adjusted_readiness * 100
        ]
    })

    lci_df["Label"] = lci_df["Score (%)"].apply(lambda x: f"{x:.1f}%")

    fig_lci = px.bar(
        lci_df,
        x="Component",
        y="Score (%)",
        text="Label",
        title="Learning Readiness Breakdown",
        height=350
    )

    fig_lci.update_traces(textposition="outside")

    fig_lci.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="Target Readiness = 70%",
        annotation_position="top left"
    )

    fig_lci.update_layout(
        yaxis_range=[0, 100],
        yaxis_title="Readiness Score (%)",
        xaxis_title="Academic Component"
    )

    st.plotly_chart(fig_lci, use_container_width=True)

    readiness_percent = risk_adjusted_readiness * 100
    readiness_gap = max(0, 70 - readiness_percent)

    if fat_fail_prob < 0.1:
        st.success(f"✅ Strong readiness: {readiness_percent:.1f}% (Very low risk student)")
    elif readiness_percent >= 70:
        st.success(f"✅ Good readiness: {readiness_percent:.1f}%")
    elif readiness_percent >= 55:
        st.warning(f"⚠️ Moderate readiness: {readiness_percent:.1f}%. Readiness gap: {readiness_gap:.1f}%")
    else:
        st.error(f"🚨 Low readiness: {readiness_percent:.1f}%. Readiness gap: {readiness_gap:.1f}%")

    st.subheader("🎯 Smart Action Plan")

    actions = predictor.generate_action_plan(
        cat2_result,
        quiz2,
        quiz3,
        attendance,
        mid2=mid2
    )

    for action in actions:
        st.success(action)

    pdf = generate_pdf(
        selected_student,
        cat2_result,
        fat_result,
        student_row,
        risk_df,
        lci_df,
        actions
    )

    st.download_button(
        label="📄 Download Student Report",
        data=pdf,
        file_name=f"{selected_student}_report.pdf",
        mime="application/pdf"
    )