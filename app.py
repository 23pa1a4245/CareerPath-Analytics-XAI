from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lime_analysis import explain_profile_lime
from shap_analysis import calculate_global_shap, explain_profile_shap

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "hr_data.csv"
MODEL_PATH = ROOT / "models" / "attrition_model.joblib"
FEATURES_PATH = ROOT / "models" / "feature_columns.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.json"
SHAP_OUTPUT_DIR = ROOT / "reports" / "shap"
LIME_OUTPUT_DIR = ROOT / "reports" / "lime"

st.set_page_config(
    page_title="CareerPath Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at top left, #172554 0%, #0b1120 42%, #070b14 100%); color: #e5e7eb; }
[data-testid="stHeader"] { background: rgba(7, 11, 20, 0.78); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #101a2d 0%, #0b1120 100%); border-right: 1px solid #24334b; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px; }
[data-testid="stMetric"] { background: linear-gradient(135deg, rgba(30, 41, 59, .96), rgba(15, 23, 42, .96)); border: 1px solid #334155; padding: 17px 18px; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,.18); }
[data-testid="stMetricLabel"] { color: #a5b4fc !important; }
[data-testid="stMetricValue"] { color: #f8fafc !important; }
[data-testid="stAlert"] { border-radius: 12px; }
div[data-testid="stDataFrame"], div[data-testid="stTable"] { border: 1px solid #334155; border-radius: 12px; overflow: hidden; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; border: 1px solid #3b82f6; font-weight: 600; }
.stButton > button[kind="primary"] { background: linear-gradient(90deg, #2563eb, #7c3aed); border: 0; }
hr { border-color: #29364a; }
.small-note { font-size: .88rem; color: #aab7ca; }
[data-testid="stCaptionContainer"] { color: #aab7ca; }

/* Premium sidebar styling */
[data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #93c5fd; font-size: .78rem; letter-spacing: 1.7px;
    text-transform: uppercase; margin-top: .4rem; margin-bottom: .35rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 7px; padding-top: 8px; padding-bottom: 8px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    background: rgba(30, 41, 59, .58); border: 1px solid #26364f;
    border-radius: 10px; padding: 10px 12px; margin: 0;
    transition: background .15s ease, border-color .15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background: rgba(37, 99, 235, .18); border-color: #3b82f6;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(100deg, rgba(37, 99, 235, .34), rgba(124, 58, 237, .20));
    border: 1px solid #60a5fa;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label p {
    color: #e2e8f0; font-weight: 600; font-size: .88rem;
}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] > label {
    color: #cbd5e1; font-size: .85rem; font-weight: 600;
}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background-color: #111c30; border-color: #334155; border-radius: 10px;
}
[data-testid="stSidebar"] [data-testid="stAlert"] { border-radius: 10px; }
[data-testid="stSidebar"] hr { margin: 1rem 0; border-color: #26364f; }
.sidebar-status {
    padding: 11px 12px; border: 1px solid #285747; border-radius: 10px;
    background: rgba(16, 185, 129, .08); color: #a7f3d0; font-size: .82rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(path, modified_time):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    return data

@st.cache_resource
def load_model():
    if MODEL_PATH.exists() and FEATURES_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)
    return None, None

@st.cache_data
def load_metrics(modified_time):
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data(show_spinner=False)
def cached_global_shap(_model, reference_data, feature_columns, model_version):
    return calculate_global_shap(
        _model,
        reference_data,
        feature_columns,
        output_dir=SHAP_OUTPUT_DIR,
    )

@st.cache_data(show_spinner=False)
def cached_profile_shap(
    _model, profile, feature_columns, profile_id, model_version
):
    return explain_profile_shap(
        _model,
        profile,
        feature_columns,
        output_dir=SHAP_OUTPUT_DIR,
        profile_id=profile_id,
    )

@st.cache_data(show_spinner=False)
def cached_profile_lime(
    _model, reference_data, profile, feature_columns, profile_id, model_version
):
    return explain_profile_lime(
        _model,
        reference_data,
        profile,
        feature_columns,
        output_dir=LIME_OUTPUT_DIR,
        profile_id=profile_id,
    )

if not DATA_PATH.exists():
    st.error("Dataset file not found.")
    st.markdown(
        "Save the IBM HR Analytics CSV as **`hr_data.csv`** in the same folder as `app.py`."
    )
    st.markdown("Then run `python train_model.py` followed by `streamlit run app.py`.")
    st.stop()

df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)

required = [
    "Department", "JobLevel", "TotalWorkingYears", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "MonthlyIncome",
    "TrainingTimesLastYear", "Attrition"
]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("This CSV is missing expected columns: " + ", ".join(missing))
    st.stop()



# PROFESSIONAL CAREERPATH SIDEBAR
st.sidebar.markdown("""
<div style="
    padding: 18px 14px;
    border-radius: 16px;
    background: linear-gradient(135deg, #172554, #111827);
    border: 1px solid #334155;
    margin-bottom: 18px;
">
    <div style="font-size: 25px; font-weight: 800; color: #60a5fa;">
        ◈ CareerPath
    </div>
    <div style="font-size: 10px; letter-spacing: 2px;
                color: #cbd5e1; margin-top: 8px;">
        ANALYTICS STUDIO
    </div>
    <div style="font-size: 12px; color: #94a3b8; margin-top: 5px;">
        CAREER INTELLIGENCE PLATFORM
    </div>
</div>
""", unsafe_allow_html=True)



page_labels = {
    "Executive Overview": "◉   Executive Overview",
    "Career Progression": "↗   Career Progression",
    "Workforce Planning": "▦   Workforce Planning",
    "ML Attrition Lab": "✦   ML Attrition Lab",
    "Recommendations & Limitations": "✓   Recommendations & Limitations",
    "Career Assistant": "✧ Career Assistant",
}

# Professional navigation using clickable sidebar buttons
if "careerpath_navigation" not in st.session_state:
    st.session_state.careerpath_navigation = "Executive Overview"

st.sidebar.markdown("### WORKSPACE")
st.sidebar.caption("Navigate your career intelligence workspace")

for nav_name, nav_label in page_labels.items():
    is_active = st.session_state.careerpath_navigation == nav_name

    if st.sidebar.button(
        nav_label,
        key=f"nav_button_{nav_name}",
        use_container_width=True,
        type="primary" if is_active else "secondary"
    ):
        st.session_state.careerpath_navigation = nav_name
        st.rerun()

page = st.session_state.careerpath_navigation


st.sidebar.divider()

# ================= TOP HEADER FILTERS =================

st.markdown("""
<style>
.header-filter-box {
    background: linear-gradient(135deg, #172554, #111827);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 16px 20px 4px 20px;
    margin: 10px 0 22px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-filter-box">', unsafe_allow_html=True)
st.markdown("### Workforce Filters")
st.caption("Choose departments and company tenure to update dashboard insights.")

filter_col1, filter_col2 = st.columns(2)

departments = sorted(df["Department"].dropna().unique().tolist())

with filter_col1:
    selected_departments = st.multiselect(
        "Department",
        options=departments,
        default=departments,
        key="header_departments"
    )

with filter_col2:
    if "YearsAtCompany" in df.columns:
        company_years = pd.to_numeric(
            df["YearsAtCompany"], errors="coerce"
        ).dropna()

        min_year = int(company_years.min())
        max_year = int(company_years.max())

        if min_year < max_year:
            selected_year_range = st.slider(
                "Years at Company",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                key="header_company_years"
            )
        else:
            selected_year_range = (min_year, max_year)
    else:
        selected_year_range = None
        st.info("Years at Company is unavailable in this dataset.")

st.markdown('</div>', unsafe_allow_html=True)

# ================= APPLY FILTERS =================

data = df[df["Department"].isin(selected_departments)].copy()

if selected_year_range is not None and "YearsAtCompany" in data.columns:
    company_tenure = pd.to_numeric(
        data["YearsAtCompany"], errors="coerce"
    )
    data = data[
        company_tenure.between(
            selected_year_range[0],
            selected_year_range[1]
        )
    ].copy()

if data.empty:
    st.warning("No records match these filters. Please adjust your selections.")
    st.stop()

if data.empty:
    st.warning("No records match these filters. Choose at least one department and job level.")
    st.stop()

st.title("◈ CareerPath Analytics")
st.caption("PEOPLE ANALYTICS  /  CAREER DEVELOPMENT  /  WORKFORCE INTELLIGENCE")
st.markdown(
    "An interactive analytics application for exploring career indicators, "
    "workforce composition, training, compensation, and attrition patterns."
)

if page == "Executive Overview":
    st.subheader("Executive Overview")
    total = len(data)
    avg_exp = pd.to_numeric(data["TotalWorkingYears"], errors="coerce").mean()
    avg_income = pd.to_numeric(data["MonthlyIncome"], errors="coerce").mean()
    avg_promo = pd.to_numeric(data["YearsSinceLastPromotion"], errors="coerce").mean()
    attrition_labels = data["Attrition"].astype(str).str.strip().str.lower()
    valid_attrition = attrition_labels.isin(["yes", "no"])
    attrition_rate = (
        attrition_labels[valid_attrition].eq("yes").mean() * 100
        if valid_attrition.any() else np.nan
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees in view", f"{total:,}")
    c2.metric("Average experience", f"{avg_exp:.1f} years")
    c3.metric("Average monthly income", f"{avg_income:,.0f}")
    c4.metric(
        "Attrition rate",
        f"{attrition_rate:.1f}%" if pd.notna(attrition_rate) else "N/A"
    )
    profile_columns = [
        col for col in data.columns
        if col != "EmployeeNumber"
    ]

    if profile_columns:
        repeated_profiles = int(
            data.duplicated(subset=profile_columns).sum()
        )
        repeated_pct = repeated_profiles / len(data) * 100

        st.markdown("### Dataset Quality")
        q1, q2 = st.columns(2)
        q1.metric("Repeated profiles (excluding ID)", f"{repeated_profiles:,}")
        q2.metric("Repeated profile share", f"{repeated_pct:.1f}%")
        if repeated_profiles:
            st.warning(
                "Repeated profiles are not independent employee observations. "
                "Treat model metrics as educational only."
            )
        else:
            st.info("No repeated profiles were found in the filtered records.")

    left, right = st.columns(2)
    with left:
        counts = data["JobLevel"].value_counts().sort_index().rename_axis(
            "Job Level"
        ).reset_index(name="Employees")
        fig = px.bar(
            counts, x="Job Level", y="Employees", text="Employees",
            title="Workforce by Job Level", color="Job Level"
        )
        fig.update_layout(showlegend=False)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#111827", font_color="#e5e7eb", title_font_color="#f8fafc", margin=dict(l=30, r=25, t=65, b=35))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        dept = data.groupby("Department", as_index=False).agg(
            Employees=("Department", "size"),
            AverageIncome=("MonthlyIncome", "mean")
        )
        fig = px.bar(
            dept, x="Department", y="Employees", color="Department",
            title="Workforce by Department", text="Employees"
        )
        fig.update_layout(showlegend=False)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#111827", font_color="#e5e7eb", title_font_color="#f8fafc", margin=dict(l=30, r=25, t=65, b=35))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.box(
            data, x="JobLevel", y="YearsInCurrentRole",
            title="Years in Current Role by Job Level",
            labels={"JobLevel": "Job Level", "YearsInCurrentRole": "Years in Current Role"}
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#111827", font_color="#e5e7eb", title_font_color="#f8fafc", margin=dict(l=30, r=25, t=65, b=35))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        attr = data.assign(
            AttritionLabel=data["Attrition"].astype(str).str.strip().str.title()
        )
        attr = attr.groupby(
            ["Department", "AttritionLabel"]
        ).size().reset_index(name="Employees")
        fig = px.bar(
            attr, x="Department", y="Employees", color="AttritionLabel",
            barmode="group", title="Stayed vs Left by Department"
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#111827", font_color="#e5e7eb", title_font_color="#f8fafc", margin=dict(l=30, r=25, t=65, b=35))
        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download filtered data (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="careerpath_filtered_data.csv",
        mime="text/csv"
    )
    summary = {
        "employees_in_view": int(len(data)),
        "average_total_working_years": round(float(avg_exp), 2) if pd.notna(avg_exp) else None,
        "average_monthly_income": round(float(avg_income), 2) if pd.notna(avg_income) else None,
        "average_years_since_last_promotion": round(float(avg_promo), 2) if pd.notna(avg_promo) else None,
        "attrition_rate_percent": round(float(attrition_rate), 2) if pd.notna(attrition_rate) else None,
        "departments_in_view": sorted(data["Department"].dropna().astype(str).unique().tolist()),
        "note": "Descriptive analysis of a synthetic static dataset; not causal or suitable for individual HR decisions."
    }
    st.download_button(
        "Download executive summary (JSON)",
        data=json.dumps(summary, indent=2).encode("utf-8"),
        file_name="careerpath_executive_summary.json",
        mime="application/json"
    )

elif page == "Career Progression":
    st.subheader("Career Progression Indicators")
    st.info(
        "This dataset is a static snapshot. It supports analysis of career-related indicators, "
        "but does not include chronological promotion events or a verified promotion outcome. "
        "Patterns below describe the filtered workforce; they do not prove promotion history or readiness."
    )
    progress_metrics = st.columns(3)
    progress_metrics[0].metric(
        "Average total experience",
        f"{pd.to_numeric(data['TotalWorkingYears'], errors='coerce').mean():.1f} years"
    )
    progress_metrics[1].metric(
        "Average time in current role",
        f"{pd.to_numeric(data['YearsInCurrentRole'], errors='coerce').mean():.1f} years"
    )
    progress_metrics[2].metric(
        "Average years since last promotion",
        f"{pd.to_numeric(data['YearsSinceLastPromotion'], errors='coerce').mean():.1f} years"
    )
    left, right = st.columns(2)
    with left:
        fig = px.box(data, x="JobLevel", y="YearsSinceLastPromotion",
                     color="JobLevel", title="Years Since Last Promotion by Job Level",
                     labels={"JobLevel": "Job Level", "YearsSinceLastPromotion": "Years Since Last Promotion"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#111827", font_color="#e5e7eb", title_font_color="#f8fafc", margin=dict(l=30, r=25, t=65, b=35))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(data, x="TotalWorkingYears", y="JobLevel",
                         color="Department", size="YearsAtCompany",
                         hover_data=["YearsInCurrentRole", "YearsSinceLastPromotion"],
                         title="Experience vs Job Level",
                         labels={"TotalWorkingYears": "Total Working Years", "JobLevel": "Job Level"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#111827", font_color="#e5e7eb", title_font_color="#f8fafc", margin=dict(l=30, r=25, t=65, b=35))
        st.plotly_chart(fig, use_container_width=True)

    role = data.groupby("JobRole", as_index=False).agg(
        Employees=("JobRole", "size"),
        MeanRoleYears=("YearsInCurrentRole", "mean"),
        MeanYearsSincePromotion=("YearsSinceLastPromotion", "mean"),
        MeanExperience=("TotalWorkingYears", "mean")
    ) if "JobRole" in data.columns else pd.DataFrame()
    if not role.empty:
        st.markdown("**Career indicators by job role**")
        st.dataframe(role.round(2), use_container_width=True, hide_index=True)

    st.markdown("**Development review indicator (group-level)**")
    threshold = st.slider("Flag groups with average years since promotion at or above", 1, 10, 3)
    group = data.groupby(["Department", "JobLevel"], as_index=False).agg(
        Employees=("JobLevel", "size"),
        AverageYearsSincePromotion=("YearsSinceLastPromotion", "mean"),
        AverageYearsInRole=("YearsInCurrentRole", "mean")
    )
    flagged = group[group["AverageYearsSincePromotion"] >= threshold].sort_values(
        "AverageYearsSincePromotion", ascending=False
    )
    st.dataframe(flagged.round(2), use_container_width=True, hide_index=True)
    st.caption(
        "These are group-level prompts for further review, not individual promotion recommendations."
    )
    if not role.empty:
        longest_role = role.loc[role["MeanRoleYears"].idxmax()]
        longest_promotion = role.loc[role["MeanYearsSincePromotion"].idxmax()]
        st.markdown("**Filtered-snapshot observations**")
        st.write(
            f"- **{longest_role['JobRole']}** has the highest average time in its current role "
            f"({longest_role['MeanRoleYears']:.1f} years) in this view."
        )
        st.write(
            f"- **{longest_promotion['JobRole']}** has the highest average years since last promotion "
            f"({longest_promotion['MeanYearsSincePromotion']:.1f}) in this view."
        )
        st.write(
            "- Use these group-level patterns to guide development conversations and check access "
            "to learning opportunities; validate them with longitudinal records and employee context."
        )

elif page == "Workforce Planning":
    st.subheader("Workforce Planning")
    left, right = st.columns(2)
    with left:
        dept_level = data.groupby(["Department", "JobLevel"], as_index=False).size()
        fig = px.bar(dept_level, x="Department", y="size", color="JobLevel",
                     barmode="stack", title="Department Workforce by Job Level",
                     labels={"size": "Employees", "JobLevel": "Job Level"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#111827", font_color="#e5e7eb", title_font_color="#f8fafc", margin=dict(l=30, r=25, t=65, b=35))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        income = data.groupby("JobLevel", as_index=False)["MonthlyIncome"].mean()
        fig = px.bar(income, x="JobLevel", y="MonthlyIncome", title="Average Monthly Income by Job Level",
                     labels={"JobLevel": "Job Level", "MonthlyIncome": "Average Monthly Income"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#111827", font_color="#e5e7eb", title_font_color="#f8fafc", margin=dict(l=30, r=25, t=65, b=35))
        st.plotly_chart(fig, use_container_width=True)

    train = data.groupby("TrainingTimesLastYear", as_index=False).agg(
        Employees=("TrainingTimesLastYear", "size"),
        AverageIncome=("MonthlyIncome", "mean"),
        AverageRoleYears=("YearsInCurrentRole", "mean")
    )
    fig = px.bar(train, x="TrainingTimesLastYear", y="Employees",
                 title="Employee Count by Training Frequency",
                 labels={"TrainingTimesLastYear": "Training Frequency Last Year"})
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#111827", font_color="#e5e7eb", title_font_color="#f8fafc", margin=dict(l=30, r=25, t=65, b=35))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(train.round(2), use_container_width=True, hide_index=True)

    attrition_labels = data["Attrition"].astype(str).str.strip().str.lower()
    attr = data.loc[attrition_labels.isin(["yes", "no"])].copy()
    attr["LeftFlag"] = (
        attr["Attrition"].astype(str).str.strip().str.lower().eq("yes").astype(int)
    )
    attrition_by_dept = attr.groupby("Department", as_index=False).agg(
        Employees=("LeftFlag", "size"),
        EmployeesLeft=("LeftFlag", "sum"),
        AttritionRate=("LeftFlag", "mean")
    )
    attrition_by_dept["AttritionRate"] *= 100
    st.markdown("**Department retention indicators**")
    st.dataframe(attrition_by_dept.round(2), use_container_width=True, hide_index=True)

    department_training = data.assign(
        TrainingTimesLastYear=pd.to_numeric(
            data["TrainingTimesLastYear"], errors="coerce"
        )
    ).groupby("Department", as_index=False).agg(
        Employees=("TrainingTimesLastYear", "count"),
        AverageTrainingSessions=("TrainingTimesLastYear", "mean")
    ).dropna(subset=["AverageTrainingSessions"])
    department_share = data["Department"].value_counts(normalize=True) * 100
    largest_department = data["Department"].value_counts().idxmax()
    common_level = data["JobLevel"].mode().iloc[0]
    st.markdown("### Workforce planning observations")
    st.write(
        f"- **{largest_department}** is the largest department in this filtered view "
        f"({department_share[largest_department]:.1f}% of records)."
    )
    st.write(
        f"- Job level **{common_level}** is the most common level in the filtered workforce."
    )
    st.write(
        "- Review reported training access against future skill needs across departments."
    )
    if not department_training.empty:
        lowest_training_department = department_training.loc[
            department_training["AverageTrainingSessions"].idxmin()
        ]
        st.write(
            f"- **{lowest_training_department['Department']}** has the lowest average reported "
            f"training frequency ({lowest_training_department['AverageTrainingSessions']:.1f} "
            "sessions in the previous year)."
        )
    if not attrition_by_dept.empty:
        highest_attrition = attrition_by_dept.loc[
            attrition_by_dept["AttritionRate"].idxmax()
        ]
        st.write(
            f"- **{highest_attrition['Department']}** has the highest observed department attrition "
            f"rate ({highest_attrition['AttritionRate']:.1f}%) in this view."
        )
    st.markdown(
        "**Possible planning follow-ups**"
        "\n- Validate staffing coverage and critical-skill needs with department leads."
        "\n- Review whether learning access is aligned with role and skill requirements."
        "\n- Investigate retention patterns with qualitative context before changing plans."
    )
    st.caption(
        "These are descriptive signals from a static, filtered dataset—not forecasts, "
        "causal findings, or individual-level recommendations."
    )

elif page == "ML Attrition Lab":
    st.subheader("Machine Learning: Attrition Classification")
    st.warning(
        "Educational demonstration only. The source data is synthetic and this model must not be "
        "used to make real employment decisions about individuals."
    )
    model, feature_columns = load_model()
    metrics = load_metrics(
    METRICS_PATH.stat().st_mtime if METRICS_PATH.exists() else 0
)
    if model is None or metrics is None:
        st.error("The trained model was not found. Run `python train_model.py` in the project terminal first.")
        st.code("python train_model.py\nstreamlit run app.py", language="bash")
    else:
        st.markdown(f"**Selected model:** {metrics['selected_model']}")
        st.markdown("**Held-out test-set comparison**")
        rows = []
        for name, m in metrics["results"].items():
            rows.append({
                "Model": name,
                "Accuracy": m["accuracy"],
                "Balanced accuracy": m["balanced_accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1 score": m["f1"],
                "ROC-AUC": m["roc_auc"]
            })
        metric_df = pd.DataFrame(rows)
        st.dataframe(metric_df.style.format({
            "Accuracy": "{:.3f}", "Balanced accuracy": "{:.3f}",
            "Precision": "{:.3f}", "Recall": "{:.3f}", "F1 score": "{:.3f}", "ROC-AUC": "{:.3f}"
        }), use_container_width=True, hide_index=True)

        
        # Feature importance analysis
        importance_path = ROOT / "models" / "feature_importance.csv"

        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)

            st.markdown("### Top 10 Attrition Factors")
            st.caption(
                "Permutation importance measures how model ROC-AUC changes "
                "when a feature is shuffled. It does not prove causation."
            )

            top_features = importance_df.head(10).sort_values(
                "Importance", ascending=True
            )

            fig_importance = px.bar(
                top_features,
                x="Importance",
                y="Feature",
                orientation="h",
                error_x="ImportanceStd",
                title="Top 10 features by permutation importance",
                color="Importance",
                color_continuous_scale=["#1e3a8a", "#38bdf8"]
            )

            fig_importance.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Mean decrease in ROC-AUC after shuffling",
                yaxis_title="",
                showlegend=False
            )

            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Run `python train_model.py` to generate feature importance.")

        selected = metrics["selected_model"]
        m = metrics["results"][selected]
        cm = np.array(m["confusion_matrix"])
        fig = px.imshow(
            cm, text_auto=True, aspect="auto",
            x=["Predicted stayed", "Predicted left"],
            y=["Actual stayed", "Actual left"],
            labels=dict(x="Prediction", y="Actual", color="Count"),
            title=f"Confusion Matrix — {selected}"
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#111827", font_color="#e5e7eb", title_font_color="#f8fafc", margin=dict(l=30, r=25, t=65, b=35))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Accuracy alone can be misleading when fewer employees leave than stay. "
            "Review recall, precision, F1, ROC-AUC, and the confusion matrix together."
        )

        st.markdown("**Explore a hypothetical profile**")
        st.caption("This is a classroom demonstration, not an employee-level HR recommendation.")
        raw = df.drop(columns=[c for c in [
            "Attrition", "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours",
            "Age", "Gender", "MaritalStatus", "EducationField"
        ] if c in df.columns], errors="ignore")
        raw = raw.loc[:, raw.nunique(dropna=False) > 1]
        input_values = {}
        cols = st.columns(2)
        for i, col in enumerate(feature_columns):
            if col not in raw.columns:
                continue
            series = raw[col]
            with cols[i % 2]:
                if pd.api.types.is_numeric_dtype(series):
                    median = float(series.median())
                    minv, maxv = float(series.min()), float(series.max())
                    if minv == maxv:
                        input_values[col] = median
                    elif col in ["JobLevel", "Education", "EnvironmentSatisfaction", "JobInvolvement",
                                 "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction",
                                 "StockOptionLevel", "WorkLifeBalance", "TrainingTimesLastYear",
                                 "NumCompaniesWorked", "YearsAtCompany", "YearsInCurrentRole",
                                 "YearsSinceLastPromotion", "YearsWithCurrManager", "TotalWorkingYears"]:
                        input_values[col] = st.number_input(
                            col, min_value=int(minv), max_value=int(maxv),
                            value=int(np.clip(round(median), minv, maxv)), step=1
                        )
                    else:
                        input_values[col] = st.number_input(
                            col, min_value=minv, max_value=maxv, value=float(np.clip(median, minv, maxv))
                        )
                else:
                    choices = sorted(series.dropna().astype(str).unique().tolist())
                    if choices:
                        mode = str(series.mode().iloc[0]) if not series.mode().empty else choices[0]
                        default_idx = choices.index(mode) if mode in choices else 0
                        input_values[col] = st.selectbox(col, choices, index=default_idx)

        if st.button("Run demonstration prediction", type="primary"):
            row = pd.DataFrame([{c: input_values.get(c, raw[c].mode().iloc[0] if c in raw.columns and not raw[c].mode().empty else np.nan)
                                 for c in feature_columns}])
            estimator = model.named_steps["model"]
            attrition_indexes = np.flatnonzero(estimator.classes_ == 1)
            if len(attrition_indexes) != 1:
                st.error(
                    "The saved model does not expose a unique attrition class label of 1."
                )
            else:
                probabilities = model.predict_proba(row)[0]
                probability = float(probabilities[int(attrition_indexes[0])])
                predicted_class = estimator.classes_[int(np.argmax(probabilities))]
                predicted_label = "Left" if predicted_class == 1 else "Stayed"
                st.metric("Model-estimated attrition probability", f"{probability:.1%}")
                st.write(f"Predicted class at the model's default threshold: **{predicted_label}**")
            st.caption(
                "This number is a model output, not a reliable personal forecast. "
                "It should not be used to label, rank, or make decisions about any employee."
            )

        st.markdown("### Explainable AI")
        st.caption(
            "Global SHAP describes patterns across unique profiles in the current filtered view. "
            "Local SHAP and LIME explain this model's output for one selected dataset profile."
        )
        st.warning(
            "Explanations describe model behavior, not causes or employee potential. "
            "The dataset is synthetic and resampled; do not use results for employment decisions."
        )
        global_tab, individual_tab = st.tabs(
            ["Global SHAP", "Individual Profile: SHAP + LIME"]
        )
        reference_data = data.reindex(columns=feature_columns).drop_duplicates()
        view_signature = (
            tuple(selected_departments),
            selected_year_range,
            len(data),
        )
        model_version = (
            MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0,
            FEATURES_PATH.stat().st_mtime if FEATURES_PATH.exists() else 0,
        )

        with global_tab:
            st.markdown("**Global SHAP feature importance and summary**")
            st.caption(
                "The SHAP TreeExplainer explains the fitted Random Forest's attrition-class "
                "probability after the saved preprocessing pipeline transforms its inputs. "
                "Its baseline follows the fitted trees' path distribution."
            )
            if st.button("Generate global SHAP plots", key="generate_global_shap"):
                with st.spinner("Calculating global SHAP values for unique filtered profiles..."):
                    try:
                        st.session_state["careerpath_global_shap"] = cached_global_shap(
                            model,
                            reference_data,
                            feature_columns,
                            model_version,
                        )
                        st.session_state["careerpath_global_shap_view"] = view_signature
                    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
                        st.error(f"Could not calculate global SHAP explanations: {exc}")

            global_result = st.session_state.get("careerpath_global_shap")
            global_result_is_current = (
                global_result is not None
                and st.session_state.get("careerpath_global_shap_view") == view_signature
            )
            if global_result_is_current:
                importance = global_result["importance"]
                top_importance = importance.head(20).sort_values(
                    "Mean absolute SHAP value"
                )
                fig = px.bar(
                    top_importance,
                    x="Mean absolute SHAP value",
                    y="Feature",
                    orientation="h",
                    title="Global SHAP feature importance",
                    color="Mean absolute SHAP value",
                    color_continuous_scale=["#1e3a8a", "#38bdf8"],
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b1120",
                    plot_bgcolor="#111827",
                    showlegend=False,
                    xaxis_title="Mean absolute contribution to attrition probability",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**SHAP summary / beeswarm**")
                st.image(
                    global_result["summary_plot"],
                    caption=(
                        f"Each point is a unique profile ({global_result['explained_rows']} "
                        "explained); colors represent transformed feature values."
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "Positive SHAP values move the model output toward attrition; negative "
                    "values move it away. Mean absolute SHAP measures contribution magnitude, "
                    "not causation. Feature values are shown after saved scaling/one-hot encoding."
                )

        with individual_tab:
            st.markdown("**Choose a filtered employee profile**")
            profile_records = data.drop_duplicates(
                subset=["EmployeeNumber"]
            ) if "EmployeeNumber" in data.columns else data
            if "EmployeeNumber" in profile_records.columns:
                profile_ids = profile_records["EmployeeNumber"].astype(str).tolist()
                selected_profile_id = st.selectbox(
                    "Dataset profile (EmployeeNumber)",
                    options=profile_ids,
                    key="xai_employee_number",
                )
                profile = profile_records.loc[
                    profile_records["EmployeeNumber"].astype(str)
                    == selected_profile_id
                ].iloc[[0]].reindex(columns=feature_columns)
            else:
                profile_indexes = list(range(len(profile_records)))
                selected_index = st.selectbox(
                    "Filtered dataset row",
                    options=profile_indexes,
                    key="xai_profile_index",
                )
                selected_profile_id = f"row_{selected_index}"
                profile = profile_records.iloc[[selected_index]].reindex(
                    columns=feature_columns
                )

            if st.button(
                "Explain selected profile with SHAP and LIME",
                key="generate_local_xai",
                type="primary",
            ):
                with st.spinner("Generating individual SHAP and LIME explanations..."):
                    try:
                        local_shap = cached_profile_shap(
                            model,
                            profile,
                            feature_columns,
                            selected_profile_id,
                            model_version,
                        )
                        local_lime = cached_profile_lime(
                            model,
                            reference_data,
                            profile,
                            feature_columns,
                            selected_profile_id,
                            model_version,
                        )
                        st.session_state["careerpath_local_xai"] = {
                            "profile_id": selected_profile_id,
                            "view_signature": view_signature,
                            "shap": local_shap,
                            "lime": local_lime,
                        }
                    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
                        st.error(f"Could not calculate local explanations: {exc}")

            local_result = st.session_state.get("careerpath_local_xai")
            if (
                local_result
                and local_result["profile_id"] == selected_profile_id
                and local_result["view_signature"] == view_signature
            ):
                shap_result = local_result["shap"]
                lime_result = local_result["lime"]
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric(
                    "Model-estimated attrition probability",
                    f"{shap_result['attrition_probability']:.1%}",
                )
                metric_col2.metric(
                    "Predicted class",
                    "Left" if shap_result["predicted_class"] == "1" else "Stayed",
                )
                st.markdown("**SHAP — features moving attrition probability**")
                st.image(shap_result["plot"], use_container_width=True)
                shap_contributions = shap_result["contributions"].head(10).sort_values(
                    "SHAP contribution to attrition probability"
                )
                shap_fig = px.bar(
                    shap_contributions,
                    x="SHAP contribution to attrition probability",
                    y="Feature",
                    orientation="h",
                    color="SHAP contribution to attrition probability",
                    color_continuous_scale=["#38bdf8", "#64748b", "#f97316"],
                    title="Largest individual SHAP contributions",
                )
                shap_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b1120",
                    plot_bgcolor="#111827",
                    showlegend=False,
                )
                st.plotly_chart(shap_fig, use_container_width=True)
                st.caption(
                    "SHAP contributions are in probability units. Positive values increase "
                    "the model's attrition probability and negative values decrease it. "
                    "Feature values in the waterfall use the saved transformed representation."
                )

                st.markdown("**LIME — factors supporting or opposing the predicted class**")
                st.image(lime_result["plot"], use_container_width=True)
                st.dataframe(
                    lime_result["contributions"].style.format(
                        {"Contribution to predicted class": "{:.4f}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    f"LIME estimated {lime_result['predicted_class_probability']:.1%} probability "
                    "for the predicted class using a local surrogate. Positive bars support that "
                    "class; negative bars oppose it. SHAP and LIME use different methods."
                )

elif page == "Recommendations & Limitations":
    st.subheader("Recommendations & Limitations")
    st.caption(
        "Observations and follow-ups below reflect the current department and tenure filters."
    )

    if data.empty:
        st.info("No employee records match the current filters.")
    else:
        attrition_labels = data["Attrition"].astype("string").str.strip().str.lower()
        valid_attrition = attrition_labels.isin(["yes", "no"])
        metrics = st.columns(2)
        metrics[0].metric("Employees in filtered view", f"{len(data):,}")
        if valid_attrition.any():
            observed_attrition = attrition_labels[valid_attrition].eq("yes").mean() * 100
            metrics[1].metric("Observed attrition in view", f"{observed_attrition:.1f}%")
        else:
            metrics[1].metric("Observed attrition in view", "Unavailable")

        st.markdown("### Data-informed observations")
        if valid_attrition.any() and "Department" in data.columns:
            department_rates = pd.DataFrame({
                "Department": data.loc[valid_attrition, "Department"],
                "Attrition": attrition_labels[valid_attrition].eq("yes").astype(float),
            }).groupby("Department", as_index=False).agg(
                Employees=("Attrition", "size"),
                ObservedAttritionRate=("Attrition", "mean"),
            )
            department_rates["ObservedAttritionRate"] *= 100
            department_rates = department_rates.sort_values(
                "ObservedAttritionRate", ascending=False
            )
            st.dataframe(
                department_rates.round(1),
                use_container_width=True,
                hide_index=True,
            )
            highest_rate = department_rates.iloc[0]
            st.write(
                f"- **{highest_rate['Department']}** has the highest observed attrition "
                f"rate in this filtered view ({highest_rate['ObservedAttritionRate']:.1f}% "
                f"across {int(highest_rate['Employees'])} records)."
            )

        if "TrainingTimesLastYear" in data.columns and "Department" in data.columns:
            training = data.assign(
                TrainingTimesLastYear=pd.to_numeric(
                    data["TrainingTimesLastYear"], errors="coerce"
                )
            ).groupby("Department", as_index=False).agg(
                AverageTrainingSessions=("TrainingTimesLastYear", "mean")
            ).dropna(subset=["AverageTrainingSessions"])
            if len(training) > 1:
                lowest_training = training.loc[
                    training["AverageTrainingSessions"].idxmin()
                ]
                st.write(
                    f"- **{lowest_training['Department']}** has the lowest average "
                    f"reported training frequency ({lowest_training['AverageTrainingSessions']:.1f} "
                    "sessions in the previous year)."
                )

        st.markdown("### Actionable follow-ups")
        st.write(
            "- Review department-level patterns with HR and department leads before "
            "changing workforce plans."
        )
        st.write(
            "- Check whether training access and role-specific development support are "
            "consistent across departments."
        )
        st.write(
            "- Use voluntary feedback and longitudinal records to investigate retention "
            "patterns; do not infer causes from these descriptive associations."
        )

    st.markdown("### Limitations")
    st.write(
        "- This is a static snapshot, not a timeline of employee histories; it cannot "
        "establish actual promotion sequences or outcomes."
    )
    st.write(
        "- Resampled or repeated employee profiles are not independent observations, so "
        "model metrics and group patterns are educational signals, not population estimates."
    )
    st.write(
        "- Attrition predictions and explanations describe model behavior and must not "
        "be used to make employment decisions about individuals."
    )

elif page == "Career Assistant":
    st.title("✧ Career Assistant")
    st.caption(
        "Rule-based answers from the dataset. Results reflect the active department and "
        "company-tenure filters; this is not a live AI chatbot."
    )
    assistant_questions = [
        "What is the attrition rate?",
        "Which department has the most employees?",
        "Which department has the highest attrition rate?",
        "How does attrition vary by job level?",
        "What is the average monthly income?",
        "Which job level has the highest average income?",
        "How does training frequency compare by department?",
        "Suggest training improvements",
        "What are the career progression indicators?",
    ]
    if "JobRole" in data.columns:
        assistant_questions.append("Which job role has the longest average time in role?")
    question = st.selectbox("Choose a question", assistant_questions)

    if st.button("Get insights", type="primary"):
        if data.empty:
            st.warning("No employee records match the current filters.")
        elif question == "What is the attrition rate?":
            attrition = data["Attrition"].astype(str).str.lower().str.strip()
            valid = attrition.isin(["yes", "no"])
            if valid.sum():
                rate = attrition[valid].eq("yes").mean() * 100
                st.metric("Attrition rate", f"{rate:.1f}%")
                st.write(
                    "This is the percentage of valid filtered records "
                    "marked as having left."
                )
            else:
                st.info("No valid attrition records are available.")

        elif question == "Which department has the most employees?":
            counts = data["Department"].value_counts()
            if not counts.empty:
                st.metric("Largest department", counts.index[0])
                st.write(f"Employees in this department: {counts.iloc[0]}")

        elif question == "Which department has the highest attrition rate?":
            attrition = data["Attrition"].astype(str).str.strip().str.lower()
            valid = data.loc[attrition.isin(["yes", "no"])].copy()
            valid["LeftFlag"] = (
                valid["Attrition"].astype(str).str.strip().str.lower().eq("yes")
            )
            rates = valid.groupby("Department", as_index=False).agg(
                Employees=("LeftFlag", "size"),
                EmployeesLeft=("LeftFlag", "sum"),
                AttritionRate=("LeftFlag", "mean")
            )
            if not rates.empty:
                rates["AttritionRate"] *= 100
                highest = rates.loc[rates["AttritionRate"].idxmax()]
                st.metric("Highest observed rate", f"{highest['AttritionRate']:.1f}%")
                st.write(
                    f"{highest['Department']} has {int(highest['EmployeesLeft'])} records marked "
                    f"left out of {int(highest['Employees'])} valid records."
                )
                st.dataframe(
                    rates.sort_values("AttritionRate", ascending=False).round(2),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No valid attrition labels are available.")

        elif question == "How does attrition vary by job level?":
            attrition = data["Attrition"].astype(str).str.strip().str.lower()
            valid = data.loc[attrition.isin(["yes", "no"])].copy()
            valid["LeftFlag"] = (
                valid["Attrition"].astype(str).str.strip().str.lower().eq("yes")
            )
            level_rates = valid.groupby("JobLevel", as_index=False).agg(
                Employees=("LeftFlag", "size"),
                EmployeesLeft=("LeftFlag", "sum"),
                AttritionRate=("LeftFlag", "mean")
            )
            if not level_rates.empty:
                level_rates["AttritionRate"] *= 100
                st.dataframe(
                    level_rates.sort_values("JobLevel").round(2),
                    use_container_width=True,
                    hide_index=True
                )
                st.caption("Rates are descriptive and may be affected by small group sizes.")
            else:
                st.info("No valid attrition labels are available.")

        elif question == "What is the average monthly income?":
            income = pd.to_numeric(
                data["MonthlyIncome"], errors="coerce"
            ).dropna()
            if not income.empty:
                st.metric("Average monthly income", f"{income.mean():,.2f}")
                st.write(f"Based on {len(income)} valid income records.")
            else:
                st.info("No valid income values are available.")

        elif question == "Which job level has the highest average income?":
            income_by_level = data.assign(
                MonthlyIncome=pd.to_numeric(data["MonthlyIncome"], errors="coerce")
            ).groupby("JobLevel", as_index=False).agg(
                AverageIncome=("MonthlyIncome", "mean"),
                Employees=("MonthlyIncome", "count")
            ).dropna(subset=["AverageIncome"])
            if not income_by_level.empty:
                highest = income_by_level.loc[income_by_level["AverageIncome"].idxmax()]
                st.metric(
                    "Highest average income",
                    f"Job level {highest['JobLevel']}: {highest['AverageIncome']:,.0f}"
                )
                st.dataframe(
                    income_by_level.sort_values("JobLevel").round(2),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No valid income values are available.")

        elif question == "How does training frequency compare by department?":
            training_by_department = data.assign(
                TrainingTimesLastYear=pd.to_numeric(
                    data["TrainingTimesLastYear"], errors="coerce"
                )
            ).groupby("Department", as_index=False).agg(
                Employees=("TrainingTimesLastYear", "count"),
                AverageSessions=("TrainingTimesLastYear", "mean")
            ).dropna(subset=["AverageSessions"]).sort_values(
                "AverageSessions", ascending=False
            )
            if not training_by_department.empty:
                st.dataframe(
                    training_by_department.round(2),
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(
                    "Training counts are descriptive; they do not measure training quality or impact."
                )
            else:
                st.info("No valid training-frequency values are available.")

        elif question == "Suggest training improvements":
            if "TrainingTimesLastYear" in data.columns:
                training = pd.to_numeric(
                    data["TrainingTimesLastYear"], errors="coerce"
                ).dropna()
                if not training.empty:
                    st.metric(
                        "Average training sessions",
                        f"{training.mean():.2f}",
                    )
            st.markdown(
                """
                **Suggested actions**
                - Review access to learning opportunities across departments.
                - Identify skills required for future roles.
                - Discuss development goals with employees.
                - Evaluate training outcomes before recommending changes.
                """
            )

        elif question == "Which job role has the longest average time in role?":
            role_time = data.assign(
                YearsInCurrentRole=pd.to_numeric(
                    data["YearsInCurrentRole"], errors="coerce"
                )
            ).groupby("JobRole", as_index=False).agg(
                Employees=("YearsInCurrentRole", "count"),
                AverageYearsInRole=("YearsInCurrentRole", "mean")
            ).dropna(subset=["AverageYearsInRole"])
            if not role_time.empty:
                longest = role_time.loc[role_time["AverageYearsInRole"].idxmax()]
                st.metric(
                    "Longest average time in role",
                    f"{longest['JobRole']}: {longest['AverageYearsInRole']:.1f} years"
                )
                st.dataframe(
                    role_time.sort_values("AverageYearsInRole", ascending=False).round(2),
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(
                    "Time in role is a snapshot indicator, not evidence of promotion readiness."
                )
            else:
                st.info("No valid job-role tenure values are available.")

        elif question == "What are the career progression indicators?":
            cols = [
                "TotalWorkingYears",
                "YearsAtCompany",
                "YearsInCurrentRole",
                "YearsSinceLastPromotion",
                "TrainingTimesLastYear",
            ]
            available = [c for c in cols if c in data.columns]
            if available:
                st.dataframe(
                    data[available].describe().T.round(2),
                    use_container_width=True,
                )
            st.caption(
                "These are descriptive indicators, not proof of promotion "
                "readiness or predictions of individual career success."
            )