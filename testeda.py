import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Jobs Dashboard",
    layout="wide"
)

# --------------------------------------------------
# CSS
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0a0a0a;
        color: #f5f5f5;
    }

    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #222222;
    }

    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #f5f5f5 !important;
    }

    .title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #bdbdbd !important;
        font-size: 16px;
        margin-top: -8px;
        margin-bottom: 24px;
    }

    .metric-box {
        background-color: #141414;
        border: 1px solid #262626;
        border-radius: 18px;
        padding: 18px;
        text-align: center;
    }

    .metric-label {
        color: #a3a3a3 !important;
        font-size: 14px;
    }

    .metric-number {
        font-size: 30px;
        font-weight: 700;
        color: #ffffff !important;
    }

    .section-block {
        background-color: #111111;
        border: 1px solid #222222;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .insight-card {
        background-color: #141414;
        border: 1px solid #262626;
        border-left: 4px solid #6ee7b7;
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 12px;
        color: #e5e5e5 !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #222222;
        border-radius: 14px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ai_jobs_market_2025_2026.csv")

df = load_data()
cleaned_df = df.drop(columns=["job_id", "salary_min_usd", "salary_max_usd"], errors="ignore")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)

    st.markdown("## Navigation")
    section = st.radio(
        "",
        ["Overview", "Understanding The Data", "Cleaning The Data", "Visualizations", "Insights", "Model"]
    )

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<p class="title">AI Jobs Market Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">EDA and simple classification model for AI job market trends.</p>', unsafe_allow_html=True)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
if section == "Overview":
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Rows</div>
            <div class="metric-number">{df.shape[0]}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Columns</div>
            <div class="metric-number">{df.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Job Titles</div>
            <div class="metric-number">{df['job_title'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Countries</div>
            <div class="metric-number">{df['country'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# UNDERSTANDING THE DATA
# --------------------------------------------------
elif section == "Understanding The Data":
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    st.subheader("Understanding The Data")

    tab1, tab2, tab3 = st.tabs(["Column Description", "Unique Values", "Preview"])

    with tab1:
        st.markdown(""" 
**job_title**: Specific title of the job position.  
**job_category**: Broader category grouping similar roles.  
**experience_level**: Required experience level.  
**years_of_experience**: Number of years of experience required.  
**education_required**: Minimum educational qualification needed.  
**annual_salary_usd**: Estimated annual salary in USD.  
**city**: City where the job is located.  
**country**: Country where the job is offered.  
**required_skills**: Main required skills for the job.     
**is_llm_role**: Whether the role is related to LLMs.  
**salary_tier**: Salary group/category.
        """)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Job Titles**")
            st.dataframe(pd.DataFrame(sorted(df["job_title"].dropna().unique()), columns=["job_title"]), use_container_width=True)
            st.write("**Job Categories**")
            st.dataframe(pd.DataFrame(sorted(df["job_category"].dropna().unique()), columns=["job_category"]), use_container_width=True)
        with col2:
            st.write("**Countries**")
            st.dataframe(pd.DataFrame(sorted(df["country"].dropna().unique()), columns=["country"]), use_container_width=True)
            st.write("**Cities**")
            st.dataframe(pd.DataFrame(sorted(df["city"].dropna().unique()), columns=["city"]), use_container_width=True)

    with tab3:
        st.write(f"**Shape:** {df.shape}")
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# CLEANING
# --------------------------------------------------
elif section == "Cleaning The Data":
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    st.subheader("Cleaning The Data")

    st.markdown("""
- Checked for missing values.  
- Found no missing values in the dataset.  
- Checked for duplicate rows.  
- Suggested dropping columns that are less useful for EDA:
  - `job_id`
  - `salary_min_usd`
  - `salary_max_usd`
- Kept the most relevant columns for analysis and modeling.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Missing Values**")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        st.dataframe(missing_df, use_container_width=True)

    with col2:
        st.write("**Duplicate Rows**")
        st.write(df.duplicated().sum())

    st.write("**Columns After Suggested Cleaning**")
    st.write(cleaned_df.columns.tolist())

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------
elif section == "Visualizations":
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    st.subheader("Visualizations")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Understanding The Job Market",
        "Salary Analysis",
        "Experience Analysis",
        "Skills Analysis"
    ])

    sns.set_style("darkgrid")
    plt.rcParams["figure.facecolor"] = "#0a0a0a"
    plt.rcParams["axes.facecolor"] = "#111111"

    with tab1:
        job_cat_counts = df["job_category"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=job_cat_counts.values, y=job_cat_counts.index, palette="magma", ax=ax)
        ax.set_title("Job Category Distribution", color="white")
        ax.set_xlabel("Count", color="white")
        ax.set_ylabel("Job Category", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

        top_titles = df["job_title"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_titles.values, y=top_titles.index, palette="viridis", ax=ax)
        ax.set_title("Top 10 Job Titles", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df["annual_salary_usd"], bins=30, kde=True, color="#60a5fa", ax=ax)
        ax.set_title("Salary Distribution", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

        avg_salary = df.groupby("job_title")["annual_salary_usd"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=avg_salary.values, y=avg_salary.index, palette="rocket", ax=ax)
        ax.set_title("Top 10 Highest Paying Job Titles", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df, x="experience_level", palette="crest", ax=ax)
        ax.set_title("Experience Level Distribution", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x="years_of_experience", y="annual_salary_usd", data=df, palette="coolwarm", ax=ax)
        ax.set_title("Salary by Years of Experience", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

    with tab4:

        llm_counts = df["is_llm_role"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(llm_counts, labels=["Non-LLM", "LLM"], autopct="%1.1f%%", colors=["#818cf8", "#f472b6"], textprops={'color': 'white'})
        ax.set_title("LLM vs Non-LLM Roles", color="white")
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# INSIGHTS
# --------------------------------------------------
elif section == "Insights":
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    st.subheader("Insights")

    insights = [
        "Dominant Job Roles: AI Engineering is the leading category in the market, accounting for the highest percentage of job postings (over 70%). Other significant roles include Data Science, Governance, and Robotics.",
        "Salary Benchmarks: The average annual salary is approximately  149,157USD,withawiderangereachingupto 250,000 USD. Roles like AI Solutions Architect and Machine Learning Engineer are among the highest-paid positions.",
        "Experience vs. Compensation: There is a clear positive correlation between years of experience and salary. As seniority increases, the salary distribution shifts significantly higher, indicating a high premium for experienced AI talent.",
        "Geographical Hubs: The United States remains the primary hub for AI opportunities, followed by the United Kingdom and China. These three countries dominate the global job count in this sector.",
        "Educational Requirements: Most roles require specific educational backgrounds, and there is a noticeable (AI Salary Premium) for candidates possessing advanced technical skills, particularly in Large Language Models (LLM).",
        "Market Growth: The data shows a strong year-over-year (YoY) growth in demand, particularly for technical roles that involve direct AI implementation and architecture.",
    ]

    for ins in insights:
        st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
elif section == "Model":
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    st.subheader("Classification Model")

    st.write("**Goal:** Predict whether a job is LLM-related.")

    features = [
        "job_category",
        "experience_level",
        "years_of_experience",
        "education_required",
        "remote_work",
        "company_size",
        "industry",
        "country"
    ]

    model_df = df[features + ["is_llm_role"]].copy()

    for col in model_df.columns:
        if model_df[col].dtype == "object":
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col])

    X = model_df[features]
    y = model_df["is_llm_role"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")

    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix", color="white")
    ax.tick_params(colors="white")
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)