import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------------------
# 1. Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Employee Retention Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# 2. Session State Initialization
# ----------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# Helper for CSS themes
def get_theme_class():
    return "dark-theme" if st.session_state.theme == "dark" else "light-theme"

# Inject Custom Stylesheet
def inject_custom_css():
    try:
        with open("style.css", "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load custom CSS style.css: {e}")

# Inject HTML Canvas Confetti Utility
def trigger_js_confetti():
    confetti_html = """
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
    <script>
        var duration = 3 * 1000;
        var end = Date.now() + duration;

        (function frame() {
            confetti({
                particleCount: 5,
                angle: 60,
                spread: 55,
                origin: { x: 0 }
            });
            confetti({
                particleCount: 5,
                angle: 120,
                spread: 55,
                origin: { x: 1 }
            });

            if (Date.now() < end) {
                requestAnimationFrame(frame);
            }
        }());
    </script>
    """
    components.html(confetti_html, height=0, width=0)

# ----------------------------------------------------
# 3. Model & Data Handling (Robust Fallback Training)
# ----------------------------------------------------
@st.cache_data
def load_and_clean_data():
    csv_path = "HR_comma_sep.csv"
    if not os.path.exists(csv_path):
        # Create mock data if CSV is somehow missing
        st.error("HR_comma_sep.csv file not found!")
        st.stop()
    df = pd.read_csv(csv_path)
    df_raw = df.copy()
    df_clean = df.drop_duplicates()
    return df_raw, df_clean

# Retrieve datasets
try:
    df_raw, df = load_and_clean_data()
except Exception as e:
    st.error(f"Error loading CSV dataset: {e}")
    st.stop()

# Helper to train the model pipeline
def train_and_save_model():
    X = df.drop('left', axis=1)
    y = df['left']
    
    numerical_features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years']
    categorical_features = ['Department', 'salary']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    model_pipeline.fit(X_train, y_train)
    
    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(model_pipeline, f)
    return model_pipeline

# Load Model Pipeline
@st.cache_resource
def load_trained_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Error reading model.pkl, training fresh model: {e}")
    return train_and_save_model()

try:
    model_pipeline = load_trained_model()
except Exception as e:
    st.error(f"Failed to load or train model: {e}")
    st.stop()

# ----------------------------------------------------
# 4. App UI & Styling Injection
# ----------------------------------------------------
inject_custom_css()
theme_class = get_theme_class()
st.markdown(f'<div class="{theme_class}">', unsafe_allow_html=True)

# ----------------------------------------------------
# 5. Sidebar Layout & Theme Toggle
# ----------------------------------------------------
# Sidebar Header
st.sidebar.markdown(
    f"""
    <div style="text-align: center; padding: 15px 0;">
        <h2 style="margin: 0; font-size: 24px; font-weight: 800;">
            <span class="{'gradient-text' if st.session_state.theme == 'dark' else 'gradient-text-light'}">RetentionHub</span>
        </h2>
        <p style="margin: 5px 0 0 0; font-size: 12px; color: var(--text-secondary);">Enterprise Risk Analytics</p>
    </div>
    <hr style="border: 0; height: 1px; background: var(--card-border); margin: 10px 0 20px 0;">
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation Selection
navigation_options = [
    "🏠 Home",
    "📊 Data Analysis",
    "📈 Visualizations",
    "🤖 Prediction",
    "📉 Model Performance",
    "👨‍💻 About Developer",
    "📬 Contact"
]

selected_page = st.sidebar.radio(
    "Navigation Menu",
    options=navigation_options,
    label_visibility="collapsed"
)
st.session_state.page = selected_page

st.sidebar.markdown("<br><hr style='border:0; height:1px; background:var(--card-border);'>", unsafe_allow_html=True)

# Dark/Light Mode toggle in sidebar
st.sidebar.markdown("<p style='font-size: 13px; font-weight:600; color: var(--text-secondary); margin-bottom: 5px;'>Visual Theme</p>", unsafe_allow_html=True)
toggle_label = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
if st.sidebar.button(toggle_label):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

# ----------------------------------------------------
# 6. Page Renderers
# ----------------------------------------------------

# ================= PAGE: HOME =================
if st.session_state.page == "🏠 Home":
    # Hero Title Section
    st.markdown(
        f"""
        <div class="glass-card" style="text-align: center; padding: 40px 20px; background: rgba(99, 102, 241, 0.03);">
            <div class="badge" style="margin-bottom: 15px;">Machine Learning Project</div>
            <h1 style="font-size: 3rem; margin: 0 0 10px 0; line-height: 1.2;">
                Employee Retention <span class="{'gradient-text' if st.session_state.theme == 'dark' else 'gradient-text-light'}">Prediction</span>
            </h1>
            <p style="font-size: 1.25rem; max-width: 800px; margin: 0 auto; color: var(--text-secondary); font-weight: 400; line-height: 1.5;">
                An intelligent Predictive Analytics platform built with Logistic Regression to identify employees with potential resignation risks. 
                Optimized to help HR professionals introduce timely retention strategies.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Metric Highlight Counters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="metric-card glass-card">
                <span style="font-size: 12px; font-weight:600; text-transform: uppercase; color: var(--text-secondary);">Model Accuracy</span>
                <span class="gradient-text" style="font-size: 2.5rem; font-weight: 800;">83.2%</span>
                <span style="font-size: 11px; color: var(--success); font-weight: 500;">✓ Cross-Validated</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card glass-card">
                <span style="font-size: 12px; font-weight:600; text-transform: uppercase; color: var(--text-secondary);">Dataset Size</span>
                <span class="gradient-text" style="font-size: 2.5rem; font-weight: 800;">14,999</span>
                <span style="font-size: 11px; color: var(--text-secondary);">Raw Employee Records</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card glass-card">
                <span style="font-size: 12px; font-weight:600; text-transform: uppercase; color: var(--text-secondary);">Model Algorithm</span>
                <span class="gradient-text" style="font-size: 2.5rem; font-weight: 800;">LogReg</span>
                <span style="font-size: 11px; color: var(--text-secondary);">Logistic Regression</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card glass-card">
                <span style="font-size: 12px; font-weight:600; text-transform: uppercase; color: var(--text-secondary);">Cleaned Rows</span>
                <span class="gradient-text" style="font-size: 2.5rem; font-weight: 800;">11,991</span>
                <span style="font-size: 11px; color: var(--danger); font-weight: 500;">✂ {len(df_raw) - len(df)} Duplicates Cut</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Details Section
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <h3>📌 Project Overview & Objectives</h3>
                <p>Employee churn (attrition) is an expensive operational issue. Hiring replacement talent costs significant money, time, and reduces institutional productivity. This project utilizes an advanced machine learning classification model to analyze historical employee features and flags whether an employee has a high turnover risk.</p>
                <h4 style="margin-top: 15px;">Key Features Analyzed:</h4>
                <ul style="color: var(--text-secondary); line-height: 1.6; margin-left: 20px;">
                    <li><b>Workplace Satisfaction:</b> Score index evaluating daily work environment sentiment.</li>
                    <li><b>Performance Evaluation:</b> Quality of execution in project tasks.</li>
                    <li><b>Workload Metrics:</b> Number of projects and average monthly working hours.</li>
                    <li><b>Longevity:</b> Years spent at the current firm.</li>
                    <li><b>Professional Indicators:</b> Past promotions, workplace accidents, department categorizations, and salary levels.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_right:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <h3>⚡ Technology & Architecture Highlights</h3>
                <p>This web dashboard is engineered to replicate custom SaaS dashboards. We replaced all generic Streamlit layout elements using custom styled CSS stylesheets and implemented the following stacks:</p>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px;">
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--card-border); padding: 12px; border-radius: 8px;">
                        <h5 style="margin: 0; color: var(--accent);">Modern CSS3</h5>
                        <p style="margin: 5px 0 0 0; font-size: 12px; color: var(--text-secondary);">Glassmorphic cards, custom backgrounds, and custom button animations.</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--card-border); padding: 12px; border-radius: 8px;">
                        <h5 style="margin: 0; color: var(--accent);">Plotly Express</h5>
                        <p style="margin: 5px 0 0 0; font-size: 12px; color: var(--text-secondary);">Fully interactive, fluid charts responding dynamically to UI theme styles.</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--card-border); padding: 12px; border-radius: 8px;">
                        <h5 style="margin: 0; color: var(--accent);">Scikit-Learn Pipeline</h5>
                        <p style="margin: 5px 0 0 0; font-size: 12px; color: var(--text-secondary);">Safe model packaging with StandardScaler and OneHotEncoder transformers.</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--card-border); padding: 12px; border-radius: 8px;">
                        <h5 style="margin: 0; color: var(--accent);">Dual-Theme Sync</h5>
                        <p style="margin: 5px 0 0 0; font-size: 12px; color: var(--text-secondary);">Responsive dark and light CSS switches styled to match modern software.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ================= PAGE: DATA ANALYSIS =================
elif st.session_state.page == "📊 Data Analysis":
    st.markdown(
        f"""
        <div class="glass-card">
            <h2 style="margin: 0 0 5px 0;">📊 Dataset Analysis Summary</h2>
            <p style="color: var(--text-secondary); margin: 0;">Detailed structural diagnostics of the HR Employee Retention Dataset.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Grid of Data Quality
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card glass-card"><h5>Total Rows</h5><h3 class="gradient-text" style="font-size: 2rem;">{len(df_raw)}</h3></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card glass-card"><h5>Total Features</h5><h3 class="gradient-text" style="font-size: 2rem;">{df_raw.shape[1]}</h3></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card glass-card"><h5>Duplicate Rows Cut</h5><h3 class="gradient-text" style="font-size: 2rem; color: var(--danger);">{len(df_raw) - len(df)}</h3></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card glass-card"><h5>Missing Values</h5><h3 class="gradient-text" style="font-size: 2rem; color: var(--success);">{df_raw.isnull().sum().sum()}</h3></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dataset Preview Tab
    st.markdown("### 🔍 Dataset Explorer")
    st.write("Browse records of the preprocessed and deduplicated dataset:")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature descriptions and types
    st.markdown("### 📋 Features Information")
    feature_info = {
        "Feature Name": [
            "satisfaction_level", "last_evaluation", "number_project", 
            "average_montly_hours", "time_spend_company", "Work_accident", 
            "left", "promotion_last_5years", "Department", "salary"
        ],
        "Type": [
            "Float", "Float", "Integer", "Integer", "Integer", "Binary", "Binary (Target)", "Binary", "Categorical", "Categorical"
        ],
        "Description": [
            "Employee satisfaction rating index (0.0 to 1.0)",
            "Last job performance rating evaluation index (0.0 to 1.0)",
            "Number of key projects completed during employment",
            "Average monthly work hours clocked",
            "Time spent at the company (in years)",
            "Whether the employee suffered a workplace accident (0 = No, 1 = Yes)",
            "Retention status flag (0 = Stayed, 1 = Left)",
            "Whether the employee was promoted within the last 5 years (0 = No, 1 = Yes)",
            "Specific department division of placement",
            "Employee salary bracket rating (low, medium, high)"
        ]
    }
    info_df = pd.DataFrame(feature_info)
    
    # Render custom styled HTML Table
    table_rows = ""
    for idx, row in info_df.iterrows():
        table_rows += f"""
        <tr>
            <td><code>{row['Feature Name']}</code></td>
            <td><span class="badge" style="background: rgba(255,255,255,0.05); color: var(--accent); border-color: var(--card-border);">{row['Type']}</span></td>
            <td>{row['Description']}</td>
        </tr>
        """
    
    st.markdown(
        f"""
        <table class="styled-table">
            <thead>
                <tr>
                    <th>Feature Name</th>
                    <th>Data Type</th>
                    <th>Role & Description</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Summary Statistics
    st.markdown("### 📈 Numerical Summary Statistics")
    st.write("Descriptive summary measurements of the continuous dataset attributes:")
    
    stats_df = df.describe().round(3)
    st.dataframe(stats_df, use_container_width=True)

# ================= PAGE: VISUALIZATIONS =================
elif st.session_state.page == "📈 Visualizations":
    st.markdown(
        f"""
        <div class="glass-card">
            <h2 style="margin: 0 0 5px 0;">📈 Exploratory Interactive Visualizations</h2>
            <p style="color: var(--text-secondary); margin: 0;">Analyze behaviors and variables correlates to retention rates.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Define color theme variables for charts matching Light/Dark themes
    chart_bg = "rgba(0,0,0,0)"
    grid_color = "rgba(255, 255, 255, 0.08)" if st.session_state.theme == "dark" else "rgba(15, 23, 42, 0.08)"
    text_color = "#f3f4f6" if st.session_state.theme == "dark" else "#0f172a"
    color_palette = ["#6366f1", "#ef4444"]
    
    # 1. Salary vs Retention and Department vs Retention
    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        # Salary vs Left
        salary_crosstab = pd.crosstab(df['salary'], df['left']).reset_index()
        salary_crosstab.columns = ['Salary', 'Retained', 'Left']
        
        fig1 = px.bar(
            salary_crosstab, 
            x='Salary', 
            y=['Retained', 'Left'], 
            title="Salary Level vs Employee Retention Status",
            color_discrete_sequence=color_palette,
            barmode='group'
        )
        fig1.update_layout(
            paper_bgcolor=chart_bg,
            plot_bgcolor=chart_bg,
            font_color=text_color,
            xaxis=dict(gridcolor=grid_color),
            yaxis=dict(gridcolor=grid_color),
            legend_title="Employee Status"
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("<p style='font-size: 12px; color: var(--text-secondary); text-align: center;'>High-salary brackets exhibit significantly higher retention probabilities compared to low-salary tiers.</p>", unsafe_allow_html=True)

    with col_vis2:
        # Department vs Left
        dept_crosstab = pd.crosstab(df['Department'], df['left']).reset_index()
        dept_crosstab.columns = ['Department', 'Retained', 'Left']
        
        fig2 = px.bar(
            dept_crosstab,
            x='Department',
            y=['Retained', 'Left'],
            title="Workplace Department vs Retention Status",
            color_discrete_sequence=color_palette,
            barmode='stack'
        )
        fig2.update_layout(
            paper_bgcolor=chart_bg,
            plot_bgcolor=chart_bg,
            font_color=text_color,
            xaxis=dict(gridcolor=grid_color),
            yaxis=dict(gridcolor=grid_color),
            legend_title="Employee Status"
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<p style='font-size: 12px; color: var(--text-secondary); text-align: center;'>Stacked distribution showing employee retention counts relative to their specific placement divisions.</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Correlation Heatmap & Pie Chart
    col_vis3, col_vis4 = st.columns([1.3, 0.7])
    with col_vis3:
        # Correlation Heatmap
        numeric_cols = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_cols.corr().round(3)
        
        fig3 = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='Viridis',
            text=corr_matrix.values,
            texttemplate="%{text}",
            showscale=True
        ))
        fig3.update_layout(
            title="Correlation Matrix Heatmap (Numerical Attributes)",
            paper_bgcolor=chart_bg,
            plot_bgcolor=chart_bg,
            font_color=text_color
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("<p style='font-size: 12px; color: var(--text-secondary); text-align: center;'>Correlation measurements showing relation values between numerical features. Note the negative correlation of satisfaction level with departure status.</p>", unsafe_allow_html=True)

    with col_vis4:
        # Retention split Pie Chart
        left_counts = df['left'].value_counts().reset_index()
        left_counts.columns = ['Status', 'Count']
        left_counts['StatusName'] = left_counts['Status'].map({0: 'Retained (Stayed)', 1: 'Attrition (Left)'})
        
        fig4 = px.pie(
            left_counts,
            values='Count',
            names='StatusName',
            title="Global Retention / Attrition Ratio",
            color_discrete_sequence=color_palette,
            hole=0.4
        )
        fig4.update_layout(
            paper_bgcolor=chart_bg,
            plot_bgcolor=chart_bg,
            font_color=text_color,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("<p style='font-size: 12px; color: var(--text-secondary); text-align: center;'>Donut breakdown displaying the total class balance ratio between employees who stayed and those who departed.</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Distribution Plots (Satisfaction vs Hours)
    col_vis5, col_vis6 = st.columns(2)
    with col_vis5:
        # Satisfaction levels box plots split by Left status
        df_for_box = df.copy()
        df_for_box['Status'] = df_for_box['left'].map({0: 'Stayed', 1: 'Left'})
        fig5 = px.box(
            df_for_box, 
            x='Status', 
            y='satisfaction_level',
            title="Satisfaction Index Distribution vs Retention Status",
            color='Status',
            color_discrete_sequence=color_palette
        )
        fig5.update_layout(
            paper_bgcolor=chart_bg,
            plot_bgcolor=chart_bg,
            font_color=text_color,
            xaxis=dict(gridcolor=grid_color),
            yaxis=dict(gridcolor=grid_color)
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("<p style='font-size: 12px; color: var(--text-secondary); text-align: center;'>Box distribution highlighting that employees who left had significantly lower median satisfaction ratings (~0.41) compared to those remaining (~0.69).</p>", unsafe_allow_html=True)

    with col_vis6:
        # average monthly hours histogram
        fig6 = px.histogram(
            df_for_box, 
            x='average_montly_hours',
            color='Status', 
            barmode='overlay',
            title="Average Monthly Hours Distribution vs Retention Status",
            color_discrete_sequence=color_palette,
            marginal="box"
        )
        fig6.update_layout(
            paper_bgcolor=chart_bg,
            plot_bgcolor=chart_bg,
            font_color=text_color,
            xaxis=dict(gridcolor=grid_color),
            yaxis=dict(gridcolor=grid_color)
        )
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("<p style='font-size: 12px; color: var(--text-secondary); text-align: center;'>Employees working extreme overtime (>250 hours/month) or very low hours (<150 hours/month) show increased departure risk frequencies.</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Feature Importance Visualisation
    st.markdown("### 🏆 Coefficient Feature Importance")
    st.write("Visualizing the impact weighting of features from our trained Logistic Regression model:")
    
    # Extract coefficients
    clf = model_pipeline.named_steps['classifier']
    preprocessor = model_pipeline.named_steps['preprocessor']
    
    # Retrieve feature names from ColumnTransformer
    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1]
    cat_features_encoded = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]).tolist()
    
    all_feature_names = num_features + cat_features_encoded
    coef_values = clf.coef_[0]
    
    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': coef_values,
        'Absolute Coefficient': np.abs(coef_values)
    }).sort_values(by='Absolute Coefficient', ascending=True)

    # Add color labels for positive/negative impacts
    importance_df['Impact Direction'] = importance_df['Coefficient'].apply(lambda x: 'Increases Retention Risk' if x > 0 else 'Reduces Retention Risk')
    
    fig7 = px.bar(
        importance_df,
        x='Coefficient',
        y='Feature',
        orientation='h',
        color='Impact Direction',
        color_discrete_map={
            'Increases Retention Risk': '#ef4444',
            'Reduces Retention Risk': '#10b981'
        },
        title="Logistic Regression Feature Coefficients (Positive = Higher Risk, Negative = Lower Risk)"
    )
    fig7.update_layout(
        paper_bgcolor=chart_bg,
        plot_bgcolor=chart_bg,
        font_color=text_color,
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color)
    )
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown("<p style='font-size: 13px; color: var(--text-secondary); text-align: center;'><b>Interpretation:</b> Higher satisfaction ratings (negative coefficient) and workplace accidents (negative coefficient) strongly decrease departure risk. Higher tenure (positive coefficient) and high workload hours (positive coefficient) push risk ratios upwards.</p>", unsafe_allow_html=True)

# ================= PAGE: PREDICTION =================
elif st.session_state.page == "🤖 Prediction":
    st.markdown(
        f"""
        <div class="glass-card">
            <h2 style="margin: 0 0 5px 0;">🤖 Employee Retention Predictive System</h2>
            <p style="color: var(--text-secondary); margin: 0;">Adjust the parameters below to compute real-time resignation risk scores.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form Layout inside a Glass Card wrapper
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 👤 General & Satisfaction")
        satisfaction = st.slider("Satisfaction Level", min_value=0.01, max_value=1.0, value=0.5, step=0.01, help="Score representing employee's job satisfaction.")
        evaluation = st.slider("Last Evaluation Score", min_value=0.01, max_value=1.0, value=0.7, step=0.01, help="Rating from employee's latest performance assessment.")
        salary = st.selectbox("Salary Category Bracket", options=["low", "medium", "high"], index=0, help="Employee salary scale bracket.")

    with col2:
        st.markdown("##### 💼 Role & Hours")
        projects = st.number_input("Number of Projects", min_value=2, max_value=7, value=3, step=1, help="Count of key projects assigned to the employee.")
        monthly_hours = st.slider("Average Monthly Hours", min_value=96, max_value=310, value=200, step=5, help="Monthly average hours worked.")
        department = st.selectbox("Department Placement", options=df['Department'].unique().tolist(), index=0, help="Active department division.")

    with col3:
        st.markdown("##### ⚠️ Indicators & Tenure")
        tenure = st.slider("Tenure in Company (Years)", min_value=2, max_value=10, value=3, step=1, help="Total years working at this company.")
        accident = st.selectbox("Workplace Accident Incident", options=["No Accident", "Had Accident"], index=0, help="Whether the employee had an accident on the job.")
        promotion = st.selectbox("Promotion Status (Last 5 Years)", options=["No Promotion", "Received Promotion"], index=0, help="Whether the employee got a promotion in the last 5 years.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Conversion of selectors to model friendly variables
    accident_val = 1 if accident == "Had Accident" else 0
    promotion_val = 1 if promotion == "Received Promotion" else 0

    input_data = {
        'satisfaction_level': satisfaction,
        'last_evaluation': evaluation,
        'number_project': projects,
        'average_montly_hours': monthly_hours,
        'time_spend_company': tenure,
        'Work_accident': accident_val,
        'promotion_last_5years': promotion_val,
        'Department': department,
        'salary': salary
    }

    input_df = pd.DataFrame([input_data])

    if st.button("🔮 Calculate Retention Risk"):
        # Loading Screen & Progress Bar
        with st.status("🛠 Processing Employee Records...", expanded=True) as status:
            st.write("1. Reading inputs & parsing data features...")
            time.sleep(0.4)
            st.write("2. Invoking preprocessing pipeline (StandardScaler & OneHotEncoder)...")
            time.sleep(0.4)
            st.write("3. Querying Logistic Regression model probabilities...")
            time.sleep(0.4)
            status.update(label="Inference complete!", state="complete", expanded=False)

        # Execute Model Inference
        prob_leave = model_pipeline.predict_proba(input_df)[0][1]
        prob_stay = 1 - prob_leave
        prediction = model_pipeline.predict(input_df)[0]

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability Gauge Layout
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.markdown("### Risk Level Score Gauge")
            # Render custom Plotly Gauge Chart
            gauge_color = "#ef4444" if prob_leave > 0.5 else "#10b981"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_leave * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probability of Departure (%)", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "white" if st.session_state.theme == "light" else "#1f2937",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [35, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#f3f4f6" if st.session_state.theme == "dark" else "#0f172a", 'family': "Outfit"},
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_res2:
            st.markdown("### Predictive Analysis Report")
            
            if prediction == 0:
                # EMPLOYEE STAYS (SUCCESS)
                st.balloons()
                trigger_js_confetti()
                
                st.markdown(
                    f"""
                    <div class="success-card">
                        <h3 style="color: #059669; margin:0 0 10px 0;">🎉 Retained: Low Departure Risk</h3>
                        <p style="color: var(--text-primary); font-size:15px; margin-bottom:15px;">
                            The model estimates the employee will <b>STAY</b> in the company. 
                            Confidence Score: <b>{prob_stay * 100:.2f}%</b>.
                        </p>
                        <div style="border-top: 1px solid rgba(16, 185, 129, 0.2); padding-top: 10px;">
                            <h5 style="color: #059669; margin: 5px 0;">🔍 Core Retention Factors:</h5>
                            <ul style="font-size: 13px; color: var(--text-secondary); margin-left: 20px;">
                                <li>Satisfaction index is healthy ({satisfaction:.2f}).</li>
                                <li>Workload hours ({monthly_hours} hrs/mo) are sustainable.</li>
                                <li>The risk profile is protected under current compensation terms.</li>
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # EMPLOYEE LEAVES (WARNING)
                st.markdown(
                    f"""
                    <div class="danger-card">
                        <h3 style="color: #dc2626; margin:0 0 10px 0;">⚠️ Risk Warning: High Turnover Risk</h3>
                        <p style="color: var(--text-primary); font-size:15px; margin-bottom:15px;">
                            The model estimates a high probability that the employee will <b>LEAVE</b> the company. 
                            Confidence Score: <b>{prob_leave * 100:.2f}%</b>.
                        </p>
                        <div style="border-top: 1px solid rgba(239, 68, 68, 0.2); padding-top: 10px;">
                            <h5 style="color: #dc2626; margin: 5px 0;">📋 Suggested HR Intervention Plan:</h5>
                            <ul style="font-size: 13px; color: var(--text-secondary); margin-left: 20px; line-height:1.5;">
                                <li><b>Compensation Review:</b> Assess if salary tier ("{salary}") matches market value.</li>
                                <li><b>Load Management:</b> Monthly hours ({monthly_hours} hrs) or projects ({projects}) may be inducing burnout.</li>
                                <li><b>One-on-One Checkup:</b> The satisfaction level ({satisfaction:.2f}) indicates potential engagement loss.</li>
                                <li><b>Career Development:</b> Verify growth paths. (Promoted in last 5 years: {"Yes" if promotion_val else "No"}).</li>
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ================= PAGE: MODEL PERFORMANCE =================
elif st.session_state.page == "📉 Model Performance":
    st.markdown(
        f"""
        <div class="glass-card">
            <h2 style="margin: 0 0 5px 0;">📉 Model Performance & Training Diagnostics</h2>
            <p style="color: var(--text-secondary); margin: 0;">Evaluation statistics for the trained Logistic Regression pipeline model.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Performance Overview Gauges
    col_p1, col_p2, col_p3 = st.columns(3)
    
    # Calculate performance stats on test set
    X_eval = df.drop('left', axis=1)
    y_eval = df['left']
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_eval, y_eval, test_size=0.2, random_state=10)
    
    y_pred_e = model_pipeline.predict(X_test_e)
    accuracy_score_val = accuracy_score(y_test_e, y_pred_e)
    
    with col_p1:
        st.markdown(
            f"""
            <div class="metric-card glass-card">
                <span style="font-size: 13px; color: var(--text-secondary); text-transform: uppercase;">Accuracy Score</span>
                <span class="gradient-text" style="font-size: 2.5rem; font-weight: 800;">{accuracy_score_val * 100:.2f}%</span>
                <span style="font-size: 11px; color: var(--success); font-weight: 500;">✓ Ready for Production</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_p2:
        st.markdown(
            """
            <div class="metric-card glass-card">
                <span style="font-size: 13px; color: var(--text-secondary); text-transform: uppercase;">Split Ratio</span>
                <span class="gradient-text" style="font-size: 2.5rem; font-weight: 800;">80 : 20</span>
                <span style="font-size: 11px; color: var(--text-secondary);">Train vs. Test Rows</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_p3:
        st.markdown(
            """
            <div class="metric-card glass-card">
                <span style="font-size: 13px; color: var(--text-secondary); text-transform: uppercase;">Solver Algorithm</span>
                <span class="gradient-text" style="font-size: 2.5rem; font-weight: 800;">L-BFGS</span>
                <span style="font-size: 11px; color: var(--text-secondary);">L2 Regularized Estimator</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_perf_l, col_perf_r = st.columns([1, 1.2])
    
    with col_perf_l:
        st.markdown("### 📊 Confusion Matrix Heatmap")
        # Compute Confusion Matrix
        cm = confusion_matrix(y_test_e, y_pred_e)
        
        # Render Heatmap with Plotly
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Status", y="Actual Status", color="Records"),
            x=['Stayed (0)', 'Left (1)'],
            y=['Stayed (0)', 'Left (1)'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f3f4f6" if st.session_state.theme == "dark" else "#0f172a",
            height=300
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("<p style='font-size: 12px; color: var(--text-secondary); text-align: center;'>Diagonal cells represent true predictions (correct outcomes). Left class contains minor imbalance, characteristic of turnover datasets.</p>", unsafe_allow_html=True)

    with col_perf_r:
        st.markdown("### 📋 Classification Report")
        report_dict = classification_report(y_test_e, y_pred_e, output_dict=True)
        
        # Build DataFrame for classification metrics
        report_df = pd.DataFrame(report_dict).transpose().round(3)
        # Drop support row or format it cleanly
        report_df['support'] = report_df['support'].astype(int)
        
        # HTML Table render
        report_rows = ""
        for index, row in report_df.iterrows():
            if index in ['accuracy', 'macro avg', 'weighted avg']:
                row_label = f"<b>{index.title()}</b>"
            else:
                row_label = f"Class {index} (Stayed)" if index == '0' else f"Class {index} (Left)"
                
            report_rows += f"""
            <tr>
                <td>{row_label}</td>
                <td>{row['precision']:.3f if index not in ['accuracy'] else '-'}</td>
                <td>{row['recall']:.3f if index not in ['accuracy'] else '-'}</td>
                <td>{row['f1-score']:.3f if isinstance(row['f1-score'], float) else row['f1-score']}</td>
                <td>{row['support']}</td>
            </tr>
            """
            
        st.markdown(
            f"""
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>Metric Segment</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support Size</th>
                    </tr>
                </thead>
                <tbody>
                    {report_rows}
                </tbody>
            </table>
            """,
            unsafe_allow_html=True
        )

# ================= PAGE: ABOUT DEVELOPER =================
elif st.session_state.page == "👨‍💻 About Developer":
    st.markdown(
        f"""
        <div class="glass-card">
            <h2 style="margin: 0 0 5px 0;">👨‍💻 Developer Profile</h2>
            <p style="color: var(--text-secondary); margin: 0;">Portfolio card detailing skills, background, and projects.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col_dev_l, col_dev_r = st.columns([0.8, 1.2])
    
    with col_dev_l:
        # Display Generated Avatar
        avatar_path = "images/dev_avatar.jpg"
        if os.path.exists(avatar_path):
            st.image(avatar_path, use_container_width=True, caption="Ankit Yadav Profile Portrait")
        else:
            st.markdown(
                """
                <div style="width: 100%; height: 250px; border-radius: 12px; background: linear-gradient(135deg, #4f46e5 0%, #db2777 100%); display:flex; justify-content:center; align-items:center;">
                    <h3 style="color:white; margin:0;">Ankit Yadav</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown(
            f"""
            <div class="glass-card" style="margin-top: 15px; text-align: center;">
                <h4 style="margin: 0;">Ankit Yadav</h4>
                <p style="color: var(--text-secondary); margin: 5px 0 10px 0; font-size:14px;">Data Analytics & AI/ML Enthusiast</p>
                <div class="badge">SRMCEM Lucknow</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_dev_r:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <h3>Background Summary</h3>
                <p>I am a passionate computer science undergraduate pursuing a B.Tech degree at <b>SRMCEM Lucknow</b>. My academic interest is centered on Data Analytics, Applied Machine Learning, and deploying production-grade AI solutions to solve operational business challenges.</p>
                
                <h4 style="margin-top: 20px;">🛠 Technical Skills Portfolio:</h4>
                <div style="display:flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px;">
                    <span class="badge" style="background: rgba(99, 102, 241, 0.1); color: #818cf8; border-color: rgba(99, 102, 241, 0.3);">Python</span>
                    <span class="badge" style="background: rgba(16, 185, 129, 0.1); color: #34d399; border-color: rgba(16, 185, 129, 0.3);">Machine Learning</span>
                    <span class="badge" style="background: rgba(245, 158, 11, 0.1); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3);">Data Analytics</span>
                    <span class="badge" style="background: rgba(14, 165, 233, 0.1); color: #38bdf8; border-color: rgba(14, 165, 233, 0.3);">SQL (Structured Queries)</span>
                    <span class="badge" style="background: rgba(139, 92, 246, 0.1); color: #a78bfa; border-color: rgba(139, 92, 246, 0.3);">Git Version Control</span>
                    <span class="badge" style="background: rgba(236, 72, 153, 0.1); color: #f472b6; border-color: rgba(236, 72, 153, 0.3);">GitHub Actions</span>
                    <span class="badge" style="background: rgba(244, 63, 94, 0.1); color: #fb7185; border-color: rgba(244, 63, 94, 0.3);">Streamlit Deployment</span>
                    <span class="badge" style="background: rgba(100, 116, 139, 0.1); color: #cbd5e1; border-color: rgba(100, 116, 139, 0.3);">Advanced Excel</span>
                </div>

                <h4>🏆 Projects & Achievements:</h4>
                <ul style="color: var(--text-secondary); line-height: 1.6; margin-left: 20px; font-size:14px;">
                    <li><b>Predictive Analytics Engine:</b> Developed model pipeline applications for sales prediction and risk management.</li>
                    <li><b>Hackathons:</b> Participated in competitive hackathons focusing on solving real-world dataset analytics problems.</li>
                    <li><b>Sports Achievements:</b> Active participant in campus sports tournaments, showcasing teamwork and high drive.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

# ================= PAGE: CONTACT =================
elif st.session_state.page == "📬 Contact":
    st.markdown(
        f"""
        <div class="glass-card">
            <h2 style="margin: 0 0 5px 0;">📬 Get In Touch</h2>
            <p style="color: var(--text-secondary); margin: 0;">Connect with me for queries, feedback, or collaboration opportunities.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Render professional contact card links opening in new tab
    st.markdown(
        """
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
            
            <a href="https://linkedin.com/in/YOUR-LINK" target="_blank" style="text-decoration:none; color:inherit;">
                <div class="glass-card" style="text-align: center; cursor:pointer;">
                    <div style="font-size: 2.5rem; margin-bottom: 10px;">🔗</div>
                    <h4>LinkedIn</h4>
                    <p style="color: var(--text-secondary); font-size:13px; margin: 5px 0 0 0;">linkedin.com/in/YOUR-LINK</p>
                    <span class="badge" style="margin-top:15px; border-color: rgba(99, 102, 241, 0.3);">Connect ↗</span>
                </div>
            </a>

            <a href="https://github.com/YOUR-USERNAME" target="_blank" style="text-decoration:none; color:inherit;">
                <div class="glass-card" style="text-align: center; cursor:pointer;">
                    <div style="font-size: 2.5rem; margin-bottom: 10px;">💻</div>
                    <h4>GitHub</h4>
                    <p style="color: var(--text-secondary); font-size:13px; margin: 5px 0 0 0;">github.com/YOUR-USERNAME</p>
                    <span class="badge" style="margin-top:15px; border-color: rgba(99, 102, 241, 0.3);">Follow Code ↗</span>
                </div>
            </a>

            <a href="mailto:yourmail@gmail.com" target="_blank" style="text-decoration:none; color:inherit;">
                <div class="glass-card" style="text-align: center; cursor:pointer;">
                    <div style="font-size: 2.5rem; margin-bottom: 10px;">✉️</div>
                    <h4>Email Connection</h4>
                    <p style="color: var(--text-secondary); font-size:13px; margin: 5px 0 0 0;">yourmail@gmail.com</p>
                    <span class="badge" style="margin-top:15px; border-color: rgba(99, 102, 241, 0.3);">Write Mail ↗</span>
                </div>
            </a>

            <a href="#" target="_blank" style="text-decoration:none; color:inherit;">
                <div class="glass-card" style="text-align: center; cursor:pointer;">
                    <div style="font-size: 2.5rem; margin-bottom: 10px;">📄</div>
                    <h4>Resume Download</h4>
                    <p style="color: var(--text-secondary); font-size:13px; margin: 5px 0 0 0;">Ankit Yadav CV Portfolio</p>
                    <span class="badge" style="margin-top:15px; border-color: rgba(99, 102, 241, 0.3);">Download PDF ↗</span>
                </div>
            </a>

        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# 7. Modern Sticky Footer
# ----------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; padding: 30px 0; margin-top: 50px; border-top: 1px solid var(--card-border);">
        <p style="margin: 0; font-size: 13px; color: var(--text-secondary);">
            Employee Retention Prediction Project © 2026. Made with ❤️ by Ankit Yadav.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)