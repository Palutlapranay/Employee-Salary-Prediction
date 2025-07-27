import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact, visible styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 50%, #3498db 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 50%, #3498db 100%);
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1rem;
        font-weight: 400;
    }
    
    .input-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .result-container {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 6px 25px rgba(0,0,0,0.25);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.01); }
        100% { transform: scale(1); }
    }
    
    .result-container h2 {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    
    .result-container p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .stSelectbox, .stSlider {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        border: 1px solid rgba(0,0,0,0.2);
    }
    
    .stSelectbox > div > div {
        color: #333 !important;
        font-weight: 500;
    }
    
    .stSlider > div > div {
        color: #333 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 3px 12px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 3px 12px rgba(0,0,0,0.1);
    }
    
    .metric-card h3 {
        color: #2c3e50;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .metric-card p {
        color: #34495e;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0;
    }
    
    /* Compact section headers */
    .section-header {
        color: #2c3e50;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        padding: 0.3rem 0;
        border-bottom: 2px solid #3498db;
    }
    
    /* Mode toggle styling */
    .mode-toggle {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .mode-toggle label {
        color: white;
        font-weight: 500;
        margin: 0 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚀 AI Salary Predictor</h1>
    <p>Predict your potential salary using advanced machine learning algorithms</p>
</div>
""", unsafe_allow_html=True)

# Theme and Mode selection
col_theme, col_mode = st.columns([1, 1])

with col_theme:
    st.markdown("""
    <div class="mode-toggle">
        <label>🎨 Select Theme:</label>
    </div>
    """, unsafe_allow_html=True)
    theme = st.radio("Theme", ["🌙 Dark", "☀️ Light", "🌊 Ocean", "🌸 Spring"], horizontal=True)

with col_mode:
    st.markdown("""
    <div class="mode-toggle">
        <label>🎯 Select Mode:</label>
    </div>
    """, unsafe_allow_html=True)
    mode = st.radio("Mode", ["📊 Basic", "🎨 Advanced", "📈 Analytics"], horizontal=True)

# Apply theme-specific styling
if theme == "🌙 Dark":
    st.markdown("""
    <style>
        .main, .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important; }
        .main-header { background: rgba(255, 255, 255, 0.1) !important; border: 1px solid rgba(255, 255, 255, 0.2) !important; }
        .main-header h1 { color: #ffffff !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.7) !important; }
        .main-header p { color: rgba(255, 255, 255, 0.9) !important; }
        .mode-toggle { background: rgba(255, 255, 255, 0.1) !important; }
        .mode-toggle label { color: white !important; }
    </style>
    """, unsafe_allow_html=True)
elif theme == "☀️ Light":
    st.markdown("""
    <style>
        .main, .stApp { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%) !important; }
        .main-header { background: rgba(255, 255, 255, 0.9) !important; border: 1px solid rgba(0, 0, 0, 0.1) !important; }
        .main-header h1 { color: #2c3e50 !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.1) !important; }
        .main-header p { color: #495057 !important; }
        .mode-toggle { background: rgba(0, 0, 0, 0.05) !important; }
        .mode-toggle label { color: #495057 !important; }
    </style>
    """, unsafe_allow_html=True)
elif theme == "🌊 Ocean":
    st.markdown("""
    <style>
        .main, .stApp { background: linear-gradient(135deg, #006994 0%, #0099cc 50%, #00bfff 100%) !important; }
        .main-header { background: rgba(255, 255, 255, 0.15) !important; border: 1px solid rgba(255, 255, 255, 0.3) !important; }
        .main-header h1 { color: #ffffff !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important; }
        .main-header p { color: rgba(255, 255, 255, 0.95) !important; }
        .mode-toggle { background: rgba(255, 255, 255, 0.1) !important; }
        .mode-toggle label { color: white !important; }
    </style>
    """, unsafe_allow_html=True)
elif theme == "🌸 Spring":
    st.markdown("""
    <style>
        .main, .stApp { background: linear-gradient(135deg, #ff6b9d 0%, #c44569 50%, #f8b5d3 100%) !important; }
        .main-header { background: rgba(255, 255, 255, 0.15) !important; border: 1px solid rgba(255, 255, 255, 0.3) !important; }
        .main-header h1 { color: #ffffff !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important; }
        .main-header p { color: rgba(255, 255, 255, 0.95) !important; }
        .mode-toggle { background: rgba(255, 255, 255, 0.1) !important; }
        .mode-toggle label { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# Create three columns for compact layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">👤 Personal Info</div>', unsafe_allow_html=True)
    
    age = st.slider("🎂 Age", 18, 65, 25)
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    education = st.selectbox("🎓 Education", ["Bachelors", "Masters", "PhD"])
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">💼 Professional</div>', unsafe_allow_html=True)
    
    job_role = st.selectbox("🏢 Job Role", ["Data Analyst", "Software Engineer", "Manager", "HR Specialist", "Developer"])
    experience = st.slider("⏰ Experience", 0, 40, 2)
    workclass = st.selectbox("🏛️ Sector", ["Private", "Government"])
    
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">👥 Demographics</div>', unsafe_allow_html=True)
    
    marital_status = st.selectbox("💍 Status", ["Single", "Married", "Divorced", "Widowed"])
    relationship = st.selectbox("👨‍👩‍👧‍👦 Family", ["Not-in-family", "Spouse", "Own-child", "Unmarried"])
    race = st.selectbox("🌍 Ethnicity", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prepare user input
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Education": education,
    "JobRole": job_role,
    "Experience": experience,
    "Workclass": workclass,
    "MaritalStatus": marital_status,
    "Relationship": relationship,
    "Race": race
}])

# Load and prepare training data
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df

df = load_data()

label_encoders = {}
le_cols = ['Gender', 'Education', 'JobRole', 'Workclass', 'MaritalStatus', 'Relationship', 'Race']

# Encode the dataset and store encoders
for col in le_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train the model
@st.cache_resource
def train_model():
    X = df.drop("Salary", axis=1)
    y = df["Salary"]
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model()

# Encode input with stored encoders
for col in le_cols:
    le = label_encoders[col]
    input_df[col] = le.transform([input_df[col][0]])

# Predict
salary_pred = model.predict(input_df)[0]

# Display result based on mode
if mode == "📊 Basic":
    st.markdown(f"""
    <div class="result-container">
        <h2>💰 Predicted Salary</h2>
        <p>₹{int(salary_pred):,} per year</p>
    </div>
    """, unsafe_allow_html=True)
    
elif mode == "🎨 Advanced":
    st.markdown(f"""
    <div class="result-container">
        <h2>💰 AI-Predicted Salary</h2>
        <p>₹{int(salary_pred):,} per year</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">Based on advanced ML algorithms</p>
    </div>
    """, unsafe_allow_html=True)
    
else:  # Analytics Mode
    st.markdown(f"""
    <div class="result-container">
        <h2>💰 Predicted Salary</h2>
        <p>₹{int(salary_pred):,} per year</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">With detailed market analysis</p>
    </div>
    """, unsafe_allow_html=True)

# Compact metrics display
metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>📈 Average</h3>
        <p>₹{int(df['Salary'].mean()):,}</p>
    </div>
    """, unsafe_allow_html=True)

with metric_col2:
    position = "Above" if salary_pred > df['Salary'].mean() else "Below"
    st.markdown(f"""
    <div class="metric-card">
        <h3>🎯 Position</h3>
        <p>{position} Average</p>
    </div>
    """, unsafe_allow_html=True)

with metric_col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>📊 Range</h3>
        <p>₹{int(df['Salary'].min()):,} - ₹{int(df['Salary'].max()):,}</p>
    </div>
    """, unsafe_allow_html=True)

# Charts based on mode
if mode == "📈 Analytics":
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)
    
    # Create interactive charts with Plotly
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🎯 Roles", "📈 Experience"])
    
elif mode == "🎨 Advanced":
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Salary Insights</div>', unsafe_allow_html=True)
    
    # Show only basic charts
    tab1, tab2 = st.tabs(["📊 Distribution", "🎯 Roles"])
    
else:  # Basic Mode
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Basic Analysis</div>', unsafe_allow_html=True)
    
    # Show only distribution chart
    tab1, = st.tabs(["📊 Distribution"])

with tab1:
    # Compact salary distribution
    fig1 = px.histogram(
        df, 
        x='Salary', 
        nbins=15,
        title="Salary Distribution",
        color_discrete_sequence=['#3498db'],
        opacity=0.8
    )
    fig1.add_vline(x=salary_pred, line_dash="dash", line_color="#e74c3c", 
                   annotation_text=f"Your: ₹{int(salary_pred):,}")
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins", size=10),
        height=300
    )
    st.plotly_chart(fig1, use_container_width=True)

if mode != "📊 Basic":
    with tab2:
        # Compact job role comparison
        if 'JobRole' in df.columns:
            role_salary = df.groupby('JobRole')['Salary'].mean().reset_index()
            fig2 = px.bar(
                role_salary,
                x='JobRole',
                y='Salary',
                title="Salary by Role",
                color='Salary',
                color_continuous_scale='viridis'
            )
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins", size=10),
                xaxis_tickangle=-45,
                height=300
            )
            st.plotly_chart(fig2, use_container_width=True)

if mode == "📈 Analytics":
    with tab3:
        # Compact experience vs salary
        fig3 = px.scatter(
            df,
            x='Experience',
            y='Salary',
            title="Experience vs Salary",
            color='Salary',
            size='Age',
            hover_data=['Age', 'Education'],
            color_continuous_scale='plasma'
        )
        fig3.add_scatter(
            x=[experience],
            y=[salary_pred],
            mode='markers',
            marker=dict(size=12, color='#e74c3c', symbol='star'),
            name='Your Prediction'
        )
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Poppins", size=10),
            height=300
        )
        st.plotly_chart(fig3, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Compact footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
    <p style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">
        🤖 ML Powered | 💼 Salary Predictor | ✨ By Pranay
    </p>
</div>
""", unsafe_allow_html=True)
