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

# Custom CSS for modern styling
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
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    .input-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .result-container {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .result-container h2 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .result-container p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.5rem;
        font-weight: 500;
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .stSelectbox, .stSlider {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 1px solid rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card h3 {
        color: #333;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        color: #666;
        font-size: 1rem;
        margin: 0;
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

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown("### 📋 Personal Information")
    
    # User Inputs with better organization
    age = st.slider("🎂 Age", 18, 65, 25, help="Select your current age")
    gender = st.selectbox("👤 Gender", ["Male", "Female"], help="Select your gender")
    education = st.selectbox("🎓 Education Level", ["Bachelors", "Masters", "PhD"], help="Select your highest education level")
    
    st.markdown("### 💼 Professional Details")
    job_role = st.selectbox("🏢 Job Role", ["Data Analyst", "Software Engineer", "Manager", "HR Specialist", "Developer", "Data Scientist", "Product Manager"], help="Select your current or desired job role")
    experience = st.slider("⏰ Years of Experience", 0, 40, 2, help="Select your years of professional experience")
    workclass = st.selectbox("🏛️ Work Sector", ["Private", "Government", "Self-employed"], help="Select your work sector")
    
    st.markdown("### 👥 Demographics")
    marital_status = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced", "Widowed"], help="Select your marital status")
    relationship = st.selectbox("👨‍👩‍👧‍👦 Relationship", ["Not-in-family", "Spouse", "Own-child", "Unmarried", "Other-relative"], help="Select your family relationship")
    race = st.selectbox("🌍 Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], help="Select your race/ethnicity")
    
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

# Display result with enhanced styling
st.markdown(f"""
<div class="result-container">
    <h2>💰 Your Predicted Salary</h2>
    <p>₹{int(salary_pred):,} per year</p>
    <p style="font-size: 1rem; margin-top: 1rem;">Based on your profile and market analysis</p>
</div>
""", unsafe_allow_html=True)

# Enhanced visualizations
with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Salary Insights")
    
    # Create metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Average Salary</h3>
            <p>₹{int(df['Salary'].mean()):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Your Position</h3>
            <p>{'Above' if salary_pred > df['Salary'].mean() else 'Below'} Average</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Salary Range</h3>
            <p>₹{int(df['Salary'].min()):,} - ₹{int(df['Salary'].max()):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced charts section
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.markdown("### 📈 Salary Analysis Dashboard")

# Create interactive charts with Plotly
tab1, tab2, tab3 = st.tabs(["📊 Salary Distribution", "🎯 Role Comparison", "📈 Experience vs Salary"])

with tab1:
    # Salary distribution histogram
    fig1 = px.histogram(
        df, 
        x='Salary', 
        nbins=20,
        title="Salary Distribution Across All Employees",
        color_discrete_sequence=['#667eea'],
        opacity=0.8
    )
    fig1.add_vline(x=salary_pred, line_dash="dash", line_color="red", 
                   annotation_text=f"Your Prediction: ₹{int(salary_pred):,}")
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins", size=12)
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    # Job role comparison (if JobRole is in the dataset)
    if 'JobRole' in df.columns:
        role_salary = df.groupby('JobRole')['Salary'].mean().reset_index()
        fig2 = px.bar(
            role_salary,
            x='JobRole',
            y='Salary',
            title="Average Salary by Job Role",
            color='Salary',
            color_continuous_scale='viridis'
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Poppins", size=12),
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    # Experience vs Salary scatter plot
    fig3 = px.scatter(
        df,
        x='Experience',
        y='Salary',
        title="Experience vs Salary Relationship",
        color='Salary',
        size='Age',
        hover_data=['Age', 'Education'],
        color_continuous_scale='plasma'
    )
    fig3.add_scatter(
        x=[experience],
        y=[salary_pred],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Your Prediction'
    )
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins", size=12)
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px;">
    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
        🤖 Powered by Machine Learning | 💼 Employee Salary Prediction System | ✨ Designed by Pranay
    </p>
</div>
""", unsafe_allow_html=True)
