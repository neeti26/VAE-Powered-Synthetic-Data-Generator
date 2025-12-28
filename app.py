import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Constants
THREAT_LEVELS = ["Low", "Medium", "High"]

# UI Styling & Font
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Orbitron', monospace;
        background: linear-gradient(135deg, #020024, #090979, #00d4ff);
        color: #00ffe7;
    }
    .stButton > button {
        background-color: #0077ff;
        color: white;
        font-weight: 700;
        border-radius: 12px;
        padding: 10px 24px;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #00ffe7;
        color: #0077ff;
        box-shadow: 0 0 8px #00ffe7;
    }
    h1, h2, h3, h4, h5 {
        text-shadow: 0 0 8px #00ffe7;
    }
    .stFileUploader>div>div>input {
        border-radius: 8px;
        border: 2px solid #00ffe7;
        padding: 8px;
    }
    .stSlider > div[role="slider"] {
        border-radius: 8px;
        background: #0077ff;
    }
    .reportview-container .main footer, .reportview-container .main header {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ VAE-powered Cybersecurity Synthetic Data & Prediction Tool")
st.markdown("Generate synthetic network data, get threat predictions & actionable insights!")

# Number of synthetic samples
num_samples = st.slider("Number of synthetic samples to generate", 10, 10000, 1000, step=10)

# Feature count fixed to 20
num_features = 20

# --- File upload feature ---
uploaded_file = st.file_uploader("Or upload your own CSV file with exactly 20 features", type=["csv"])
user_data = None
if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file)
        if user_data.shape[1] != 20:
            st.error("Uploaded CSV must have exactly 20 features (columns).")
            user_data = None
        else:
            st.success(f"Uploaded data with {len(user_data)} samples loaded successfully.")
            st.dataframe(user_data.head(10))
    except Exception as e:
        st.error(f"Error reading uploaded CSV file: {e}")
        user_data = None

# Generate synthetic data locally
def generate_synthetic(num_samples, num_features):
    np.random.seed(42)  # for reproducibility
    data = np.random.rand(num_samples, num_features) * 10000  # values scaled 0-10000
    columns = [f"feature_{i+1}" for i in range(num_features)]
    df = pd.DataFrame(data, columns=columns)
    return df

# Mock prediction logic (random assignment weighted by sum of features)
def mock_predict(df):
    scores = df.sum(axis=1)
    preds = []
    for s in scores:
        if s < 80000:
            preds.append("Low")
        elif s < 140000:
            preds.append("Medium")
        else:
            preds.append("High")
    return preds

def run_analysis(df):
    st.subheader("Data Sample (First 10 rows):")
    st.dataframe(df.head(10))

    with st.spinner("Predicting threat levels..."):
        preds = mock_predict(df)
        df["Predicted Threat Level"] = preds

    st.subheader("Predictions")
    st.dataframe(df.head(10))

    # Pie chart of threat levels
    st.markdown("### 🔍 Threat Level Distribution")
    dist = df["Predicted Threat Level"].value_counts(normalize=True).reset_index()
    dist.columns = ["Threat Level", "Proportion"]
    fig = px.pie(dist, values="Proportion", names="Threat Level",
                 color_discrete_sequence=["#00ffe7", "#0077ff", "#0047b3"])
    st.plotly_chart(fig, use_container_width=True)

    # Simulated real-time monitoring graph
    st.markdown("### 📈 Real-Time Threat Monitoring (Simulated)")
    times = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='T')
    monitoring_df = df.copy()
    monitoring_df["Timestamp"] = times
    threat_map = {"Low": 1, "Medium": 2, "High": 3}
    monitoring_df["Threat_Level_Num"] = monitoring_df["Predicted Threat Level"].map(threat_map)
    fig2 = px.line(monitoring_df, x="Timestamp", y="Threat_Level_Num",
                   labels={"Threat_Level_Num": "Threat Level (1=Low, 3=High)"},
                   title="Threat Level Over Time (Simulated)")
    st.plotly_chart(fig2, use_container_width=True)

    # Conclusions & recommendations
    st.markdown("---")
    st.subheader("Conclusions & Insights")

    st.markdown(f"""
    - **Sample Size:** {len(df)} samples analyzed.
    - **Threat Level Breakdown:**
    {dist.to_markdown(index=False)}

    - **Use Case:** Synthetic or uploaded data helps augment limited datasets and improve detection accuracy.
    - **Real-Time Applications:** Useful for proactive threat detection in live network monitoring.
    - **Efficiency:** Saves costly data collection & labeling time.
    - **Precautions:** Regular validation with real data is essential; monitor model drift.
    - **Future Improvements:** Integrate with actual VAE models and real prediction engines for production-grade results.
    """)

if user_data is not None:
    run_analysis(user_data)
else:
    if st.button("Generate Synthetic Data & Predict"):
        synthetic_df = generate_synthetic(num_samples, num_features)
        st.success(f"Generated {len(synthetic_df)} synthetic data samples!")
        run_analysis(synthetic_df)
    else:
        st.info("Click the button above to generate synthetic data and get predictions.")

# Footer
st.markdown(
    """
    <hr style="border: 1px solid #00ffe7; margin-top: 2rem;">
    <p style='text-align:center; color:#00ffe7; font-size:12px;'>Powered by VAE | Network Intrusion Detection Tool | Made by YourName</p>
    """,
    unsafe_allow_html=True,
)
