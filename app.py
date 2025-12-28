import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Attack-Dreams | Cyber Threat Lab",
    page_icon="🛡️",
    layout="wide",
)

# ------------------------------------------------------------
# THEME: ELEGANT CYBER (BLACK / BLUE) + BACKGROUND
# ------------------------------------------------------------
CYBER_BG_URL = "https://images.pexels.com/photos/546819/pexels-photo-546819.jpeg"  # change to your own cyber image if you like

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* App background */
    [data-testid="stAppViewContainer"] {{
        background:
            linear-gradient(135deg, rgba(15,23,42,0.96), rgba(2,6,23,0.98)),
            url("{CYBER_BG_URL}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    html, body, [class*="css"] {{
        font-family: "Space Grotesk", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #e5f2ff;
    }}

    /* Main container card */
    .main > div {{
        background: linear-gradient(135deg, rgba(2,6,23,0.92) 0%, rgba(2,6,23,0.96) 55%, rgba(2,8,20,0.96) 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 18px;
        border: 1px solid rgba(15, 23, 42, 0.95);
        box-shadow:
            0 20px 40px rgba(0, 0, 0, 0.95),
            0 0 0 1px rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(10px);
    }}

    /* Attack-Dreams header block */
    .attack-hero {{
        color: #f9fafb;
    }}
    .attack-hero a {{
        color: #38bdf8 !important;
        text-decoration: none;
    }}
    .attack-hero a:hover {{
        color: #0ea5e9 !important;
        text-decoration: underline;
    }}

    /* Section text blocks that must be clearly visible */
    .attack-section-text {{
        color: #ffffff !important;
    }}
    .attack-section-title {{
        color: #ffffff !important;
    }}

    h1 {{
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-size: 1.4rem;
        color: #f9fafb;
    }}
    h2, h3 {{
        font-weight: 600;
        color: #e5f2ff;
    }}
    h1, h2, h3, h4, h5 {{
        text-shadow: none;
    }}

    /* Accent label */
    .app-subtitle {{
        font-size: 0.78rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #cbd5f5;
    }}

    /* Divider */
    hr {{
        border: none;
        height: 1px;
        margin: 1.2rem 0 1.0rem 0;
        background: linear-gradient(90deg, transparent, #1e293b, transparent);
    }}

    /* KPI cards */
    .kpi-card {{
        background: radial-gradient(circle at top left, rgba(15,23,42,0.95) 0, rgba(2,6,23,0.98) 45%, rgba(0,0,0,1) 100%);
        border-radius: 14px;
        border: 1px solid rgba(37, 99, 235, 0.9);
        padding: 0.9rem 1rem;
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.9);
    }}
    .kpi-label {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #9ca3af;
        margin-bottom: 0.15rem;
    }}
    .kpi-value {{
        font-size: 1.15rem;
        font-weight: 600;
        color: #e5f2ff;
    }}
    .kpi-tag {{
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #38bdf8;
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #020617, #020617);
        color: #e5f2ff;
        font-weight: 500;
        border-radius: 999px;
        border: 1px solid #2563eb;
        padding: 0.45rem 1.55rem;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        font-size: 0.78rem;
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.95);
        transition:
            background 0.18s ease,
            border-color 0.18s ease,
            transform 0.12s ease,
            box-shadow 0.12s ease;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, #0ea5e9, #1d4ed8);
        border-color: #38bdf8;
        color: #020617;
        transform: translateY(-1px);
        box-shadow: 0 18px 30px rgba(15, 23, 42, 0.95);
    }}

    /* File uploader */
    .stFileUploader > div {{
        background: rgba(2,6,23,0.9);
        border-radius: 12px;
        border: 1px solid #1f2937;
    }}
    .stFileUploader label {{
        color: #e5f2ff;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
    }}

    /* Slider */
    .stSlider [data-baseweb="slider"] > div {{
        background: #020617;
    }}
    .stSlider [data-baseweb="slider"] > div > div {{
        background: linear-gradient(90deg, #020617, #2563eb);
    }}
    .stSlider [role="slider"] {{
        background: #e5f2ff;
        border-radius: 999px;
        box-shadow: 0 0 0 2px #2563eb;
    }}

    /* Tables / DataFrames */
    .stDataFrame, .stTable {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 14px 26px rgba(15, 23, 42, 0.9);
    }}
    .stDataFrame table, .stTable table {{
        background: rgba(2,6,23,0.95);
        color: #e5f2ff;
    }}
    .stDataFrame thead tr, .stTable thead tr {{
        background: #020617 !important;
        border-bottom: 1px solid #1f2937 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.74rem;
    }}
    .stDataFrame tbody tr:nth-child(odd),
    .stTable tbody tr:nth-child(odd) {{
        background: #020617 !important;
    }}
    .stDataFrame tbody tr:nth-child(even),
    .stTable tbody tr:nth-child(even) {{
        background: #030712 !important;
    }}

    /* Alerts */
    .stAlert {{
        border-radius: 10px;
        border: 1px solid #1f2937;
    }}

    /* Sidebar styling – force white text */
    section[data-testid="stSidebar"] {{
        background: radial-gradient(circle at top, rgba(15,23,42,0.98) 0, rgba(15,23,42,0.98) 40%, rgba(0,0,0,0.98) 100%);
        border-right: 1px solid #020617;
        color: #e5f2ff;
    }}
    section[data-testid="stSidebar"] * {{
        color: #e5f2ff !important;
    }}
    section[data-testid="stSidebar"] h2 {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
    }}

    /* Footer */
    .attack-footer {{
        text-align: center;
        color: #9ca3af;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        padding-top: 0.75rem;
    }}
    .attack-footer span {{
        color: #38bdf8;
        font-weight: 600;
    }}

    /* Hide default header/footer */
    .reportview-container .main footer,
    .reportview-container .main header {{
        visibility: hidden;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: #020617;
    }}
    ::-webkit-scrollbar-thumb {{
        background: #111827;
        border-radius: 999px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# TITLE / HERO
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="attack-hero">
        <h1 id="attack-dreams">🛡️ Attack-Dreams</h1>
        <div class="app-subtitle">CYBER THREAT SIMULATION &amp; ANALYTICS LAB</div>
        <p>
            Work with <strong>synthetic</strong> and uploaded telemetry to explore threat levels,
            model behavior, and stability under different traffic regimes.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# SIDEBAR: GLOBAL CONTROLS
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Controls")
    mode = st.radio(
        "Analysis mode",
        ["Synthetic only", "Use uploaded data if available"],
    )

    num_samples = st.slider(
        "Synthetic samples",
        100,
        10000,
        2000,
        step=100,
        help="Number of synthetic events generated when using synthetic mode.",
    )

    st.markdown("---")
    st.markdown("## Threat calibration")
    low_cutoff = st.slider(
        "Low / Medium boundary",
        min_value=40000,
        max_value=120000,
        value=50000,
        step=5000,
    )
    high_cutoff = st.slider(
        "Medium / High boundary",
        min_value=low_cutoff + 10000,
        max_value=200000,
        value=140000,
        step=5000,
    )

    st.markdown("---")
    show_raw = st.checkbox("Show full scored dataset", value=False)
    allow_download = st.checkbox("Enable CSV export", value=True)

# fixed feature space
num_features = 20

# ------------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV with exactly 20 feature columns",
    type=["csv"],
)

user_data = None
if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file)
        if user_data.shape[1] != 20:
            st.error("Uploaded CSV must have exactly 20 features (columns).")
            user_data = None
        else:
            st.success(f"Ingested {len(user_data)} records from uploaded dataset.")
    except Exception as e:
        st.error(f"Error reading uploaded CSV file: {e}")
        user_data = None

# ------------------------------------------------------------------
# DATA GENERATION & PREDICTION
# ------------------------------------------------------------------
def generate_synthetic(num_samples, num_features):
    np.random.seed(42)
    data = np.random.rand(num_samples, num_features) * 10000
    columns = [f"feature_{i+1}" for i in range(num_features)]
    df = pd.DataFrame(data, columns=columns)
    return df

def mock_predict(df, low_cutoff, high_cutoff):
    scores = df.sum(axis=1)
    preds = []
    for s in scores:
        if s < low_cutoff:
            preds.append("Low")
        elif s < high_cutoff:
            preds.append("Medium")
        else:
            preds.append("High")
    return preds, scores

# ------------------------------------------------------------------
# ANALYSIS PIPELINE
# ------------------------------------------------------------------
def run_analysis(df):
    preds, scores = mock_predict(df, low_cutoff, high_cutoff)
    df = df.copy()
    df["AttackDreams_Score"] = scores
    df["Predicted Threat Level"] = preds

    # KPI row
    total_events = len(df)
    threat_counts = df["Predicted Threat Level"].value_counts()
    pct_high = 100 * threat_counts.get("High", 0) / total_events
    pct_medium = 100 * threat_counts.get("Medium", 0) / total_events
    pct_low = 100 * threat_counts.get("Low", 0) / total_events

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total events</div>
                <div class="kpi-value">{total_events:,}</div>
                <div class="kpi-tag">Current batch</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">High-risk share</div>
                <div class="kpi-value">{pct_high:.1f}%</div>
                <div class="kpi-tag">Threat tier = high</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Medium / low mix</div>
                <div class="kpi-value">{pct_medium:.1f}% / {pct_low:.1f}%</div>
                <div class="kpi-tag">Threat tiers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    snapshot_col, meta_col = st.columns([3, 2])

    with snapshot_col:
        # Sample view title and content in white
        st.markdown(
            '<h3 class="attack-section-title">Sample view</h3>',
            unsafe_allow_html=True,
        )
        st.dataframe(df.head(10), use_container_width=True)

    with meta_col:
        # Scoring profile title and text in white
        st.markdown(
            '<h3 class="attack-section-title">Scoring profile</h3>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="attack-section-text">
            <a id="scoring-profile"></a>
            <strong>Score distribution:</strong> continuous sum of 20 features.<br>
            <strong>Low:</strong> score &lt; 50,000<br>
            <strong>Medium:</strong> 50,000 ≤ score &lt; 140,000<br>
            <strong>High:</strong> score ≥ 140,000.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Threat composition – dynamic / interactive
    st.markdown("### Threat composition")
    dist = df["Predicted Threat Level"].value_counts(normalize=True).reset_index()
    dist.columns = ["Threat Level", "Proportion"]

    filter_choice = st.selectbox(
        "View",
        ["All tiers", "High only", "Medium only", "Low only"],
        index=0,
    )

    if filter_choice != "All tiers":
        tier = filter_choice.split()[0]
        df_filtered = df[df["Predicted Threat Level"] == tier]
        dist_filtered = df_filtered["Predicted Threat Level"].value_counts(
            normalize=True
        ).reset_index()
        dist_filtered.columns = ["Threat Level", "Proportion"]
    else:
        df_filtered = df
        dist_filtered = dist

    # Donut chart
    fig = px.pie(
        dist_filtered,
        values="Proportion",
        names="Threat Level",
        color="Threat Level",
        color_discrete_map={
            "Low": "#22c55e",
            "Medium": "#eab308",
            "High": "#ef4444",
        },
        hole=0.45,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="#e5f2ff"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(15,23,42,0.8)",
            borderwidth=1,
        ),
    )
    fig.update_traces(
        textfont_size=12,
        marker=dict(line=dict(color="#020617", width=1.5)),
    )

    # Horizontal bar with percentages
    dist_bar = dist.copy()
    dist_bar["Percentage"] = dist_bar["Proportion"] * 100
    fig_bar = px.bar(
        dist_bar,
        x="Percentage",
        y="Threat Level",
        orientation="h",
        text="Percentage",
        color="Threat Level",
        color_discrete_map={
            "Low": "#22c55e",
            "Medium": "#eab308",
            "High": "#ef4444",
        },
    )
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="#e5f2ff"),
        showlegend=False,
        xaxis_title="Percentage of events",
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.plotly_chart(fig_bar, use_container_width=True)

    # Time series view – static 24h simulation
    st.markdown("### Time-profiled threat levels (simulated)")

    time_index = pd.date_range(
        end=pd.Timestamp.now(),
        periods=24,
        freq="H",
    )  # 24 hours [web:81]

    mix = df["Predicted Threat Level"].value_counts(normalize=True).to_dict()
    mix_low = mix.get("Low", 0.0)
    mix_med = mix.get("Medium", 0.0)
    mix_high = mix.get("High", 0.0)

    base = np.linspace(1.0, 2.6, len(time_index))
    noise = np.random.default_rng(0).normal(0, 0.15, len(time_index))
    intensity = np.clip(base + noise, 0.8, 3.0)

    level_series = []
    for val in intensity:
        if val < 1.2 + mix_low:
            level_series.append(1)
        elif val < 2.0 + mix_med:
            level_series.append(2)
        else:
            level_series.append(3)

    monitoring_df = pd.DataFrame(
        {
            "Timestamp": time_index,
            "Threat_Level_Num": level_series,
        }
    )

    fig2 = px.line(
        monitoring_df,
        x="Timestamp",
        y="Threat_Level_Num",
        labels={"Threat_Level_Num": "Threat level index"},
        color_discrete_sequence=["#38bdf8"],
    )
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,0.95)",
        font=dict(family="Space Grotesk", color="#e5f2ff"),
        hovermode="x unified",
        xaxis=dict(showgrid=False),
        yaxis=dict(
            tickmode="array",
            tickvals=[1, 2, 3],
            ticktext=["Low", "Medium", "High"],
            gridcolor="rgba(30,64,175,0.7)",
            zeroline=False,
        ),
    )
    fig2.update_traces(line=dict(width=2.4, shape="spline"))
    st.plotly_chart(fig2, use_container_width=True)

    # Optional raw table & download
    if show_raw:
        st.markdown("### Scored dataset")
        st.dataframe(df, use_container_width=True)

    if allow_download:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scored dataset as CSV",
            csv,
            file_name="attack_dreams_scored.csv",
            mime="text/csv",
        )

    # Analyst summary: heading and text forced to white
    st.markdown("---")
    st.markdown(
        '<h3 class="attack-section-title" id="analyst-notes">Analyst notes</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="attack-section-text">
        <strong>Events scored:</strong> {total_events:,} with a 20-dimensional synthetic feature space.<br><br>
        <strong>Threat mix:</strong><br>
        {dist.to_markdown(index=False)}<br><br>
        <strong>Usage:</strong> Prototype threat thresholds, study stability of tiers, and plug into downstream SIEM/SOAR pipelines.<br>
        <strong>Next step:</strong> Replace the mock score with your VAE encoder–decoder and calibrate thresholds on real incidents.
        </div>
        """,
        unsafe_allow_html=True,
    )

    return df

# ------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------
if mode == "Use uploaded data if available" and user_data is not None:
    st.info("Using uploaded dataset as primary signal source.")
    final_df = run_analysis(user_data)
else:
    if st.button("Run synthetic simulation"):
        synthetic_df = generate_synthetic(num_samples, num_features)
        st.success(f"Generated {len(synthetic_df)} synthetic samples for Attack-Dreams.")
        final_df = run_analysis(synthetic_df)
    else:
        st.info("Upload a dataset or run a synthetic simulation to start Attack-Dreams.")

# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
st.markdown(
    """
    <hr>
    <p class="attack-footer">
        <span>Attack-Dreams</span> • Synthetic-first cyber threat simulation surface • For research & experimentation only
    </p>
    """,
    unsafe_allow_html=True,
)
