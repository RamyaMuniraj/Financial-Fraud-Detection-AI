
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- Page setup ----------
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fraud_detection_model.joblib"
DATA_PATH = ROOT / "data" / "synthetic_fraud_dataset_v2.csv"

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Premium fintech theme ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #071225 0%, #10142c 55%, #17102e 100%);
        color: #edf4ff;
    }
    [data-testid="stSidebar"] {
        background: #0b1022;
        border-right: 1px solid #29254b;
    }
    h1, h2, h3, p, label, [data-testid="stMetricLabel"] {
        color: #edf4ff !important;
    }
    [data-testid="stMetric"] {
        background: rgba(22, 31, 59, 0.85);
        border: 1px solid #34305f;
        padding: 18px;
        border-radius: 14px;
    }
    .hero {
        padding: 24px;
        border: 1px solid #35316a;
        border-radius: 18px;
        background: linear-gradient(110deg, #171d3c, #251344);
        margin-bottom: 22px;
    }
    .hero p { color: #aab8d5 !important; }
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #00d9ff);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #141b32;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load model and data ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

st.sidebar.title("🛡️ FraudGuard AI")
st.sidebar.caption("Financial Intelligence Platform")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Fraud Analytics",
        "Fraud Prediction",
        "Transaction Explorer",
        "Model Information",
    ],
)

try:
    model_bundle = load_model()
    model = model_bundle["model"]
    threshold = float(model_bundle.get("threshold", 0.8))
    feature_names = model_bundle.get("feature_names", [])
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

data = load_data()

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <h1>⚡ FraudGuard AI</h1>
      <p>Intelligent transaction monitoring • Machine learning • Risk insights</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Overview ----------
if page == "Overview":
    st.subheader("Financial Risk Overview")

    if data.empty:
        st.warning("The synthetic transaction dataset was not found.")
        st.stop()

    total = len(data)
    fraud_count = int(data["Fraud_Label"].sum())
    fraud_rate = fraud_count / total * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Cases", f"{fraud_count:,}")
    c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    c4.metric("Decision Threshold", f"{threshold:.2f}")

    left, right = st.columns(2)

    with left:
        st.markdown("### Transaction Class Distribution")
        counts = (
            data["Fraud_Label"]
            .map({0: "Legitimate", 1: "Fraudulent"})
            .value_counts()
            .rename_axis("Transaction Status")
            .reset_index(name="Count")
        )
        fig = px.pie(
            counts,
            names="Transaction Status",
            values="Count",
            hole=0.62,
            color="Transaction Status",
            color_discrete_map={
                "Legitimate": "#00D9FF",
                "Fraudulent": "#A855F7",
            },
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#EDF4FF",
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Fraud by Transaction Amount")
        amount_col = "Transaction_Amount"
        fig = px.histogram(
            data,
            x=amount_col,
            color=data["Fraud_Label"].map(
                {0: "Legitimate", 1: "Fraudulent"}
            ),
            nbins=35,
            barmode="overlay",
            opacity=0.75,
            labels={"color": "Transaction Status"},
            color_discrete_map={
                "Legitimate": "#00D9FF",
                "Fraudulent": "#A855F7",
            },
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#EDF4FF",
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Recent Transactions")
    st.dataframe(data.head(10), use_container_width=True)


# ---------- Fraud Analytics ----------
elif page == "Fraud Analytics":
    st.subheader("📊 Fraud Analytics")
    st.caption("Explore fraud patterns across transaction categories.")

    if data.empty or "Fraud_Label" not in data.columns:
        st.warning("The dataset or Fraud_Label column is missing.")
    else:
        analytics_data = data.copy()
        analytics_data["Fraud Status"] = analytics_data["Fraud_Label"].map(
            {0: "Legitimate", 1: "Fraudulent"}
        )

        total_transactions = len(analytics_data)
        fraud_cases = int(analytics_data["Fraud_Label"].sum())
        fraud_rate = (
            fraud_cases / total_transactions * 100
            if total_transactions else 0
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions Analyzed", f"{total_transactions:,}")
        col2.metric("Fraud Cases", f"{fraud_cases:,}")
        col3.metric("Overall Fraud Rate", f"{fraud_rate:.2f}%")

        st.markdown("### Fraud Rate by Category")

        category_options = [
            ("Transaction_Type", "Transaction Type"),
            ("Device_Type", "Device Type"),
            ("Merchant_Category", "Merchant Category"),
            ("Card_Type", "Card Type"),
            ("Location", "Location"),
        ]
        available_categories = [
            (column, label)
            for column, label in category_options
            if column in analytics_data.columns
        ]

        if available_categories:
            selected = st.selectbox(
                "Choose a category",
                options=[column for column, _ in available_categories],
                format_func=lambda column: next(
                    label for key, label in available_categories
                    if key == column
                ),
            )

            category_stats = (
                analytics_data.groupby(selected, dropna=False)
                .agg(
                    Transactions=("Fraud_Label", "size"),
                    Fraud_Cases=("Fraud_Label", "sum"),
                    Fraud_Rate=("Fraud_Label", "mean"),
                )
                .reset_index()
            )
            category_stats["Fraud_Rate_Pct"] = (
                category_stats["Fraud_Rate"] * 100
            )
            category_stats = category_stats.sort_values(
                "Fraud_Rate_Pct", ascending=False
            )

            fig = px.bar(
                category_stats,
                x=selected,
                y="Fraud_Rate_Pct",
                color="Fraud_Rate_Pct",
                color_continuous_scale=["#00D9FF", "#7C3AED", "#D946EF"],
                hover_data=["Transactions", "Fraud_Cases"],
                labels={
                    selected: "Category",
                    "Fraud_Rate_Pct": "Fraud Rate (%)",
                    "Transactions": "Total Transactions",
                    "Fraud_Cases": "Fraud Cases",
                },
                title="Percentage of Transactions Labeled as Fraud",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#EDF4FF",
                xaxis_title="",
                yaxis_title="Fraud Rate (%)",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                category_stats[
                    [selected, "Transactions", "Fraud_Cases", "Fraud_Rate_Pct"]
                ].rename(
                    columns={
                        "Fraud_Rate_Pct": "Fraud Rate (%)",
                        "Fraud_Cases": "Fraud Cases",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Transaction Amount by Fraud Status")

        if "Transaction_Amount" in analytics_data.columns:
            fig = px.box(
                analytics_data,
                x="Fraud Status",
                y="Transaction_Amount",
                color="Fraud Status",
                color_discrete_map={
                    "Legitimate": "#00D9FF",
                    "Fraudulent": "#A855F7",
                },
                points="outliers",
                title="Transaction Amount Distribution",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#EDF4FF",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "These charts describe labels in the synthetic dataset; "
            "they do not establish real-world fraud rates."
        )


# ---------- Individual prediction ----------
elif page == "Fraud Prediction":
    st.subheader("Check a Transaction")
    st.write(
        "Enter transaction details to estimate the model's fraud probability."
    )

    with st.form("prediction_form"):
        amount = st.number_input(
            "Transaction Amount", min_value=0.01, value=100.0, step=10.0
        )
        balance = st.number_input(
            "Account Balance", min_value=0.0, value=10000.0, step=100.0
        )
        previous_fraud = st.selectbox(
            "Previous Fraudulent Activity", [0, 1]
        )
        daily_count = st.number_input(
            "Daily Transaction Count", min_value=0, value=5, step=1
        )
        card_age = st.number_input(
            "Card Age (days)", min_value=0, value=365, step=30
        )
        transaction_type = st.selectbox(
            "Transaction Type",
            ["Online", "POS", "ATM", "Bank Transfer"],
        )
        device_type = st.selectbox(
            "Device Type", ["Mobile", "Laptop", "Tablet"]
        )
        location = st.selectbox(
            "Location", ["New York", "London", "Tokyo", "Other"]
        )
        merchant = st.selectbox(
            "Merchant Category",
            ["Retail", "Food", "Travel", "Online", "Other"],
        )
        card_type = st.selectbox(
            "Card Type", ["Credit", "Debit", "Other"]
        )
        submitted = st.form_submit_button(
            "Analyze Transaction", use_container_width=True
        )

    
    if submitted:
        from datetime import datetime

        # Calculate the additional features required by the saved model.
        now = datetime.now()
        txn_to_balance_pct = (
            (amount / balance) * 100 if balance > 0 else 0.0
        )

        sample = {
            "Transaction_Amount": amount,
            "Transaction_Type": transaction_type,
            "Account_Balance": balance,
            "Device_Type": device_type,
            "Location": location,
            "Merchant_Category": merchant,
            "Previous_Fraudulent_Activity": previous_fraud,
            "Daily_Transaction_Count": daily_count,
            "Card_Type": card_type,
            "Card_Age": card_age,
            "Txn_to_Balance_Pct": txn_to_balance_pct,
            "Year": now.year,
            "Month": now.month,
            "Day": now.day,
            "DayOfWeek": now.weekday(),
            "Hour": now.hour,
        }

        required = model_bundle["features"]
        row = pd.DataFrame([sample])[required]

        try:
            probability = float(model.predict_proba(row)[0][1])
            prediction = probability >= threshold

            
            st.metric(
                    "Estimated Fraud Probability",
                    f"{probability:.1%}",
                )
            
            if probability >= threshold:
                    st.error(
                        "⚠️ HIGH RISK: This transaction was flagged "
                        "by the fraud detection model."
                    )
            elif probability >= threshold * 0.7:
                st.warning(
                        "🔎 ELEVATED RISK: Review this transaction "
                        "before taking action."
                    )
            else:
                
                st.success(
                        "✅ LOWER MODEL-ESTIMATED RISK: No fraud "
                        "alert was triggered at the configured threshold."
                    )

            st.caption(
                    "This is a model estimate, not proof of fraud. "
                    "Review transaction details before making decisions."
                )


            gauge_color = (
                    "#A855F7" if prediction
                    else "#00D9FF"
                )

                
            gauge_fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        number={"suffix": "%"},
                        title={"text": "Transaction Risk Score"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": gauge_color},
                            "bgcolor": "#11182D",
                            "bordercolor": "#34305F",
                            "steps": [
                                {"range": [0, 30], "color": "#123747"},
                                {"range": [30, 60], "color": "#38245D"},
                                {"range": [60, 100], "color": "#51234F"},
                            ],
                            "threshold": {
                                "line": {
                                    "color": "#00D9FF",
                                    "width": 4,
                                },
                                "thickness": 0.8,
                                "value": threshold * 100,
                            },
                        },
                    )
                )

                
            st.plotly_chart(
                gauge_fig,
                use_container_width=True,
            )

                
            st.divider()
            st.subheader("📥 Download Prediction Report")

            report = pd.DataFrame([{
                    "Prediction": (
                        "Fraudulent" if prediction
                        else "Legitimate"
                    ),
                    "Fraud_Probability": round(
                        probability * 100, 2
                    ),
                    "Decision_Threshold": round(
                        threshold * 100, 2
                    ),
                    "Transaction_Amount": amount,
                    "Account_Balance": balance,
                    "Transaction_Type": transaction_type,
                    "Device_Type": device_type,
                }])

            st.download_button(
                    label="⬇️ Download Report (CSV)",
                    data=report.to_csv(index=False),
                    file_name="fraud_prediction_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


           


            if prediction:
                st.error("⚠️ Potential Fraud — review this transaction.")
            else:
                st.success("✓ Classified as Legitimate by the model.")

                st.caption(
                "This is a model estimate, not a definitive determination "
                "of fraud."
            )
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")


# ---------- Transaction Explorer ----------

elif page == "Transaction Explorer":
    st.title("🔎 Transaction Explorer")
    st.caption("Filter, inspect, and export transaction records.")

    explorer_df = data.copy()

    status = st.selectbox(
        "Transaction Status",
        ["All", "Legitimate", "Fraudulent"],
    )

    if status == "Legitimate":
        explorer_df = explorer_df[explorer_df["Fraud_Label"] == 0]
    elif status == "Fraudulent":
        explorer_df = explorer_df[explorer_df["Fraud_Label"] == 1]

    if "Transaction_Type" in explorer_df.columns:
        transaction_types = ["All"] + sorted(
            explorer_df["Transaction_Type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_type = st.selectbox(
            "Transaction Type",
            transaction_types,
        )

        if selected_type != "All":
            explorer_df = explorer_df[
                explorer_df["Transaction_Type"] == selected_type
            ]

    st.metric("Matching Transactions", f"{len(explorer_df):,}")

    st.dataframe(
        explorer_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Filtered Transactions",
        data=explorer_df.to_csv(index=False),
        file_name="filtered_transactions.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------- Model information ----------
elif page == "Model Information":
    st.subheader("Model Information")
    st.write("Saved model: `models/fraud_detection_model.joblib`")
    st.metric("Classification Threshold", f"{threshold:.2f}")
    st.write("Features stored with model:")
    st.code("\n".join(feature_names) if feature_names else "Not stored in bundle")

    st.info(
        "Evaluation on the held-out synthetic test set: "
        "Accuracy 88.94%, Precision 34.61%, Recall 44.57%, "
        "F1-score 38.96%, ROC-AUC 74.37%. "
        "Results on synthetic data may not reflect real-world fraud detection."
    )
