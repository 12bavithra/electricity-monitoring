import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import time
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Real-Time Electricity Monitoring", layout="wide")

st.title("Real-Time Electricity Monitoring")
st.write("Upload a CSV file with columns: `date`, `total_daily_usage`.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_columns = {"date", "total_daily_usage"}
    if not required_columns.issubset(df.columns):
        st.error("CSV must contain columns: date, total_daily_usage")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["total_daily_usage"] = pd.to_numeric(df["total_daily_usage"], errors="coerce")
    df = df.dropna(subset=["date", "total_daily_usage"]).copy()
    df = df.sort_values("date")

    st.subheader("Uploaded Data")
    st.dataframe(df, use_container_width=True)

    start_simulation = st.button("Start Simulation")

    chart_placeholder = st.empty()
    alert_placeholder = st.empty()
    current_data_placeholder = st.empty()

    if start_simulation:
        streamed_rows = []

        for i in range(len(df)):
            current_row = df.iloc[i]
            current_usage = current_row["total_daily_usage"]
            mean_so_far = df.iloc[: i + 1]["total_daily_usage"].mean()
            high_threshold = mean_so_far * 1.4
            low_threshold = mean_so_far * 0.6

            # Fit model on data received so far and predict current point.
            if i >= 1:
                current_data_so_far = df.iloc[: i + 1][["total_daily_usage"]]
                model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
                model.fit(current_data_so_far)
                current_anomaly = int(model.predict([[current_usage]])[0])
            else:
                # First point is treated as normal to keep startup simple.
                current_anomaly = 1

            # Intelligent explanation based on how far usage is from average.
            if current_usage > high_threshold:
                current_explanation = (
                    "High usage spike detected (possible AC / heater / heavy appliance)"
                )
            elif current_usage < low_threshold:
                current_explanation = (
                    "Unusual drop (possible power outage or low usage day)"
                )
            else:
                current_explanation = "Normal household usage"

            # Live alert with explanation message.
            if current_anomaly == -1:
                alert_placeholder.error(f"⚠️ High usage detected\n\n{current_explanation}")
            else:
                alert_placeholder.success(current_explanation)

            streamed_rows.append(
                {
                    "date": current_row["date"],
                    "total_daily_usage": current_usage,
                    "anomaly": current_anomaly,
                    "explanation": current_explanation,
                }
            )

            live_df = pd.DataFrame(streamed_rows)

            current_data_placeholder.subheader("Current Incoming Data")
            current_data_placeholder.dataframe(
                live_df.tail(1),
                use_container_width=True,
            )

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(live_df["date"], live_df["total_daily_usage"], color="blue", label="Usage")

            live_anomalies = live_df[live_df["anomaly"] == -1]
            ax.scatter(
                live_anomalies["date"],
                live_anomalies["total_daily_usage"],
                color="red",
                label="Anomaly",
                zorder=3,
            )

            ax.set_title("Live Electricity Usage")
            ax.set_xlabel("Date")
            ax.set_ylabel("Usage")
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            chart_placeholder.pyplot(fig)
            plt.close(fig)
            time.sleep(1)

        st.subheader("Anomaly Rows")
        final_anomalies = live_df[live_df["anomaly"] == -1]
        st.dataframe(final_anomalies, use_container_width=True)
    else:
        st.info("Click 'Start Simulation' to begin live monitoring.")
else:
    st.info("Please upload a CSV file to begin.")
