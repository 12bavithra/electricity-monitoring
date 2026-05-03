import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1) Load the dataset
df = pd.read_csv("cleaned_daily_data.csv")

# (Optional) Convert date column to datetime for better readability
df["date"] = pd.to_datetime(df["date"])

# 2) Apply Isolation Forest for anomaly detection
# We use only the numeric usage column as input.
model = IsolationForest(
    n_estimators=100,      # number of trees
    contamination=0.05,    # expected % of anomalies (adjust if needed)
    random_state=42
)

# Fit model and predict:
# 1 = normal point, -1 = anomaly
df["anomaly"] = model.fit_predict(df[["total_daily_usage"]])

# 3) Add explanation column
# Simple thresholds based on average usage
mean_usage = df["total_daily_usage"].mean()
high_threshold = mean_usage * 1.5
low_threshold = mean_usage * 0.5

def get_explanation(row):
    usage = row["total_daily_usage"]
    anomaly_flag = row["anomaly"]

    if anomaly_flag == 1:
        return "Normal usage"

    # anomaly_flag == -1
    if usage > high_threshold:
        return "High usage spike detected (possible heavy appliance usage)"
    elif usage <= 0 or usage < low_threshold:
        return "Unusual drop in usage (possible outage or missing data)"
    else:
        return "Irregular usage pattern detected"

df["explanation"] = df.apply(get_explanation, axis=1)

# 5) Print only anomaly rows with explanations
anomaly_rows = df[df["anomaly"] == -1][["date", "total_daily_usage", "anomaly", "explanation"]]
print("Anomaly rows with explanations:")
print(anomaly_rows)

# 6) Plot usage with anomalies highlighted
# Sort by date so the line chart is in time order
df = df.sort_values("date")

# Separate normal and anomaly data
normal_days = df[df["anomaly"] == 1]
anomaly_days = df[df["anomaly"] == -1]

plt.figure(figsize=(12, 6))

# 1) Daily usage as line graph (normal in blue)
plt.plot(
    normal_days["date"],
    normal_days["total_daily_usage"],
    color="blue",
    label="Normal"
)

# 3) Anomaly days as red scatter points
plt.scatter(
    anomaly_days["date"],
    anomaly_days["total_daily_usage"],
    color="red",
    label="Anomaly",
    zorder=3
)

# 4) Titles and labels
plt.title("Daily Electricity Usage with Detected Anomalies")
plt.xlabel("Date")
plt.ylabel("Usage")

# 5) Legend
plt.legend()

# 6) Rotate x-axis labels for readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
