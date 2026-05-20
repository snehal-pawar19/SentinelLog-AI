import pandas as pd
from sklearn.ensemble import IsolationForest

# Load logs
df = pd.read_csv("data/auth_logs.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Encode status
df["status_flag"] = df["status"].map({"success": 0, "fail": 1})

# -----------------------------
# Rule-based detection (SOC logic)
# -----------------------------
df["rule_alert"] = "Normal"

# Brute-force: 3+ failures from same user + IP
failures = (
    df[df["status"] == "fail"]
    .groupby(["user", "ip"])
    .size()
)

for (user, ip), count in failures.items():
    if count >= 3:
        df.loc[
            (df["user"] == user) &
            (df["ip"] == ip) &
            (df["status"] == "fail"),
            "rule_alert"
        ] = "BruteForce"

# -----------------------------
# ML-based anomaly detection
# -----------------------------
model = IsolationForest(contamination=0.3, random_state=42)
df["ml_anomaly"] = model.fit_predict(df[["status_flag"]])
df["ml_anomaly"] = df["ml_anomaly"].map({1: "Normal", -1: "Anomaly"})

# -----------------------------
# Final alert decision
# -----------------------------
df["final_alert"] = df.apply(
    lambda x: "ALERT" if x["rule_alert"] != "Normal" or x["ml_anomaly"] == "Anomaly"
    else "Normal",
    axis=1
)

print("\n🚨 SentinelLog-AI — Threat Detection Output\n")
print(df[[
    "timestamp", "user", "ip", "status",
    "rule_alert", "ml_anomaly", "final_alert"
]])