import json
import os
import pandas as pd
from src.agents.analyzer import analyze_severity

DEMO_COMPLAINT_IDS = [
    "CMP-000050",
    "CMP-000005",
    "CMP-000022",
    "CMP-000008",
    "CMP-000009",
    "CMP-000011",
    "CMP-000013",
    "CMP-000007",
    "CMP-000010",
]

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

df = pd.read_csv("data/raw/complaints_test.csv")

demo_df = df[df["complaint_id"].isin(DEMO_COMPLAINT_IDS)]

# Insert a critical and a low severity complaint to make testing clear
critical_complaint = {
    "complaint_id": "SYN-000001",
    "complaint_text": (
        "We've had multiple customer reports of illness after purchasing "
        "apples from order ORD-449213. One customer was hospitalized. We "
        "need to know if this batch is contaminated immediately. This is "
        "a serious liability issue for us and we may need to pull the "
        "product from our shelves."
    ),
}

low_complaint = {
    "complaint_id": "SYN-000002",
    "complaint_text": (
        "Delivery for order ORD-118204 arrived about 20 minutes earlier "
        "than our scheduled window. Not a big deal, just wanted it noted "
        "so future deliveries stick closer to the window."
    ),
}


synthetic_complaints = [critical_complaint, low_complaint]

results = []

for _, row in demo_df.iterrows():
    print(f"Analyzing {row['complaint_id']}...")

    state = {"complaint_text": row["complaint_text"]}
    state = analyze_severity(state)  # type: ignore

    results.append(
        {
            "complaint_id": row["complaint_id"],
            "complaint_text": row["complaint_text"],
            "severity": state.get("severity"),
            "severity_reasoning": state.get("severity_reasoning"),
            "predicted_category": state.get("predicted_category"),
        }
    )

for complaint in synthetic_complaints:
    print(f"Analyzing {complaint['complaint_id']}...")

    state = {"complaint_text": complaint["complaint_text"]}
    state = analyze_severity(state)  # type: ignore
    results.append(
        {
            "complaint_id": complaint["complaint_id"],
            "complaint_text": complaint["complaint_text"],
            "severity": state.get("severity"),
            "severity_reasoning": state.get("severity_reasoning"),
            "predicted_category": state.get("predicted_category"),
        }
    )
print("\nQueue summary:")
for r in results:
    print(f"  {r['complaint_id']:<12} {r['severity']:<10} {r['predicted_category']}")

# Sort by severity
results.sort(key=lambda r: SEVERITY_ORDER[r["severity"]])

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/queue_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nWrote data/processed/queue_results.json")
