import json
import streamlit as st
from src.graph import app

st.title("AgriChain Multi-Agent Resolution System")

# Load the precomputed queue (built by scripts/build_queue.py)
with open("data/processed/queue_results.json", encoding="utf-8") as f:
    queue = json.load(f)

st.header("Complaint Queue")

if "selected_complaint" not in st.session_state:
    st.session_state.selected_complaint = None

for item in queue:
    if st.button(
        f"{item['severity'].upper()} — {item['complaint_id']} — {item['predicted_category']}"
    ):
        st.session_state.selected_complaint = item

if st.session_state.selected_complaint:
    st.write(f"Selected: **{st.session_state.selected_complaint['complaint_id']}**")

    if st.button("Run Full Pipeline"):
        with st.spinner("Running multi-agent pipeline..."):
            result = app.invoke(
                {
                    "complaint_text": st.session_state.selected_complaint[
                        "complaint_text"
                    ]
                }
            )
        st.session_state.pipeline_result = result

    if "pipeline_result" in st.session_state:
        result = st.session_state.pipeline_result

        severity = result.get("severity", "unknown")
        category = result.get("predicted_category", "unknown")
        st.subheader(f"{severity.upper()} · {category}")
        if result.get("escalate"):
            st.error(
                f"⚠️ Needs supervisor review: {result.get('escalation_reason', 'N/A')}"
            )
        with st.expander("Why this severity"):
            st.write(result.get("severity_reasoning", "N/A"))
        st.header("Reply to Customer")
        st.code(result.get("customer_response", ""), language=None)
        st.subheader("Internal Notes")
        with st.expander("Resolution Plan"):
            plan = result.get("resolution_plan", {})
            st.write(f"**Summary:** {plan.get('plan_summary', 'N/A')}")
            st.write(f"**Steps:**")
            for step in plan.get("steps", []):
                st.write(f"- {step}")

        with st.expander("Evidence used"):
            docs = result.get("retrieved_documents", [])
            top_docs = sorted(docs, key=lambda d: d.get("score", 0))
            for doc in top_docs:
                st.write(
                    f"**{doc.get('doc_id', 'N/A')}** (score: {doc.get('score', 0):.3f})"
                )
                st.write(doc.get("content", "N/A"))

st.divider()
st.header("New Complaint")
new_complaint_text = st.text_area("Complaint text")

if st.button("Process New Complaint"):
    if new_complaint_text.strip():
        with st.spinner("Running multi-agent pipeline..."):
            new_result = app.invoke({"complaint_text": new_complaint_text})
        st.session_state.new_result = new_result
    else:
        st.warning("Please enter a complaint first.")

if "new_result" in st.session_state:
    result = st.session_state.new_result

    severity = result.get("severity", "unknown")
    category = result.get("predicted_category", "unknown")
    st.subheader(f"**{severity.upper()}** · {category}")
    if result.get("escalate"):
        st.error(
            f"⚠️ Needs supervisor review: {result.get('escalation_reason', 'N/A')}"
        )
    st.header("Reply to Customer")
    st.code(result.get("customer_response", ""), language=None)
    st.subheader("Internal Notes")
    with st.expander("Resolution Plan"):
        st.write(f"**Why this severity:** {result.get('severity_reasoning', 'N/A')}")
        plan = result.get("resolution_plan", {})
        st.write(f"**Summary:** {plan.get('plan_summary', 'N/A')}")
        st.write(f"**Steps:**")
        for step in plan.get("steps", []):
            st.write(f"- {step}")

    with st.expander("Evidence used"):
        docs = result.get("retrieved_documents", [])
        top_docs = sorted(docs, key=lambda d: d.get("score", 0))
        for doc in top_docs:
            st.write(
                f"**{doc.get('doc_id', 'N/A')}** (score: {doc.get('score', 0):.3f})"
            )
            st.write(doc.get("content", "N/A"))
# st.write("Submit a complaint to see the full agent pipeline in action.")

# complaint_text = st.text_area("Complaint text")

# if st.button("Process Complaint"):
#     if complaint_text.strip():
#         with st.spinner("Running multi-agent pipeline..."):
#             result = app.invoke({"complaint_text": complaint_text})

#         st.header("Severity Assessment")
#         st.write(f"**Severity:** {result.get('severity', 'N/A')}")
#         st.write(result.get("severity_reasoning", ""))

#         st.header("Category")
#         st.write(f"**Predicted category:** {result.get('predicted_category', 'N/A')}")

#         st.header("Retrieved Documents")
#         for doc in result.get("retrieved_documents", []):
#             st.write(f"- {doc.get('doc_id', 'N/A')}")
#             st.write(f"  **Type:** {doc.get('doc_type', 'N/A')}")
#             st.write(f"  **Score:** {doc.get('score', 0):.3f}")
#             st.write(f"  **Content:** {doc.get('content', 'N/A')}")

#         st.header("Resolution Plan")
#         plan = result.get("resolution_plan", {})
#         st.write(f"**Summary:** {plan.get('plan_summary', 'N/A')}")
#         st.write("**Steps:**")
#         for step in plan.get("steps", []):
#             st.write(f"- {step}")

#         st.header("Customer Response")
#         st.write(f"**Response:** {result.get('customer_response', 'N/A')}")

#         st.header("Escalation Status")
#         if result.get("escalate"):
#             st.error(f"Escalation needed: {result.get('escalation_reason', 'N/A')}")
#         else:
#             st.success("No escalation needed.")
#     else:
#         st.warning("Please enter a complaint first.")
