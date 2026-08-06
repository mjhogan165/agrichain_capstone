import streamlit as st
from src.graph import app

st.title("AgriChain Multi-Agent Resolution System")
st.write("Submit a complaint to see the full agent pipeline in action.")

complaint_text = st.text_area("Complaint text")

if st.button("Process Complaint"):
    if complaint_text.strip():
        with st.spinner("Running multi-agent pipeline..."):
            result = app.invoke({"complaint_text": complaint_text})

        st.header("Severity Assessment")
        st.write(f"**Severity:** {result.get('severity', 'N/A')}")
        st.write(result.get("severity_reasoning", ""))

        st.header("Category")
        st.write(f"**Predicted category:** {result.get('predicted_category', 'N/A')}")

        st.header("Retrieved Documents")
        for doc in result.get("retrieved_documents", []):
            st.write(f"- {doc.get('doc_id', 'N/A')}")
            st.write(f"  **Type:** {doc.get('doc_type', 'N/A')}")
            st.write(f"  **Score:** {doc.get('score', 0):.3f}")
            st.write(f"  **Content:** {doc.get('content', 'N/A')}")

        st.header("Resolution Plan")
        plan = result.get("resolution_plan", {})
        st.write(f"**Summary:** {plan.get('plan_summary', 'N/A')}")
        st.write("**Steps:**")
        for step in plan.get("steps", []):
            st.write(f"- {step}")

        st.header("Customer Response")
        st.write(f"**Response:** {result.get('customer_response', 'N/A')}")

        st.header("Escalation Status")
        if result.get("escalate"):
            st.error(f"⚠️ Escalation needed: {result.get('escalation_reason', 'N/A')}")
        else:
            st.success("No escalation needed.")
    else:
        st.warning("Please enter a complaint first.")
