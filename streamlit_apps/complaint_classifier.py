import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path so `from src...` imports work,
# regardless of where Streamlit's working directory happens to be
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.classify import predict_category

st.title("AgriChain Complaint Classifier")
st.write("Enter a complaint below to see its predicted category.")

complaint_text = st.text_area("Complaint text")

if st.button("Classify"):
    if complaint_text.strip():
        category = predict_category(complaint_text)
        st.success(f"Predicted category: {category}")
    else:
        st.warning("Please enter some complaint text first.")
