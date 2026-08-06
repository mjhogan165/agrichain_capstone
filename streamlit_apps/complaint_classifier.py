import streamlit as st
from pathlib import Path

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
