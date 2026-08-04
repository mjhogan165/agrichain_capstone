import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("AgriChain EDA Dashboard")


train_df = pd.read_csv("data/raw/complaints_train.csv")
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])


def plot_bar_chart(counts, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


st.header("Complaints by Category")
category_counts = train_df["category"].value_counts()
plot_bar_chart(
    category_counts, "Complaint Volume by Category", "Category", "Number of Complaints"
)

st.header("Complaints by Region")
region_counts = train_df["region"].value_counts()
plot_bar_chart(
    region_counts, "Complaint Volume by Region", "Region", "Number of Complaints"
)

st.header("Complaints by Channel")
channel_counts = train_df["channel"].value_counts()
plot_bar_chart(
    channel_counts, "Complaint Volume by Channel", "Channel", "Number of Complaints"
)

st.header("Complaints by Priority")
priority_counts = train_df["priority"].value_counts()
plot_bar_chart(
    priority_counts, "Complaint Volume by Priority", "Priority", "Number of Complaints"
)

st.header("Complaints by Product Category")
product_category_counts = train_df["category"].value_counts()
plot_bar_chart(
    product_category_counts,
    "Complaint Volume by Product Category",
    "Product Category",
    "Number of Complaints",
)

st.header("Complaints by Month")
monthly_counts = train_df.set_index("timestamp").resample("ME").size()
fig, ax = plt.subplots(figsize=(10, 5))
monthly_counts.plot(kind="line", marker="o", ax=ax)
ax.set_title("Complaint Volume Over Time (Monthly)")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Complaints")
ax.set_ylim(
    bottom=0
)  # keep this from Task 1 — forces y-axis to start at 0, avoids a misleading trend line
plt.tight_layout()
st.pyplot(fig)
