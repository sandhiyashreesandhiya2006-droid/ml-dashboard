import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# PAGE SETTINGS
st.set_page_config(
    page_title="AI Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# TITLE
st.title("🤖 AI Sales Forecasting Dashboard")
st.markdown("### Machine Learning Based Sales Analysis & Prediction")

st.markdown("---")

# LOAD DATA
df = pd.read_csv("customer_orders_1500.csv")

# DATE CONVERSION
df["Order Date"] = pd.to_datetime(df["Order Date"])

# MONTH COLUMN
df["Month"] = df["Order Date"].dt.month

# KPI CARDS
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = len(df)
avg_sales = df["Sales"].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("💰 Total Sales", f"₹{total_sales:,.2f}")
c2.metric("📈 Total Profit", f"₹{total_profit:,.2f}")
c3.metric("📦 Total Orders", total_orders)
c4.metric("📊 Average Sales", f"₹{avg_sales:,.2f}")

st.markdown("---")

# DATA PREVIEW
st.subheader("📋 Dataset Preview")
st.dataframe(df.head(20))

st.markdown("---")

# MONTHLY SALES TREND
st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    df.groupby("Month")["Sales"]
    .sum()
    .reset_index()
)

fig_sales = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    markers=True,
    title="Monthly Sales Trend"
)

st.plotly_chart(fig_sales, use_container_width=True)

st.markdown("---")

# MONTHLY PROFIT ANALYSIS
st.subheader("💹 Monthly Profit Analysis")

monthly_profit = (
    df.groupby("Month")["Profit"]
    .sum()
    .reset_index()
)

fig_profit = px.bar(
    monthly_profit,
    x="Month",
    y="Profit",
    title="Monthly Profit Analysis"
)

st.plotly_chart(fig_profit, use_container_width=True)

st.markdown("---")

# CATEGORY SALES
st.subheader("🛒 Category Sales Distribution")

category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

fig_category = px.pie(
    category_sales,
    names="Category",
    values="Sales",
    title="Category Wise Sales"
)

st.plotly_chart(fig_category, use_container_width=True)

st.markdown("---")

# REGION SALES
st.subheader("🌍 Region Wise Sales")

region_sales = (
    df.groupby("Region")["Sales"]
    .sum()
    .reset_index()
)

fig_region = px.bar(
    region_sales,
    x="Region",
    y="Sales",
    title="Region Wise Sales"
)

st.plotly_chart(fig_region, use_container_width=True)

st.markdown("---")

# TOP CUSTOMERS
st.subheader("🏆 Top 10 Customers")

top_customers = (
    df.groupby("Customer Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(top_customers)

st.markdown("---")

# PAYMENT METHOD ANALYSIS
st.subheader("💳 Payment Method Analysis")

payment_sales = (
    df.groupby("Payment Method")["Sales"]
    .sum()
    .reset_index()
)

fig_payment = px.pie(
    payment_sales,
    names="Payment Method",
    values="Sales",
    title="Payment Method Contribution"
)

st.plotly_chart(fig_payment, use_container_width=True)

st.markdown("---")

# MACHINE LEARNING MODEL
st.subheader("🤖 Future Sales Prediction")

X = monthly_sales[["Month"]]
y = monthly_sales["Sales"]

model = LinearRegression()
model.fit(X, y)

future_month = monthly_sales["Month"].max() + 1

prediction = model.predict([[future_month]])[0]

st.success(
    f"📈 Predicted Future Sales: ₹{prediction:,.2f}"
)

st.markdown("---")

# DOWNLOAD DATA
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Dataset",
    data=csv,
    file_name="forecast_data.csv",
    mime="text/csv"
)

st.markdown("---")

st.caption("✨ Developed using Python, Streamlit, Plotly & Scikit-Learn")


