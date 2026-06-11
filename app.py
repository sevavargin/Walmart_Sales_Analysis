import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

# Настройки страницы
st.set_page_config(
    page_title="Walmart Sales Analysis",
    page_icon="📊",
    layout="wide"
)

# Заголовок
st.title("Walmart Sales Analysis")
@st.cache_data
def load_data():
    df = pd.read_csv("data/Walmart.csv")

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d-%m-%Y"
    )

    return df


df = load_data()

st.sidebar.header("Filters")

selected_store = st.sidebar.selectbox(
    "Select Store",
    ["All Stores"] + sorted(df["Store"].unique().tolist())
)

if selected_store != "All Stores":
    filtered_df = df[df["Store"] == selected_store]
else:
    filtered_df = df.copy()

# KPI
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Sales",
        f"${filtered_df['Weekly_Sales'].sum():,.0f}"
    )

with col2:
    st.metric(
        "Average Sales",
        f"${filtered_df['Weekly_Sales'].mean():,.0f}"
    )

with col3:
    st.metric(
        "Stores",
        filtered_df["Store"].nunique()
    )


tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "EDA",
    "Hypotheses",
    "Correlation"
])

# Таблица
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head())

with tab2:
# График продаж во времени
    st.subheader("Weekly Sales Trend")

    sales_trend = (
        filtered_df.groupby("Date")["Weekly_Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        sales_trend,
        x="Date",
        y="Weekly_Sales",
        title="Weekly Sales Trend"
    )

    st.plotly_chart(fig, use_container_width=True)


    #Кварталы

    filtered_df = filtered_df.copy()
    filtered_df["Quarter"] = filtered_df["Date"].dt.quarter

    st.subheader("Average Sales by Quarter")

    quarter_sales = (
        filtered_df.groupby("Quarter")["Weekly_Sales"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        quarter_sales,
        x="Quarter",
        y="Weekly_Sales",
        title="Average Sales by Quarter"
    )

    st.plotly_chart(fig, use_container_width=True)

    #Топ 10

    st.subheader("Top 10 Stores by Average Sales")

    top_stores = (
        df.groupby("Store")["Weekly_Sales"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
    top_stores,
    x="Store",
    y="Weekly_Sales",
    color="Weekly_Sales",
    color_continuous_scale="Greens",
    text_auto=".2s",
    title="Top 10 Stores by Average Sales"
    )

    fig.update_layout(
        title_x=0.5,
        height=600
    )

    fig.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="outside"
    )


    st.plotly_chart(fig, use_container_width=True)



#Гипотеза 1
with tab3:

    st.subheader(
        "Hypothesis 1: Holiday Weeks vs Non-Holiday Weeks"
    )

    holiday_sales = filtered_df[
        filtered_df["Holiday_Flag"] == 1
    ]["Weekly_Sales"]

    non_holiday_sales = filtered_df[
        filtered_df["Holiday_Flag"] == 0
    ]["Weekly_Sales"]

    stat, p_value = mannwhitneyu(
        holiday_sales,
        non_holiday_sales,
        alternative="two-sided"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Mann-Whitney Statistic",
            f"{stat:.0f}"
        )

    with col2:
        st.metric(
            "p-value",
            f"{p_value:.6f}"
        )
        
    if p_value < 0.05:
        st.success(
            f"Statistically significant difference found (p = {p_value:.4f})"
        )
    else:
        st.warning(
            f"No statistically significant difference found (p = {p_value:.4f})"
            )
    # график гипотезы 1

    fig = px.box(
        filtered_df,
        x="Holiday_Flag",
        y="Weekly_Sales",
        color="Holiday_Flag",
        title="Sales Distribution: Holiday vs Non-Holiday Weeks"
    )

    fig.update_layout(
        xaxis_title="Holiday Flag",
        yaxis_title="Weekly Sales"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


#Гиптеза 2
    from scipy.stats import spearmanr

    st.subheader("Hypothesis 2: Unemployment vs Sales")

    hyp_df = filtered_df[
        ["Unemployment", "Weekly_Sales"]
    ].dropna()

    corr, p = spearmanr(
        hyp_df["Unemployment"],
        hyp_df["Weekly_Sales"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Spearman Correlation",
            f"{corr:.4f}"
        )

    with col2:
        st.metric(
            "p-value",
            f"{p:.8f}"
        )

#Скаттер плот

    fig = px.scatter(
        filtered_df,
        x="Unemployment",
        y="Weekly_Sales",
        trendline="ols",
        opacity=0.5,
        color="Holiday_Flag",
        title="Impact of Unemployment on Weekly Sales"
    )

    fig.update_layout(
        title_x=0.5,
        xaxis_title="Unemployment Rate (%)",
        yaxis_title="Weekly Sales ($)",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)



#Матрица коррлеяций

with tab4:
    st.subheader("Correlation Matrix")
    corr_matrix = df.select_dtypes(include="number").corr()

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title(
        "Correlation Matrix",
        fontsize=16,
        fontweight="bold",
        pad=20
    )

    st.pyplot(fig)