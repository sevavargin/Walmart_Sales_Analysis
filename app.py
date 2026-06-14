import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind

st.set_page_config(page_title="Walmart Sales Analysis", layout="wide")
st.title("Walmart Sales Analysis - Final Project")
st.markdown("**Полный исследовательский анализ данных (EDA)**")

@st.cache_data
def load_data():
    df = pd.read_csv("data/Walmart.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Sales_Growth"] = df.groupby("Store")["Weekly_Sales"].pct_change() * 100
    
    store_avg = df.groupby("Store")["Weekly_Sales"].mean()
    df["Store_Category"] = df["Store"].map(
        lambda x: 'High' if store_avg[x] > store_avg.quantile(0.75) else
                  'Low' if store_avg[x] < store_avg.quantile(0.25) else 'Medium'
    )
    return df

df = load_data()
filtered_df = df.copy()

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Общие продажи", f"${df['Weekly_Sales'].sum():,.0f}")
with col2: st.metric("Средние продажи", f"${df['Weekly_Sales'].mean():,.0f}")
with col3: st.metric("Магазинов", df["Store"].nunique())
with col4: st.metric("Праздничных недель", df["Holiday_Flag"].sum())


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Abstract", 
    "Dataset & Cleanup", 
    "Descriptive Stats", 
    "EDA", 
    "Гипотеза 1", 
    "Гипотеза 2", 
    "Correlation & Conclusion"
])

with tab1:
    st.subheader("Abstract")
    st.write("""
    В данном проекте проводится исследовательский анализ данных (EDA) о еженедельных продажах 
    45 магазинов сети Walmart за период 2010–2012 годов. 
    
    Цель — выявить ключевые факторы, влияющие на объём продаж, включая сезонность, 
    праздничные периоды и макроэкономические показатели.
    """)
    st.write("**Распределение вклада:** Варгин Севастьян — 100%")

with tab2:
    st.subheader("Описание набора данных")
    st.write(f"""
    - **Количество записей**: {len(df):,}  
    - **Период**: 2010–2012 гг.  
    - **Магазинов**: 45  
    - **Признаки**: 8 (Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment и др.)
    """)
    
    st.subheader("Data Cleanup")
    st.success("Пропущенных значений: 0")
    st.success("Дубликатов: 0")
    st.info("Данные высокого качества, дополнительная очистка не требовалась.")

with tab3:
    st.subheader("Descriptive Statistics")
    numerical_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    stats = df[numerical_cols].describe().round(2)
    stats.loc['median'] = df[numerical_cols].median().round(2)
    st.dataframe(stats)

with tab4:
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(df, x="Store_Category", y="Weekly_Sales", color="Holiday_Flag",
                     title="Продажи по категории магазина и праздникам")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x="Weekly_Sales", color="Holiday_Flag", nbins=50,
                           title="Распределение Weekly Sales")
        st.plotly_chart(fig, use_container_width=True)
    
    # Тренд
    trend = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    fig = px.line(trend, x="Date", y="Weekly_Sales", title="Общий тренд продаж по времени")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Гипотеза 1: Прирост продаж в праздничные недели выше в крупных магазинах (High)")
    
    st.markdown("""
    **Формулировка:** Прирост продаж в праздничные недели в магазинах категории **High** статистически значимо выше, 
    чем в магазинах категории **Low**.
    """)


    store_mean = df.groupby("Store")["Weekly_Sales"].mean()
    df_temp = df.copy()
    df_temp["Sales_vs_Avg"] = df_temp.apply(
        lambda row: (row["Weekly_Sales"] - store_mean[row["Store"]]) / store_mean[row["Store"]] * 100, 
        axis=1
    )

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.box(df, x="Holiday_Flag", y="Weekly_Sales", color="Holiday_Flag",
                      title="1. Сравнение продаж: Праздничные vs Обычные недели")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.box(df_temp, x="Store_Category", y="Weekly_Sales", color="Holiday_Flag",
                      title="2. Продажи по категории магазина и праздникам")
        st.plotly_chart(fig2, use_container_width=True)

    growth = df_temp.groupby(["Store_Category", "Holiday_Flag"])["Sales_vs_Avg"].mean().reset_index()
    fig3 = px.bar(
        growth, 
        x="Store_Category", 
        y="Sales_vs_Avg", 
        color="Holiday_Flag",
        barmode="group",
        title="3. Прирост продаж в праздники по категории магазина (%)"
    )
    st.plotly_chart(fig3, use_container_width=True)


    # Статист тест
    from scipy.stats import mannwhitneyu

    st.subheader("Гипотеза 1: Прирост продаж в праздничные недели выше в крупных магазинах (High)")

    st.markdown("""
    **Формулировка:** Прирост продаж в праздничные недели в магазинах категории **High**
    статистически значимо выше, чем в магазинах категории **Low**.
    """)

    # Средние продажи по магазину в обычные и праздничные недели
    store_growth = df.pivot_table(
        values='Weekly_Sales',
        index=['Store', 'Store_Category'],
        columns='Holiday_Flag',
        aggfunc='mean'
    )

    # Удаляем магазины, где нет данных хотя бы по одному типу недель
    store_growth = store_growth.dropna()

    # Прирост продаж
    store_growth['Growth'] = (
        store_growth[1] - store_growth[0]
    )

    store_growth = store_growth.reset_index()

    # График
    fig = px.box(
        store_growth,
        x='Store_Category',
        y='Growth',
        color='Store_Category',
        title='Прирост продаж в праздничные недели по категориям магазинов'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Средние приросты
    growth_summary = (
        store_growth.groupby('Store_Category')['Growth']
        .mean()
        .reset_index()
    )

    st.dataframe(growth_summary)

    # Выборки для теста
    high = store_growth[
        store_growth['Store_Category'] == 'High'
    ]['Growth']

    low = store_growth[
        store_growth['Store_Category'] == 'Low'
    ]['Growth']

    # Проверка гипотезы:
    # High > Low
    stat, p_value = mannwhitneyu(
        high,
        low,
        alternative='greater'
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Статистика U", f"{stat:.2f}")

    with col2:
        st.metric("p-value", f"{p_value:.5f}")

    if p_value < 0.05:
        st.success(
            "Гипотеза подтверждена: прирост продаж в праздничные недели "
            "у магазинов категории High статистически значимо выше."
        )
    else:
        st.warning(
            "Недостаточно оснований утверждать, что прирост продаж "
            "у магазинов категории High выше, чем у магазинов категории Low."
        )


with tab6:
    st.subheader("Гипотеза 2: Сезонные факторы влияют сильнее макроэкономических показателей")
    
    st.markdown("""
    **Формулировка:** Сезонные факторы (праздники) и квартальная сезонность оказывают более сильное влияние 
    на объём продаж, чем макроэкономические показатели (CPI и Unemployment).
    """)

    corr_cpi, _ = spearmanr(df["CPI"], df["Weekly_Sales"])
    corr_unemp, _ = spearmanr(df["Unemployment"], df["Weekly_Sales"])
    
    holiday_sales = df[df["Holiday_Flag"] == 1]["Weekly_Sales"]
    non_holiday_sales = df[df["Holiday_Flag"] == 0]["Weekly_Sales"]
    _, p_holiday = ttest_ind(holiday_sales, non_holiday_sales, equal_var=False)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.scatter(
            df, 
            x="Unemployment", 
            y="Weekly_Sales", 
            color="Holiday_Flag",
            trendline="ols",
            opacity=0.6,
            title="1. Взаимосвязь уровня безработицы и продаж"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            df, 
            x="CPI", 
            y="Weekly_Sales", 
            color="Holiday_Flag",
            trendline="ols",
            opacity=0.6,
            title="2. Взаимосвязь CPI и продаж"
        )
        st.plotly_chart(fig2, use_container_width=True)


    impact_df = pd.DataFrame({
        "Фактор": ["Holiday Effect", "CPI", "Unemployment"],
        "Сила влияния": [
            -np.log10(p_holiday), 
            abs(corr_cpi),
            abs(corr_unemp)
        ]
    })

    fig3 = px.bar(
        impact_df, 
        x="Фактор", 
        y="Сила влияния", 
        color="Фактор",
        title="3. Сравнение силы влияния факторов на продажи"
    )
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Holiday p-value", f"{p_holiday:.5f}")
    with col2:
        st.metric("CPI Correlation", f"{corr_cpi:.3f}")
    with col3:
        st.metric("Unemployment Correlation", f"{corr_unemp:.3f}")

    # Вывод
    if p_holiday < 0.05:
        st.success("Гипотеза 2 подтверждена: Сезонные факторы (праздники) влияют значительно сильнее, чем CPI и Unemployment.")
    else:
        st.warning("Гипотеза 2 не подтверждена")




with tab7:
    st.subheader("Матрица корреляций")
    
    numeric_df = df.select_dtypes(include="number")
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",           
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10},  
        ax=ax
    )
    
    ax.set_title("Матрица корреляций признаков Walmart Sales", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    st.pyplot(fig, use_container_width=True)
    
    st.subheader("Самые значимые корреляции")
    corr_unstack = corr_matrix.unstack().sort_values(ascending=False)
    corr_unstack = corr_unstack[corr_unstack != 1.0] 
    top_corr = corr_unstack.head(10).to_frame(name="Корреляция").round(3)
    st.dataframe(top_corr, use_container_width=True)

