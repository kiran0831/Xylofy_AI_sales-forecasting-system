import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    return df

@st.cache_data
def get_monthly(df):
    daily = df.groupby("Order Date")["Sales"].sum().reset_index()
    monthly = daily.set_index("Order Date").resample("MS")["Sales"].sum()
    return monthly

@st.cache_data
def get_weekly(df):
    daily = df.groupby("Order Date")["Sales"].sum().reset_index()
    weekly = daily.set_index("Order Date").resample("W")["Sales"].sum()
    return weekly

@st.cache_resource
def fit_sarima(series):
    model = SARIMAX(series, order=(2, 1, 0), seasonal_order=(1, 0, 0, 12),
                     enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)

@st.cache_data
def get_anomalies(_weekly):
    weekly_df = _weekly.to_frame(name="Sales").copy()
    iso = IsolationForest(contamination=0.08, random_state=42)
    weekly_df["iso_anomaly"] = iso.fit_predict(weekly_df[["Sales"]])
    weekly_df["iso_anomaly"] = weekly_df["iso_anomaly"].map({1: 0, -1: 1})

    roll_mean = weekly_df["Sales"].rolling(8, center=True, min_periods=4).mean()
    roll_std = weekly_df["Sales"].rolling(8, center=True, min_periods=4).std()
    weekly_df["zscore"] = (weekly_df["Sales"] - roll_mean) / roll_std
    weekly_df["z_anomaly"] = (weekly_df["zscore"].abs() > 2).astype(int)
    return weekly_df

@st.cache_data
def get_clusters(df):
    sub_agg = df.groupby("Sub-Category").agg(
        total_sales=("Sales", "sum"),
        avg_order_value=("Sales", "mean"),
    ).reset_index()

    yearly_sub = df.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack()
    growth = ((yearly_sub[2018] - yearly_sub[2015]) / yearly_sub[2015] * 100).rename("growth_rate")

    monthly_sub_ts = df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="MS")])["Sales"].sum().reset_index()
    volatility = monthly_sub_ts.groupby("Sub-Category")["Sales"].std().rename("volatility")

    sub_agg = sub_agg.set_index("Sub-Category").join(growth).join(volatility).reset_index().dropna()

    features = ["total_sales", "growth_rate", "volatility", "avg_order_value"]
    X_scaled = StandardScaler().fit_transform(sub_agg[features])

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    sub_agg["cluster"] = km.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    sub_agg["pca1"], sub_agg["pca2"] = X_pca[:, 0], X_pca[:, 1]

    label_map = {}
    means = sub_agg.groupby("cluster")[["growth_rate", "total_sales"]].mean()
    for c in means.index:
        g, v = means.loc[c, "growth_rate"], means.loc[c, "total_sales"]
        if g < 0:
            label_map[c] = "Declining Demand"
        elif g > 200:
            label_map[c] = "Growing Demand, High Value"
        elif v > sub_agg["total_sales"].median() * 1.5:
            label_map[c] = "High Volume, Stable Demand"
        else:
            label_map[c] = "Low Volume, Steady"
    sub_agg["cluster_label"] = sub_agg["cluster"].map(label_map)
    return sub_agg


df = load_data()
monthly = get_monthly(df)
weekly = get_weekly(df)

st.title("📊 Sales Forecasting & Demand Intelligence System")
page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)

if page == "Sales Overview":
    st.header("Sales Overview Dashboard")

    yearly_totals = df.groupby("Year")["Sales"].sum().reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(yearly_totals, x="Year", y="Sales", title="Total Sales by Year",
                     text_auto=".2s", color="Year")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(monthly.reset_index(), x="Order Date", y="Sales",
                       title="Monthly Sales Trend", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales by Region & Category")
    col1, col2 = st.columns(2)
    with col1:
        region_filter = st.multiselect("Filter Region(s)", sorted(df["Region"].unique()),
                                        default=sorted(df["Region"].unique()))
    with col2:
        cat_filter = st.multiselect("Filter Category(s)", sorted(df["Category"].unique()),
                                     default=sorted(df["Category"].unique()))

    filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(cat_filter)]
    agg = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig = px.bar(agg, x="Region", y="Sales", color="Category", barmode="group",
                 title="Sales by Region and Category")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Forecast Explorer":
    st.header("Forecast Explorer")

    dim = st.selectbox("Select dimension", ["Category", "Region"])
    options = sorted(df[dim].unique())
    choice = st.selectbox(f"Select {dim}", options)
    horizon = st.select_slider("Forecast horizon (months ahead)", options=[1, 2, 3], value=3)

    sub = df[df[dim] == choice]
    monthly_sub = sub.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"].sum().asfreq("MS").fillna(0)

    train = monthly_sub[:-3]
    test = monthly_sub[-3:]
    fit = fit_sarima(train)
    fc = fit.get_forecast(steps=horizon)
    pred = fc.predicted_mean
    conf = fc.conf_int()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_sub.index, y=monthly_sub.values, name="Actual", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name="Forecast", mode="lines+markers", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=conf.index, y=conf.iloc[:, 0], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=conf.index, y=conf.iloc[:, 1], fill="tonexty", line=dict(width=0),
                              name="Confidence Interval", fillcolor="rgba(255,0,0,0.15)"))
    fig.update_layout(title=f"SARIMA Forecast — {choice} ({dim}), next {horizon} month(s)")
    st.plotly_chart(fig, use_container_width=True)

    val_fit = fit_sarima(train)
    val_pred = val_fit.get_forecast(steps=3).predicted_mean
    mae = np.mean(np.abs(test.values - val_pred.values))
    rmse = np.sqrt(np.mean((test.values - val_pred.values) ** 2))

    c1, c2 = st.columns(2)
    c1.metric("MAE (on last 3 known months)", f"${mae:,.0f}")
    c2.metric("RMSE (on last 3 known months)", f"${rmse:,.0f}")

    st.dataframe(pred.rename("Forecasted Sales").to_frame().style.format("${:,.0f}"))

elif page == "Anomaly Report":
    st.header("Anomaly Report")

    weekly_anom = get_anomalies(weekly)
    iso_dates = weekly_anom[weekly_anom["iso_anomaly"] == 1]
    z_dates = weekly_anom[weekly_anom["z_anomaly"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_anom.index, y=weekly_anom["Sales"], name="Weekly Sales", mode="lines"))
    fig.add_trace(go.Scatter(x=iso_dates.index, y=iso_dates["Sales"], name="Isolation Forest anomaly",
                              mode="markers", marker=dict(color="red", symbol="x", size=10)))
    fig.add_trace(go.Scatter(x=z_dates.index, y=z_dates["Sales"], name="Z-score anomaly",
                              mode="markers", marker=dict(color="orange", size=10, symbol="circle-open")))
    fig.update_layout(title="Weekly Sales Anomalies")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected anomaly dates")
    tab1, tab2 = st.tabs(["Isolation Forest", "Z-Score"])
    with tab1:
        st.dataframe(iso_dates[["Sales"]].style.format("${:,.0f}"))
    with tab2:
        st.dataframe(z_dates[["Sales", "zscore"]].style.format({"Sales": "${:,.0f}", "zscore": "{:.2f}"}))

elif page == "Product Demand Segments":
    st.header("Product Demand Segments")

    clusters = get_clusters(df)

    fig = px.scatter(clusters, x="pca1", y="pca2", color="cluster_label", text="Sub-Category",
                      title="Product Sub-Category Clusters (PCA projection)", size="total_sales")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-categories by cluster")
    display_cols = ["Sub-Category", "cluster_label", "total_sales", "growth_rate", "volatility", "avg_order_value"]
    st.dataframe(
        clusters[display_cols].sort_values("cluster_label").style.format({
            "total_sales": "${:,.0f}", "growth_rate": "{:.1f}%",
            "volatility": "${:,.0f}", "avg_order_value": "${:,.0f}",
        })
    )
