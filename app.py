
"""
NovaMart Marketing Analytics Dashboard - Completed Version
=========================================================
Masters of AI in Business - Data Visualization Assignment

This Streamlit app implements a full analytics dashboard for the
NovaMart marketing dataset. It follows the structure and logic
of the provided starter template, with all TODOs completed and
extra polish for interactivity and insights.

To run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(
    page_title="NovaMart Marketing Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA LOADING (with caching)
# =============================================================================
@st.cache_data
def load_data():
    """Load all datasets from the data/ folder"""
    data = {}
    data_path = "data/"
    try:
        data["campaigns"] = pd.read_csv(f"{data_path}campaign_performance.csv", parse_dates=["date"])
        data["customers"] = pd.read_csv(f"{data_path}customer_data.csv")
        data["products"] = pd.read_csv(f"{data_path}product_sales.csv")
        data["leads"] = pd.read_csv(f"{data_path}lead_scoring_results.csv")
        data["feature_importance"] = pd.read_csv(f"{data_path}feature_importance.csv")
        data["learning_curve"] = pd.read_csv(f"{data_path}learning_curve.csv")
        data["geographic"] = pd.read_csv(f"{data_path}geographic_data.csv")
        data["attribution"] = pd.read_csv(f"{data_path}channel_attribution.csv")
        data["funnel"] = pd.read_csv(f"{data_path}funnel_data.csv")
        data["journey"] = pd.read_csv(f"{data_path}customer_journey.csv")
        data["correlation"] = pd.read_csv(f"{data_path}correlation_matrix.csv", index_col=0)
        return data
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.info("Please ensure all CSV files are in the 'data/' folder")
        return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def format_money_indian(num):
    """Format number in Indian style crores and lakhs"""
    if pd.isna(num):
        return "-"
    if abs(num) >= 1e7:
        return f"₹{num/1e7:.2f} Cr"
    if abs(num) >= 1e5:
        return f"₹{num/1e5:.2f} L"
    return f"₹{num:,.0f}"


def side_nav():
    """Render sidebar and return selected page"""
    st.sidebar.title("NovaMart Dashboard")
    st.sidebar.markdown("Masters of AI in Business\n\n**Data Visualization Assignment**")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Executive Overview",
            "📈 Campaign Analytics",
            "👥 Customer Insights",
            "📦 Product Performance",
            "🗺️ Geographic Analysis",
            "🎯 Attribution & Funnel",
            "🤖 ML Model Evaluation",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Built for NovaMart Marketing Analytics dataset")
    return page


# =============================================================================
# PAGE: EXECUTIVE OVERVIEW
# =============================================================================
def page_executive_overview(data):
    st.title("🏠 Executive Overview")
    st.markdown("High-level summary of NovaMart performance.")

    campaigns = data["campaigns"]
    customers = data["customers"]

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    total_revenue = campaigns["revenue"].sum()
    total_conversions = campaigns["conversions"].sum()
    total_spend = campaigns["spend"].sum()
    overall_roas = total_revenue / total_spend if total_spend > 0 else np.nan

    with col1:
        st.metric("Total Revenue", format_money_indian(total_revenue))
    with col2:
        st.metric("Total Conversions", f"{total_conversions:,}")
    with col3:
        st.metric("Total Spend", format_money_indian(total_spend))
    with col4:
        st.metric("Overall ROAS", f"{overall_roas:.2f}x")

    st.markdown("---")

    # Revenue trend (monthly) + channel comparison
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Revenue Trend by Month & Channel")

        temp = campaigns.copy()
        temp["month_year"] = temp["date"].dt.to_period("M").dt.to_timestamp()
        metric = st.selectbox("Metric", ["revenue", "conversions", "spend"], index=0)

        agg = (
            temp.groupby(["month_year", "channel"], as_index=False)[metric]
            .sum()
            .sort_values("month_year")
        )
        fig = px.line(
            agg,
            x="month_year",
            y=metric,
            color="channel",
            markers=True,
            title=f"Monthly {metric.title()} by Channel",
        )
        fig.update_layout(xaxis_title="Month", yaxis_title=metric.title())
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Channel Share of Revenue")
        channel_rev = campaigns.groupby("channel", as_index=False)["revenue"].sum()
        fig = px.bar(
            channel_rev.sort_values("revenue", ascending=True),
            x="revenue",
            y="channel",
            orientation="h",
            text_auto=".2s",
            labels={"revenue": "Revenue", "channel": "Channel"},
        )
        fig.update_layout(title="Total Revenue by Channel")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Customer Snapshot")

    # Simple customer age distribution
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(
            customers,
            x="age",
            nbins=20,
            title="Customer Age Distribution",
            labels={"age": "Age"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        seg = customers.groupby("customer_segment", as_index=False)["lifetime_value"].mean()
        fig = px.bar(
            seg,
            x="customer_segment",
            y="lifetime_value",
            title="Average Lifetime Value by Segment",
            labels={"customer_segment": "Customer Segment", "lifetime_value": "Avg LTV"},
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: CAMPAIGN ANALYTICS
# =============================================================================
def page_campaign_analytics(data):
    st.title("📈 Campaign Analytics")
    st.markdown("Deep-dive into campaign performance across channels, regions and time.")

    campaigns = data["campaigns"]
    correlation = data["correlation"]

    tab1, tab2, tab3 = st.tabs(["Time Series", "Comparisons", "Calendars & Correlation"])

    # ---- Time Series Tab ----
    with tab1:
        st.subheader("Daily / Weekly / Monthly Revenue Trend")

        min_date, max_date = campaigns["date"].min(), campaigns["date"].max()
        date_range = st.date_input(
            "Select Date Range",
            (min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, (list, tuple)):
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date

        df = campaigns[(campaigns["date"] >= pd.to_datetime(start_date)) & (campaigns["date"] <= pd.to_datetime(end_date))]

        agg_level = st.radio("Aggregation Level", ["Daily", "Weekly", "Monthly"], horizontal=True)

        if agg_level == "Daily":
            df["period"] = df["date"]
        elif agg_level == "Weekly":
            df["period"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
        else:
            df["period"] = df["date"].dt.to_period("M").dt.to_timestamp()

        channels = st.multiselect(
            "Filter Channels",
            sorted(df["channel"].unique().tolist()),
            default=sorted(df["channel"].unique().tolist()),
        )
        df = df[df["channel"].isin(channels)]

        metric = st.selectbox("Metric", ["revenue", "conversions", "spend"], index=0)
        agg = df.groupby(["period", "channel"], as_index=False)[metric].sum()

        fig = px.line(
            agg,
            x="period",
            y=metric,
            color="channel",
            title=f"{metric.title()} over Time ({agg_level}) by Channel",
        )
        fig.update_layout(xaxis_title="Date", yaxis_title=metric.title())
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Weekly Conversions by Channel (Stacked Area)")
        weekly = campaigns.copy()
        weekly["week"] = weekly["date"].dt.to_period("W").apply(lambda r: r.start_time)
        conv_week = weekly.groupby(["week", "channel"], as_index=False)["conversions"].sum()
        fig = px.area(
            conv_week,
            x="week",
            y="conversions",
            color="channel",
            title="Weekly Conversions by Channel (Stacked)",
        )
        fig.update_layout(xaxis_title="Week", yaxis_title="Conversions")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Comparisons Tab ----
    with tab2:
        st.subheader("Channel Performance Comparison")

        metric = st.selectbox(
            "Select Metric",
            ["revenue", "conversions", "spend", "roas", "ctr", "conversion_rate"],
            index=0,
        )
        channel_agg = campaigns.groupby("channel", as_index=False)[metric].mean() if metric in ["ctr", "conversion_rate", "roas"] else campaigns.groupby("channel", as_index=False)[metric].sum()

        fig = px.bar(
            channel_agg.sort_values(metric, ascending=True),
            x=metric,
            y="channel",
            orientation="h",
            labels={metric: metric.title(), "channel": "Channel"},
            title=f"Channel-wise {metric.title()}",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Regional Revenue by Quarter")
        region_years = sorted(campaigns["year"].unique().tolist())
        year_sel = st.selectbox("Select Year", region_years, index=len(region_years) - 1)
        reg = campaigns[campaigns["year"] == year_sel]
        reg_agg = reg.groupby(["region", "quarter"], as_index=False)["revenue"].sum()

        fig = px.bar(
            reg_agg,
            x="quarter",
            y="revenue",
            color="region",
            barmode="group",
            title=f"Regional Revenue by Quarter - {year_sel}",
            labels={"quarter": "Quarter", "revenue": "Revenue", "region": "Region"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Calendar & Correlation Tab ----
    with tab3:
        st.subheader("Calendar Heatmap - Revenue by Day of Week & Week Number")

        tmp = campaigns.copy()
        tmp["week_of_year"] = tmp["date"].dt.isocalendar().week
        tmp["day_of_week"] = tmp["date"].dt.day_name()

        cal = tmp.groupby(["week_of_year", "day_of_week"], as_index=False)["revenue"].sum()

        # Ensure correct ordering of days
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        cal["day_of_week"] = pd.Categorical(cal["day_of_week"], categories=days_order, ordered=True)

        pivot = cal.pivot_table(index="day_of_week", columns="week_of_year", values="revenue", fill_value=0)

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.astype(str),
                y=pivot.index.astype(str),
                coloraxis="coloraxis",
            )
        )
        fig.update_layout(
            title="Revenue Calendar Heatmap (Week vs Day of Week)",
            xaxis_title="Week of Year",
            yaxis_title="Day of Week",
            coloraxis_colorscale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Metric Correlation Heatmap")
        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Between Marketing Metrics",
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: CUSTOMER INSIGHTS
# =============================================================================
def page_customer_insights(data):
    st.title("👥 Customer Insights")
    st.markdown("Understand who NovaMart customers are and how they behave.")

    customers = data["customers"]

    tab1, tab2 = st.tabs(["Distributions", "Relationships & Segments"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Age Distribution")
            bin_size = st.slider("Bin Size (years)", min_value=2, max_value=10, value=5)
            fig = px.histogram(
                customers,
                x="age",
                nbins=int((customers["age"].max() - customers["age"].min()) / bin_size),
                color="customer_segment",
                barmode="overlay",
                opacity=0.7,
                labels={"age": "Age", "customer_segment": "Segment"},
                title="Age Distribution by Customer Segment",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Lifetime Value by Segment")
            show_points = st.checkbox("Show individual customers", value=True)
            fig = px.box(
                customers,
                x="customer_segment",
                y="lifetime_value",
                points="all" if show_points else "outliers",
                labels={"customer_segment": "Segment", "lifetime_value": "Lifetime Value"},
                title="Lifetime Value Distribution by Segment",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Satisfaction vs NPS Category")

        fig = px.violin(
            customers,
            x="nps_category",
            y="satisfaction_score",
            color="nps_category",
            box=True,
            points="all",
            title="Satisfaction Score by NPS Category",
            labels={"nps_category": "NPS Category", "satisfaction_score": "Satisfaction Score"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Income vs Lifetime Value")

        fig = px.scatter(
            customers,
            x="income",
            y="lifetime_value",
            color="customer_segment",
            hover_data=["region", "city_tier"],
            labels={"income": "Annual Income", "lifetime_value": "Lifetime Value"},
            title="Income vs Lifetime Value by Segment",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Churn Probability by Segment & Region")
        churn = (
            customers.groupby(["customer_segment", "region"], as_index=False)["churn_probability"]
            .mean()
        )
        fig = px.bar(
            churn,
            x="customer_segment",
            y="churn_probability",
            color="region",
            barmode="group",
            labels={"customer_segment": "Segment", "churn_probability": "Churn Probability"},
            title="Average Churn Probability by Segment & Region",
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: PRODUCT PERFORMANCE
# =============================================================================
def page_product_performance(data):
    st.title("📦 Product Performance")
    st.markdown("Analyse NovaMart product categories and profitability.")

    products = data["products"]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Category → Subcategory → Product Treemap")
        fig = px.treemap(
            products,
            path=["category", "subcategory", "product_name"],
            values="sales",
            color="profit_margin",
            color_continuous_scale="RdYlGn",
            title="Sales & Profit Margin Treemap",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Categories by Sales")
        cat = products.groupby("category", as_index=False)["sales"].sum()
        fig = px.bar(
            cat.sort_values("sales", ascending=True),
            x="sales",
            y="category",
            orientation="h",
            title="Total Sales by Category",
            labels={"sales": "Sales", "category": "Category"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Regional Category Performance")
    region_sel = st.selectbox("Select Region", sorted(products["region"].unique().tolist()))
    filt = products[products["region"] == region_sel]
    reg_cat = filt.groupby("category", as_index=False)[["sales", "profit"]].sum()

    fig = px.bar(
        reg_cat,
        x="category",
        y=["sales", "profit"],
        barmode="group",
        title=f"Sales & Profit by Category - {region_sel}",
        labels={"value": "Value", "category": "Category", "variable": "Metric"},
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: GEOGRAPHIC ANALYSIS
# =============================================================================
def page_geographic_analysis(data):
    st.title("🗺️ Geographic Analysis")
    st.markdown("State-level and regional performance across India.")

    geo = data["geographic"]

    st.subheader("State Revenue & Customers (Bubble Map)")
    metric = st.selectbox(
        "Bubble Size Metric",
        ["total_revenue", "total_customers", "store_count"],
        index=0,
    )
    color_metric = st.selectbox(
        "Color Metric",
        ["revenue_per_customer", "market_penetration", "yoy_growth", "customer_satisfaction"],
        index=1,
    )

    fig = px.scatter_geo(
        geo,
        lat="latitude",
        lon="longitude",
        size=metric,
        color=color_metric,
        hover_name="state",
        hover_data={"region": True, metric: True, color_metric: True},
        projection="natural earth",
        title="Store & Revenue Footprint Across India",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Region vs Market Penetration")

    reg = geo.groupby("region", as_index=False)[["market_penetration", "customer_satisfaction"]].mean()
    fig = px.bar(
        reg,
        x="region",
        y=["market_penetration", "customer_satisfaction"],
        barmode="group",
        title="Market Penetration & Satisfaction by Region",
        labels={"value": "Score", "variable": "Metric", "region": "Region"},
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: ATTRIBUTION & FUNNEL
# =============================================================================
def page_attribution_funnel(data):
    st.title("🎯 Attribution & Funnel")
    st.markdown("How different channels contribute along the marketing funnel.")

    attribution = data["attribution"]
    funnel = data["funnel"]
    correlation = data["correlation"]
    journey = data["journey"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Channel Attribution Models")
        model = st.selectbox(
            "Attribution Model",
            ["first_touch", "last_touch", "linear", "time_decay", "position_based"],
            index=1,
        )

        fig = px.pie(
            attribution,
            names="channel",
            values=model,
            hole=0.4,
            title=f"Channel Attribution - {model.replace('_', ' ').title()}",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Conversion Funnel")
        fig = go.Figure(
            go.Funnel(
                y=funnel["stage"],
                x=funnel["visitors"],
                textinfo="value+percent initial",
            )
        )
        fig.update_layout(title="Marketing Funnel")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Customer Journey Sankey (Touchpoints)")

    # Build Sankey nodes and links from 4 touchpoints
    stages = ["touchpoint_1", "touchpoint_2", "touchpoint_3", "touchpoint_4"]
    
    # Collect all unique labels, ignoring missing values and converting to string
    label_set = set()
    for col in stages:
        label_set |= set(journey[col].dropna().astype(str))
    
    all_labels = sorted(label_set)
    label_to_idx = {lab: i for i, lab in enumerate(all_labels)}
    
    sources, targets, values = [], [], []
    
    for i in range(len(stages) - 1):
        s_col = stages[i]
        t_col = stages[i + 1]
    
        # Use only rows where both touchpoints exist
        subset = journey.dropna(subset=[s_col, t_col]).copy()
        subset[s_col] = subset[s_col].astype(str)
        subset[t_col] = subset[t_col].astype(str)
    
        pair_agg = subset.groupby([s_col, t_col], as_index=False)["customer_count"].sum()
        for _, row in pair_agg.iterrows():
            sources.append(label_to_idx[row[s_col]])
            targets.append(label_to_idx[row[t_col]])
            values.append(row["customer_count"])


    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    label=all_labels,
                ),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(title_text="Customer Journey Across Touchpoints", font_size=10)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Metric Correlation Heatmap")
    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Between Marketing Metrics",
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: ML MODEL EVALUATION
# =============================================================================
def page_ml_evaluation(data):
    st.title("🤖 ML Model Evaluation - Lead Scoring")
    st.markdown("Evaluate the binary classifier used for lead conversion prediction.")

    leads = data["leads"]
    lc = data["learning_curve"]
    fi = data["feature_importance"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )

        y_true = leads["actual_converted"].values
        y_score = leads["predicted_probability"].values
        y_pred = (y_score >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"],
        )

        fig = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix (threshold = {threshold:.2f})",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cm_df.style.format("{:.0f}"))

    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash"))
        )
        fig.update_layout(
            title=f"ROC Curve (AUC = {auc_score:.3f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Learning Curve")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lc["training_size"],
            y=lc["train_score"],
            mode="lines+markers",
            name="Training Score",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lc["training_size"],
            y=lc["validation_score"],
            mode="lines+markers",
            name="Validation Score",
        )
    )

    fig.update_layout(
        xaxis_title="Training Set Size",
        yaxis_title="Score",
        title="Learning Curve (Training vs Validation Score)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature Importance")
    fi_sorted = fi.sort_values("importance", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=fi_sorted["importance"],
            y=fi_sorted["feature"],
            orientation="h",
            name="Importance",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fi_sorted["importance"] + fi_sorted["importance_std"],
            y=fi_sorted["feature"],
            mode="markers",
            name="Std Dev",
        )
    )
    fig.update_layout(
        title="Feature Importance with Uncertainty",
        xaxis_title="Importance",
        yaxis_title="Feature",
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    data = load_data()
    if data is None:
        return

    page = side_nav()

    if page == "🏠 Executive Overview":
        page_executive_overview(data)
    elif page == "📈 Campaign Analytics":
        page_campaign_analytics(data)
    elif page == "👥 Customer Insights":
        page_customer_insights(data)
    elif page == "📦 Product Performance":
        page_product_performance(data)
    elif page == "🗺️ Geographic Analysis":
        page_geographic_analysis(data)
    elif page == "🎯 Attribution & Funnel":
        page_attribution_funnel(data)
    elif page == "🤖 ML Model Evaluation":
        page_ml_evaluation(data)


if __name__ == "__main__":
    main()
