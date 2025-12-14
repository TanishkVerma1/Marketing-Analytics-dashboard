
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



def format_big_number(num):
    """Format large numbers with K / M suffixes."""
    if pd.isna(num):
        return "-"
    if abs(num) >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    if abs(num) >= 1_000:
        return f"{num/1_000:.1f}K"
    return f"{num:,.0f}"


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
    """Executive Overview page – close to professor's reference dashboard."""
    st.title("📈 Executive Overview")
    st.markdown(
        "Key performance indicators and high-level metrics for NovaMart marketing operations."
    )

    campaigns = data["campaigns"]
    customers = data["customers"]

    # ----- Year filter (sidebar) -----
    years = sorted(campaigns["year"].unique().tolist())
    year_options = ["All"] + [str(y) for y in years]
    year_choice = st.sidebar.selectbox(
        "Select Year",
        year_options,
        index=0,
        key="eo_year_filter",
    )
    if year_choice == "All":
        cam_filt = campaigns.copy()
        selected_year = None
    else:
        selected_year = int(year_choice)
        cam_filt = campaigns[campaigns["year"] == selected_year].copy()

    # ----- Year-over-year calculations -----
    year_summary = (
        campaigns.groupby("year", as_index=False)
        .agg(revenue=("revenue", "sum"), conversions=("conversions", "sum"))
        .sort_values("year")
        .reset_index(drop=True)
    )

    def get_yoy(df, metric, year_selected):
        if len(df) < 2:
            return None
        if year_selected is None:
            curr = df.iloc[-1][metric]
            prev = df.iloc[-2][metric]
        else:
            idx_list = df.index[df["year"] == year_selected].tolist()
            if not idx_list:
                return None
            idx = idx_list[0]
            if idx == 0:
                return None
            curr = df.iloc[idx][metric]
            prev = df.iloc[idx - 1][metric]
        if prev == 0:
            return None
        return (curr - prev) / prev * 100.0

    yoy_revenue = get_yoy(year_summary, "revenue", selected_year)
    yoy_conversions = get_yoy(year_summary, "conversions", selected_year)

    # ----- KPIs -----
    total_revenue = cam_filt["revenue"].sum()
    total_conversions = cam_filt["conversions"].sum()
    total_spend = cam_filt["spend"].sum()
    overall_roas = total_revenue / total_spend if total_spend > 0 else np.nan

    customer_base = len(customers)
    active_customers = int((customers["is_churned"] == 0).sum())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Revenue",
            format_money_indian(total_revenue),
            delta=f"{yoy_revenue:.1f}% YoY" if yoy_revenue is not None else None,
        )
    with col2:
        st.metric(
            "Total Conversions",
            format_big_number(total_conversions),
            delta=f"{yoy_conversions:.1f}% YoY" if yoy_conversions is not None else None,
        )
    with col3:
        st.metric(
            "Overall ROAS",
            f"{overall_roas:.2f}x" if not pd.isna(overall_roas) else "-",
        )
    with col4:
        st.metric(
            "Customer Base",
            format_big_number(customer_base),
            delta=f"{active_customers} active",
        )

        st.markdown("---")

    
    # ----- Revenue trend & channel performance -----
    left, right = st.columns([2, 1])

    # ===== LEFT: Revenue Trend Over Time with aggregation toggle =====
    with left:
        st.subheader("Revenue Trend Over Time")

        agg_choice = st.radio(
            "Aggregation",
            ["Daily", "Weekly", "Monthly"],
            horizontal=True,
            key="eo_revenue_agg",
        )

        tmp = cam_filt.copy()
        if tmp.empty:
            st.info("No data available for the selected year.")
        else:
            if agg_choice == "Daily":
                tmp["period"] = tmp["date"]
                subtitle = "Daily Revenue Trend"
            elif agg_choice == "Weekly":
                tmp["period"] = tmp["date"].dt.to_period("W").apply(lambda r: r.start_time)
                subtitle = "Weekly Revenue Trend"
            else:
                tmp["period"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
                subtitle = "Monthly Revenue Trend"

            trend = (
                tmp.groupby("period", as_index=False)["revenue"]
                .sum()
                .sort_values("period")
            )

            if trend.empty:
                st.info("No data available for the selected year.")
            else:
                trend["ma7"] = trend["revenue"].rolling(window=7, min_periods=1).mean()

                st.caption(subtitle)
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=trend["period"],
                        y=trend["revenue"],
                        mode="lines",
                        name="Revenue",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=trend["period"],
                        y=trend["ma7"],
                        mode="lines",
                        name="7-period MA",
                        line=dict(dash="dash"),
                    )
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Revenue (₹)",
                )
                st.plotly_chart(fig, use_container_width=True)

    # ===== RIGHT: Channel Performance with metric dropdown =====
    with right:
        st.subheader("Channel Performance")

        metric_label = st.selectbox(
            "Select Metric",
            ["Revenue", "Conversions", "ROAS"],
            index=0,
            key="eo_channel_metric",
        )

        # Map selection to actual field names
        if metric_label == "Revenue":
            metric_col = "revenue"
            agg_type = "sum"
        elif metric_label == "Conversions":
            metric_col = "conversions"
            agg_type = "sum"
        else:  # ROAS
            metric_col = "roas"
            agg_type = "mean"

        if cam_filt.empty:
            st.info("No channel data for the selected year.")
        else:
            # Aggregate values
            if agg_type == "sum":
                channel_perf = (
                    cam_filt.groupby("channel", as_index=False)[metric_col]
                    .sum()
                    .sort_values(metric_col, ascending=True)
                )
            else:
                channel_perf = (
                    cam_filt.groupby("channel", as_index=False)[metric_col]
                    .mean()
                    .sort_values(metric_col, ascending=True)
                )

            if not channel_perf.empty:
                fig = px.bar(
                    channel_perf,
                    x=metric_col,
                    y="channel",
                    orientation="h",
                    color=metric_col,
                    color_continuous_scale="Blues",
                    labels={metric_col: metric_label, "channel": "Channel"},
                    title=f"{metric_label} by Channel",
                )

                # Remove ₹ symbol from x-axis for all metrics
                fig.update_layout(xaxis_title=metric_label)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No channel data for the selected year.")

    st.markdown("---")

    # ----- Regional revenue & campaign type performance -----
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Regional Revenue Distribution")
        reg = (
            cam_filt.groupby("region", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
        )
        if not reg.empty:
            fig = px.pie(
                reg,
                values="revenue",
                names="region",
                title="Revenue Share by Region",
                hole=0.0,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No regional data for the selected year.")

    with col_b:
        st.subheader("Campaign Type Performance")
        ct = (
            cam_filt.groupby("campaign_type", as_index=False)[["revenue", "spend"]]
            .sum()
            .sort_values("revenue", ascending=False)
        )
        if not ct.empty:
            melted = ct.melt(
                id_vars="campaign_type",
                value_vars=["revenue", "spend"],
                var_name="metric",
                value_name="amount",
            )
            fig = px.bar(
                melted,
                x="campaign_type",
                y="amount",
                color="metric",
                barmode="group",
                labels={
                    "campaign_type": "Campaign Type",
                    "amount": "Amount (₹)",
                    "metric": "Metric",
                },
                title="Revenue vs Spend by Campaign Type",
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No campaign type data for the selected year.")

    st.markdown("---")

    # ----- Channel performance summary table & insights -----
    st.subheader("Channel Performance Summary")
    summary = (
        cam_filt.groupby("channel", as_index=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            ctr=("ctr", "mean"),
            conversion_rate=("conversion_rate", "mean"),
            roas=("roas", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )

    if not summary.empty:
        display = summary.copy()
        display["impressions"] = display["impressions"].map(lambda x: f"{int(x):,}")
        display["clicks"] = display["clicks"].map(lambda x: f"{int(x):,}")
        display["conversions"] = display["conversions"].map(lambda x: f"{int(x):,}")
        display["spend"] = display["spend"].map(format_money_indian)
        display["revenue"] = display["revenue"].map(format_money_indian)
        display["ctr"] = display["ctr"].map(lambda x: f"{x*100:.2f}%")
        display["conversion_rate"] = display["conversion_rate"].map(
            lambda x: f"{x*100:.2f}%"
        )
        display["roas"] = display["roas"].map(lambda x: f"{x:.2f}x")

        st.dataframe(display, use_container_width=True)

        # Key insights
        top_rev_row = summary.loc[summary["revenue"].idxmax()]
        best_roas_row = summary.loc[summary["roas"].idxmax()]

        st.markdown("### 💡 Key Insights")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.success(
                f"**Top Performing Channel:** {top_rev_row['channel']} with "
                f"{format_money_indian(top_rev_row['revenue'])} revenue."
            )
        with col_i2:
            st.info(
                f"**Best ROAS:** {best_roas_row['channel']} at "
                f"{best_roas_row['roas']:.2f}x return on ad spend."
            )
    else:
        st.info("No channel performance data available for the selected year.")


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
    
        # ----- Customer Overview KPIs -----
    total_customers = len(customers)
    avg_ltv = customers["lifetime_value"].mean()
    avg_satisfaction = customers["satisfaction_score"].mean()
    churn_rate = customers["is_churned"].mean() * 100
    avg_purchases = customers["total_purchases"].mean()

    st.markdown("### Customer Overview")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.metric("Total Customers", format_big_number(total_customers))

    with kpi2:
        st.metric("Avg. Lifetime Value", format_money_indian(avg_ltv))

    with kpi3:
        st.metric("Avg. Satisfaction", f"{avg_satisfaction:.2f}/5")

    with kpi4:
        st.metric("Churn Rate", f"{churn_rate:.1f}%")

    with kpi5:
        st.metric("Avg. Purchases", f"{avg_purchases:.0f}")

    st.markdown("---")

    
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
    st.markdown("Analyze product sales hierarchy, category performance, and regional distribution.")

    products = data["products"]

    # ----- Sidebar filters -----
    st.sidebar.markdown("### Product Filters")

    year_options = sorted(products["year"].unique().tolist())
    year_choice = st.sidebar.selectbox(
        "Year",
        ["All"] + [str(y) for y in year_options],
        index=0,
        key="pp_year",
    )

    cat_options = sorted(products["category"].unique().tolist())
    cat_selected = st.sidebar.multiselect(
        "Categories",
        cat_options,
        default=cat_options,
        key="pp_categories",
    )

    region_options = sorted(products["region"].unique().tolist())
    region_selected = st.sidebar.multiselect(
        "Regions",
        region_options,
        default=region_options,
        key="pp_regions",
    )

    prod_filt = products.copy()
    if year_choice != "All":
        prod_filt = prod_filt[prod_filt["year"] == int(year_choice)]
    if cat_selected:
        prod_filt = prod_filt[prod_filt["category"].isin(cat_selected)]
    if region_selected:
        prod_filt = prod_filt[prod_filt["region"].isin(region_selected)]

    # ----- Product Overview KPIs -----
    st.markdown("### Product Overview")

    total_sales = prod_filt["sales"].sum()
    total_units = prod_filt["units_sold"].sum()
    total_profit = prod_filt["profit"].sum()
    if total_sales > 0:
        avg_margin = total_profit / total_sales * 100
    else:
        avg_margin = float("nan")
    avg_rating = prod_filt["avg_rating"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.metric("Total Sales", format_money_indian(total_sales))
    with k2:
        st.metric("Units Sold", format_big_number(total_units))
    with k3:
        st.metric("Total Profit", format_money_indian(total_profit))
    with k4:
        st.metric("Avg Profit Margin", f"{avg_margin:.1f}%")
    with k5:
        st.metric("Avg Rating", f"{avg_rating:.2f}/5" if not pd.isna(avg_rating) else "-")

    st.markdown("---")

    # ===== Product Sales Hierarchy (Treemap) =====
    st.markdown("### Section 5: Product Sales Hierarchy (Treemap)")

    left, right = st.columns([4, 1])

    with right:
        color_by = st.selectbox(
            "Color by",
            ["profit_margin", "sales", "profit", "avg_rating", "return_rate"],
            index=0,
            key="pp_color_by",
        )

    with left:
        st.subheader("Product Hierarchy (Size: Sales, Color: Selected Metric)")
        if prod_filt.empty:
            st.info("No data for the selected filters.")
        else:
            fig = px.treemap(
                prod_filt,
                path=["category", "subcategory", "product_name"],
                values="sales",
                color=color_by,
                color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ===== Subcategory Performance =====
    st.markdown("### Subcategory Performance")
    st.subheader("Sales by Subcategory")

    if prod_filt.empty:
        st.info("No data for the selected filters.")
    else:
        sub = (
            prod_filt.groupby("subcategory", as_index=False)
            .agg({"sales": "sum", "profit_margin": "mean"})
        )
        sub = sub.sort_values("sales", ascending=False).head(10)

        fig = px.bar(
            sub,
            x="sales",
            y="subcategory",
            orientation="h",
            color="profit_margin",
            color_continuous_scale="RdYlGn",
            labels={"sales": "Sales", "subcategory": "Subcategory", "profit_margin": "profit_margin"},
            title="Sales by Subcategory",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===== Regional Product Performance & Quarterly Trends =====
    st.markdown("### Regional Product Performance")

    col_reg, col_trend = st.columns(2)

    with col_reg:
        st.subheader("Sales by Region")
        if prod_filt.empty:
            st.info("No data for the selected filters.")
        else:
            reg_cat = (
                prod_filt.groupby(["region", "category"], as_index=False)["sales"].sum()
            )
            fig = px.bar(
                reg_cat,
                x="region",
                y="sales",
                color="category",
                barmode="stack",
                labels={"sales": "Sales", "region": "Region", "category": "Category"},
                title="Category Sales by Region",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_trend:
        st.subheader("Quarterly Trends")
        if prod_filt.empty:
            st.info("No data for the selected filters.")
        else:
            q_cat = (
                prod_filt.groupby(["quarter", "category"], as_index=False)["sales"].sum()
            )

            # Order quarters as strings (Q1 2023, Q1 2024, Q2 2023, Q2 2024, ...)
            all_quarters = sorted(q_cat["quarter"].unique().tolist())

            fig = px.line(
                q_cat,
                x="quarter",
                y="sales",
                color="category",
                markers=True,
                category_orders={"quarter": all_quarters},
                labels={"sales": "Sales", "quarter": "Quarter", "category": "Category"},
                title="Quarterly Sales Trend by Category",
            )

            # Make quarter labels slanted like your professor's chart
            fig.update_layout(
                xaxis_title="Quarter",
                xaxis_tickangle=-45,
            )

            st.plotly_chart(fig, use_container_width=True)

    # ===== Top Products & Key Insights =====
    st.markdown("### Top Products by Sales")

    if prod_filt.empty:
        st.info("No data for the selected filters.")
    else:
        agg = (
            prod_filt.groupby(["product_name", "category", "subcategory"], as_index=False)
            .agg(
                {
                    "sales": "sum",
                    "units_sold": "sum",
                    "profit": "sum",
                    "profit_margin": "mean",
                    "avg_rating": "mean",
                    "return_rate": "mean",
                }
            )
        )
        top_prod = agg.sort_values("sales", ascending=False).head(10)

        display_df = pd.DataFrame(
            {
                "Product": top_prod["product_name"],
                "Category": top_prod["category"],
                "Subcategory": top_prod["subcategory"],
                "Sales": top_prod["sales"].apply(format_money_indian),
                "Units Sold": top_prod["units_sold"].astype(int),
                "Profit": top_prod["profit"].apply(format_money_indian),
                "Margin": top_prod["profit_margin"].map(
                    lambda x: f"{x:.1f}%" if not pd.isna(x) else "-"
                ),
                "Rating": top_prod["avg_rating"].map(
                    lambda x: f"{x:.1f}" if not pd.isna(x) else "-"
                ),
                "Return Rate": top_prod["return_rate"].map(
                    lambda x: f"{x:.1f}%" if not pd.isna(x) else "-"
                ),
            }
        )

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Key insights
        st.markdown("### Key Insights")
        c1, c2, c3 = st.columns(3)

        # Top Category by Sales
        cat_sales = prod_filt.groupby("category", as_index=False)["sales"].sum()
        top_cat = cat_sales.sort_values("sales", ascending=False).iloc[0]

        # Best Margins (by category, weighted by sales)
        cat_margin_df = prod_filt.groupby("category", as_index=False)[["profit", "sales"]].sum()
        cat_margin_df["margin"] = cat_margin_df["profit"] / cat_margin_df["sales"] * 100
        best_margin = cat_margin_df.sort_values("margin", ascending=False).iloc[0]

        # Top Rated subcategory
        sub_rating = prod_filt.groupby("subcategory", as_index=False)["avg_rating"].mean()
        best_rating = sub_rating.sort_values("avg_rating", ascending=False).iloc[0]

        with c1:
            st.markdown(
                f"**Top Category:** {top_cat['category']} with {format_money_indian(top_cat['sales'])} in sales"
            )
        with c2:
            st.markdown(
                f"**Best Margins:** {best_margin['category']} at {best_margin['margin']:.1f}% average margin"
            )
        with c3:
            st.markdown(
                f"**Top Rated:** {best_rating['subcategory']} at {best_rating['avg_rating']:.1f}/5 average rating"
            )

# =============================================================================
# PAGE: GEOGRAPHIC ANALYSIS
# =============================================================================
def page_geographic_analysis(data):
    st.title("🗺️ Geographic Analysis")
    st.markdown(
        "State-wise performance metrics and store distribution across India."
    )

    geo = data["geographic"]

    # ==========================
    # Geographic Overview KPIs
    # ==========================
    st.markdown("### Geographic Overview")

    states_covered = geo["state"].nunique()
    total_customers = geo["total_customers"].sum()
    total_revenue = geo["total_revenue"].sum()
    total_stores = geo["store_count"].sum()
    avg_satisfaction = geo["customer_satisfaction"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("States Covered", f"{states_covered}")
    with c2:
        st.metric("Total Customers", format_big_number(total_customers))
    with c3:
        st.metric("Total Revenue", format_money_indian(total_revenue))
    with c4:
        st.metric("Total Stores", f"{int(total_stores)}")
    with c5:
        st.metric(
            "Avg Satisfaction",
            f"{avg_satisfaction:.2f}/5" if not pd.isna(avg_satisfaction) else "-",
        )

    st.markdown("---")

    # ==========================
    # Section 6: State-wise Performance
    # ==========================
    st.markdown("### Section 6: State-wise Performance")

    left, right = st.columns([4, 1])

    metric_map = {
        "Total Revenue": "total_revenue",
        "Total Customers": "total_customers",
        "Market Penetration": "market_penetration",
        "YoY Growth": "yoy_growth",
        "Customer Satisfaction": "customer_satisfaction",
    }

    with right:
        metric_label = st.selectbox(
            "Select Metric",
            list(metric_map.keys()),
            index=0,
            key="geo_state_metric",
        )
        metric_col = metric_map[metric_label]

    with left:
        st.subheader("State-wise Total Revenue")
        if geo.empty:
            st.info("No geographic data available.")
        else:
            fig = px.scatter_geo(
                geo,
                lat="latitude",
                lon="longitude",
                size="total_revenue",          # bubble size – total revenue
                color=metric_col,              # color – selected metric
                hover_name="state",
                hover_data={
                    "region": True,
                    "total_revenue": True,
                    "total_customers": True,
                    "store_count": True,
                    metric_col: True,
                },
                projection="natural earth",
                color_continuous_scale="Blues",
            )

            # 👇 all of this stays inside the ELSE block
            fig.update_geos(
                fitbounds="locations",
                visible=True,
                showcountries=True,
                countrycolor="LightGray",
                showland=True,
                landcolor="#F5F5F5",
                lataxis_range=[5, 37],
                lonaxis_range=[68, 98],
            )

            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                coloraxis_colorbar_title=metric_col,
            )

            st.plotly_chart(fig, use_container_width=True)


    st.markdown("---")

    # ==========================
    # Store Performance Map
    # ==========================
    st.markdown("### Store Performance Map")
    st.caption("Store Count (Size) vs Customer Satisfaction (Color)")

    if geo.empty:
        st.info("No geographic data available.")
    else:
        fig = px.scatter_geo(
            geo,
            lat="latitude",
            lon="longitude",
            size="store_count",               # size: number of stores
            color="customer_satisfaction",    # color: satisfaction
            hover_name="state",
            hover_data={
                "region": True,
                "store_count": True,
                "customer_satisfaction": True,
            },
            projection="natural earth",
            color_continuous_scale="RdYlGn",
        )

        fig.update_geos(
            fitbounds="locations",
            visible=True,
            showcountries=True,
            countrycolor="LightGray",
            showland=True,
            landcolor="#F5F5F5",
            lataxis_range=[5, 37],
            lonaxis_range=[68, 98],
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar_title="customer_satisfaction",
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Map Legend", expanded=True):
            st.markdown("**Size:** Store Count  \n**Color:** Customer Satisfaction")


        # Small legend text like your sir's map
        with st.expander("Map Legend", expanded=True):
            st.markdown("**Size:** Store Count  \\n**Color:** Customer Satisfaction")

    st.markdown("---")

    # ==========================
    # Regional Comparison
    # ==========================
    st.markdown("### Regional Comparison")

    col_rev, col_bubble = st.columns(2)

    # --- Revenue by State (bar chart)
    with col_rev:
        st.subheader("Revenue by State")
        if geo.empty:
            st.info("No geographic data available.")
        else:
            rev_sorted = geo.sort_values("total_revenue", ascending=False)
            fig = px.bar(
                rev_sorted,
                x="total_revenue",
                y="state",
                orientation="h",
                color="total_revenue",
                color_continuous_scale="Blues",
                labels={"total_revenue": "Revenue", "state": "State"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- YoY Growth vs Market Penetration (bubble chart)
    with col_bubble:
        st.subheader("YoY Growth vs Market Penetration")
        if geo.empty:
            st.info("No geographic data available.")
        else:
            avg_growth = geo["yoy_growth"].mean()
            avg_pen = geo["market_penetration"].mean()

            fig = px.scatter(
                geo,
                x="market_penetration",
                y="yoy_growth",
                size="total_revenue",
                color="region",
                hover_name="state",
                labels={
                    "market_penetration": "Market Penetration (%)",
                    "yoy_growth": "YoY Growth (%)",
                    "region": "Region",
                },
            )
            # reference lines (overall averages)
            fig.add_hline(y=avg_growth, line_dash="dash", line_color="gray")
            fig.add_vline(x=avg_pen, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==========================
    # State Performance Details (table)
    # ==========================
    st.markdown("### State Performance Details")

    regions = ["All"] + sorted(geo["region"].unique().tolist())
    region_choice = st.selectbox(
        "Filter by Region",
        regions,
        index=0,
        key="geo_region_filter",
    )

    table_data = geo.copy()
    if region_choice != "All":
        table_data = table_data[table_data["region"] == region_choice]

    if table_data.empty:
        st.info("No data for the selected region.")
    else:
        display_df = pd.DataFrame(
            {
                "State": table_data["state"],
                "Region": table_data["region"],
                "Customers": table_data["total_customers"].apply(
                    lambda x: f"{x:,.0f}"
                ),
                "Revenue": table_data["total_revenue"].apply(format_money_indian),
                "Rev/Customer": table_data["revenue_per_customer"].apply(
                    lambda x: format_money_indian(x) if not pd.isna(x) else "-"
                ),
                "Stores": table_data["store_count"].astype(int),
                "Penetration": table_data["market_penetration"].map(
                    lambda x: f"{x:.1f}%"
                ),
                "YoY Growth": table_data["yoy_growth"].map(
                    lambda x: f"{x:+.1f}%"
                ),
                "Satisfaction": table_data["customer_satisfaction"].map(
                    lambda x: f"{x:.2f}"
                ),
                "Avg Delivery": table_data["avg_delivery_days"].map(
                    lambda x: f"{x:.1f} days"
                ),
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ==========================
    # Key Insights
    # ==========================
    st.markdown("### Key Insights")

    if geo.empty:
        st.info("No data available for insights.")
    else:
        # Top revenue state
        top_state = geo.sort_values("total_revenue", ascending=False).iloc[0]

        # Fastest growing (max YoY)
        fastest = geo.sort_values("yoy_growth", ascending=False).iloc[0]

        # Highest satisfaction
        happiest = geo.sort_values("customer_satisfaction", ascending=False).iloc[0]

        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown(
                f"**Top State:** {top_state['state']} with "
                f"{format_money_indian(top_state['total_revenue'])} revenue"
            )
        with k2:
            st.markdown(
                f"**Fastest Growing:** {fastest['state']} "
                f"at {fastest['yoy_growth']:.1f}% YoY"
            )
        with k3:
            st.markdown(
                f"**Highest Satisfaction:** {happiest['state']} "
                f"at {happiest['customer_satisfaction']:.2f}/5"
            )

# =============================================================================
# PAGE: ATTRIBUTION & FUNNEL
# =============================================================================
def page_attribution_funnel(data):
    st.title("🎯 Attribution & Funnel")
    st.markdown("How different channels contribute along the marketing funnel.")

    attribution = data["attribution"].copy()
    funnel = data["funnel"].copy()
    correlation = data["correlation"].copy()
    journey = data["journey"].copy()

    # -------------------------------
    # Funnel Metrics (table-style)
    # -------------------------------
    st.subheader("Funnel Metrics")

    # Build drop % between stages
    f = funnel[["stage", "visitors"]].copy()
    f["drop_pct"] = 0.0
    for i in range(1, len(f)):
        prev = f.loc[i - 1, "visitors"]
        cur = f.loc[i, "visitors"]
        f.loc[i, "drop_pct"] = (1 - (cur / prev)) * 100 if prev else 0

    # Display like your screenshot
    display = f.copy()
    display["drop_pct"] = display["drop_pct"].map(lambda x: "✓" if x == 0 else f"↓ {x:.1f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -------------------------------
    # Attribution model descriptions (bullets)
    # -------------------------------
    st.markdown("### Model Descriptions:")
    st.markdown(
        """
- **First Touch:** Credit to first interaction  
- **Last Touch:** Credit to last interaction  
- **Linear:** Equal credit across all touchpoints  
- **Time Decay:** More credit to recent touches  
- **Position Based:** 40% first, 40% last, 20% middle
        """.strip()
    )

    st.markdown("---")

    # -------------------------------
    # Attribution Model Comparison Table
    # -------------------------------
    st.subheader("Attribution Model Comparison Table")

    model_cols = [c for c in ["first_touch", "last_touch", "linear", "time_decay", "position_based"] if c in attribution.columns]
    show = ["channel"] + model_cols
    st.dataframe(attribution[show].sort_values(model_cols[0], ascending=False) if model_cols else attribution,
                 use_container_width=True, hide_index=True)

    st.markdown("---")

    # -------------------------------
    # Channel Attribution (donut) + Funnel (plot)
    # -------------------------------
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

    # -------------------------------
    # Improved Customer Journey Sankey (cleaner + stage-separated labels)
    # -------------------------------
    # -------------------------------
    # Customer Journey Sankey (Clean, budget-style)
    # -------------------------------
    st.subheader("Customer Journey Sankey (Touchpoints)")
    st.caption("Clear stage-wise flow (left ➜ right), colored nodes, cleaner links")
    
    stages = ["touchpoint_1", "touchpoint_2", "touchpoint_3", "touchpoint_4"]
    stages = [c for c in stages if c in journey.columns]
    
    if len(stages) < 2:
        st.warning("Journey dataset does not have enough touchpoint columns to build Sankey.")
    else:
        # ----- 1) Prepare stage-wise unique nodes -----
        stage_values = []
        for col in stages:
            vals = journey[col].dropna().astype(str).unique().tolist()
            stage_values.append(sorted(vals))
    
        # Create nodes with stage-separated labels so they don't merge visually
        node_labels = []
        node_stage = []
        node_key_to_idx = {}
    
        for si, vals in enumerate(stage_values):
            for v in vals:
                label = f"{v}"   # keep label clean (budget-style)
                node_key_to_idx[(si, v)] = len(node_labels)
                node_labels.append(label)
                node_stage.append(si)
    
        # ----- 2) Fixed x/y positions to align like budget diagram -----
        # x positions = columns by stage
        n_stages = len(stages)
        node_x = []
        node_y = []
    
        for si, vals in enumerate(stage_values):
            x = si / (n_stages - 1)  # 0.0 ... 1.0
            m = max(1, len(vals))
            for j in range(m):
                y = (j + 1) / (m + 1)  # evenly spread 0..1
                node_x.append(x)
                node_y.append(y)
    
        # ----- 3) Build links (source, target, value) -----
        sources, targets, values = [], [], []
    
        for si in range(n_stages - 1):
            s_col = stages[si]
            t_col = stages[si + 1]
    
            subset = journey.dropna(subset=[s_col, t_col]).copy()
            subset[s_col] = subset[s_col].astype(str)
            subset[t_col] = subset[t_col].astype(str)
    
            # aggregate
            grp = subset.groupby([s_col, t_col], as_index=False)["customer_count"].sum()
    
            # reduce clutter: keep top flows
            grp = grp.sort_values("customer_count", ascending=False)
            top = grp.head(30)  # increase/decrease if needed
    
            for _, r in top.iterrows():
                s_idx = node_key_to_idx[(si, r[s_col])]
                t_idx = node_key_to_idx[(si + 1, r[t_col])]
                sources.append(s_idx)
                targets.append(t_idx)
                values.append(float(r["customer_count"]))
    
        # ----- 4) Nice node colors + link colors derived from source node -----
        palette = [
            "#2E86AB", "#F18F01", "#2AA876", "#C73E1D", "#6C5B7B",
            "#1B998B", "#E84855", "#4A4E69", "#3A86FF", "#FF006E"
        ]
        node_colors = [palette[si % len(palette)] for si in node_stage]
    
        def hex_to_rgba(hex_color, a=0.35):
            h = hex_color.lstrip("#")
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return f"rgba({r},{g},{b},{a})"
    
        link_colors = [hex_to_rgba(node_colors[s], 0.35) for s in sources]
    
        sankey = go.Figure(
            data=[
                go.Sankey(
                    arrangement="fixed",  # IMPORTANT: makes it look like budget-style columns
                    node=dict(
                        pad=18,
                        thickness=18,
                        label=node_labels,
                        color=node_colors,
                        x=node_x,
                        y=node_y,
                        line=dict(color="rgba(0,0,0,0.35)", width=0.6),
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                    ),
                )
            ]
        )
    
        sankey.update_layout(
            title="Customer Journey Across Touchpoints (Stage-wise Sankey)",
            font_size=12,
            margin=dict(l=10, r=10, t=50, b=10),
        )
    
        st.plotly_chart(sankey, use_container_width=True)


    st.markdown("---")

    # -------------------------------
    # Correlation heatmap
    # -------------------------------
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
