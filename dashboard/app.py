"""EDA & Data Quality Dashboard.

Interactive Streamlit dashboard to profile quality, explore behavior patterns,
and provide role-focused insights as a foundation for persona dashboards.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "silver" / "cleaned_data.csv"
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "gold" / "warehouse.db"
FILTER_KEYS = [
	"f_cluster",
	"f_category",
	"f_region",
	"f_date_range",
	"f_numeric_range",
]
COLORWAY = ["#0f766e", "#2563eb", "#f59e0b", "#dc2626", "#7c3aed", "#16a34a"]


def inject_theme() -> None:
	"""Apply a custom visual style to make dashboard look polished."""
	st.markdown(
		"""
		<style>
			:root {
				--text-dark: #0f172a;
				--text-light: #f8fafc;
			}
			.stApp {
				background: radial-gradient(circle at top right, #f8fafc 0%, #e6f4ff 40%, #f8fafc 100%);
				color: var(--text-dark);
			}
			.stApp, .stMarkdown, .stText, p, label, h1, h2, h3, h4, h5, h6, span, div {
				color: var(--text-dark);
			}
			[data-testid="stSidebar"] {
				background: linear-gradient(180deg, #ffffff 0%, #f8fafc 55%, #f1f5f9 100%);
				border-right: 1px solid #cbd5e1;
			}
			[data-testid="stSidebar"] * {
				color: #0f172a !important;
			}
			[data-testid="stSidebar"] .stButton > button {
				background: #e2e8f0;
				color: #0f172a;
				border: 1px solid #94a3b8;
				font-weight: 600;
			}
			[data-testid="stSidebar"] .stButton > button:hover {
				background: #cbd5e1;
			}
			/* Off-white form controls for requested inputs */
			.stSelectbox div[data-baseweb="select"] > div,
			.stTextInput input,
			.stDateInput input,
			.stDateInput div[data-baseweb="input"] > div,
			.stSlider [data-baseweb="input"] input,
			.stNumberInput input,
			div[data-testid="stDataFrame"] [data-baseweb="input"] input,
			div[data-testid="stDataFrame"] [data-baseweb="input"] > div {
				background-color: #fffaf0 !important;
				color: #0f172a !important;
				border: 1px solid #d6d3d1 !important;
				border-radius: 8px !important;
			}
			.stSelectbox div[data-baseweb="select"] > div {
				min-height: 38px;
			}
			/* Download button in off-white with dark text */
			.stDownloadButton button {
				background: #fffaf0 !important;
				color: #0f172a !important;
				border: 1px solid #d6d3d1 !important;
				font-weight: 600;
			}
			.stDownloadButton button:hover {
				background: #fef3c7 !important;
			}
			.hero {
				padding: 18px 20px;
				border-radius: 16px;
				background: linear-gradient(120deg, #0f172a 0%, #0f766e 55%, #2563eb 100%);
				color: var(--text-light);
				box-shadow: 0 10px 30px rgba(2, 6, 23, 0.2);
				margin-bottom: 10px;
			}
			.hero h1, .hero h2, .hero h3, .hero p, .hero span, .hero div {
				color: var(--text-light) !important;
			}
			.kpi-wrap {
				display: grid;
				grid-template-columns: repeat(6, minmax(0, 1fr));
				gap: 10px;
			}
			.kpi-card {
				background: #ffffff;
				border: 1px solid #dbeafe;
				border-radius: 14px;
				padding: 10px 12px;
				box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
			}
			.kpi-title {
				font-size: 11px;
				font-weight: 600;
				text-transform: uppercase;
				letter-spacing: .04em;
				color: #64748b;
			}
			.kpi-value {
				font-size: 22px;
				font-weight: 700;
				color: #1e3a8a;
			}
			.section-title {
				padding: 8px 12px;
				border-left: 4px solid #0f766e;
				background: #ecfeff;
				border-radius: 6px;
				font-weight: 600;
				color: #0f172a;
				margin-bottom: 8px;
			}
			.chip {
				display: inline-block;
				padding: 4px 10px;
				margin: 2px 5px 2px 0;
				border-radius: 999px;
				border: 1px solid #d1d5db;
				background: #fff;
				font-size: 12px;
			}
		</style>
		""",
		unsafe_allow_html=True,
	)


def style_figure(fig) -> None:
	"""Apply a consistent visual style to all Plotly figures."""
	fig.update_layout(
		template="plotly_white",
		colorway=COLORWAY,
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(255,255,255,0.95)",
		font=dict(color="#0f172a", size=13),
		title_font=dict(color="#1e3a8a", size=18),
		xaxis=dict(title_font=dict(color="#1e3a8a"), tickfont=dict(color="#0f172a")),
		yaxis=dict(title_font=dict(color="#1e3a8a"), tickfont=dict(color="#0f172a")),
		legend=dict(title_font=dict(color="#1e3a8a"), font=dict(color="#0f172a")),
		margin=dict(l=20, r=20, t=45, b=20),
	)


def _existing_default_source() -> str:
	if DEFAULT_CSV_PATH.exists():
		return "CSV"
	if DEFAULT_SQLITE_PATH.exists():
		return "SQLite"
	return "CSV"


@st.cache_data(show_spinner=False)
def load_csv_data(csv_path: str) -> pd.DataFrame:
	path = Path(csv_path)
	if not path.exists():
		raise FileNotFoundError(f"CSV not found: {path}")
	return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_sql_data(sqlite_path: str) -> pd.DataFrame:
	"""Load SQLite data directly from DB path without requiring manual SQL."""
	path = Path(sqlite_path)
	if not path.exists():
		raise FileNotFoundError(f"SQLite DB not found: {path}")

	with sqlite3.connect(path) as conn:
		# Prefer the fact table for dashboard analytics.
		tables = pd.read_sql_query(
			"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
			conn,
		)["name"].tolist()
		if not tables:
			raise ValueError("No tables found in SQLite database.")

		preferred_tables = [
			"fact_customer_churn_metrics",
			"fact_customer_behavior",
		]
		table_name = next((t for t in preferred_tables if t in tables), tables[0])
		return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def get_column_case_map(df: pd.DataFrame) -> dict[str, str]:
	return {col.lower(): col for col in df.columns}


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
	datetime_columns: list[str] = []
	for col in df.columns:
		col_lower = col.lower()
		if any(token in col_lower for token in ["date", "time", "timestamp"]):
			parsed = pd.to_datetime(df[col], errors="coerce")
			if parsed.notna().sum() > 0:
				df[col] = parsed
				datetime_columns.append(col)
	return datetime_columns


def column_or_none(df: pd.DataFrame, candidates: list[str]) -> str | None:
	case_map = get_column_case_map(df)
	for name in candidates:
		hit = case_map.get(name.lower())
		if hit is not None:
			return hit
	return None


def data_quality_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, int, pd.DataFrame]:
	missing = df.isna().sum().reset_index()
	missing.columns = ["column_name", "missing_values"]
	missing = missing.sort_values("missing_values", ascending=False)
	missing["missing_pct"] = (missing["missing_values"] / max(len(df), 1) * 100).round(2)
	duplicate_rows = int(df.duplicated().sum())
	dtypes = pd.DataFrame(
		{"column_name": df.columns, "data_type": [str(dtype) for dtype in df.dtypes]}
	)
	return missing, duplicate_rows, dtypes


def outlier_percentage(df: pd.DataFrame) -> float:
	"""Estimate share of outlier cells across numeric features via IQR rule."""
	numeric = df.select_dtypes(include=["number"])
	if numeric.empty:
		return 0.0
	outlier_cells = 0
	total_cells = 0
	for col in numeric.columns:
		series = numeric[col].dropna()
		if series.empty:
			continue
		q1 = series.quantile(0.25)
		q3 = series.quantile(0.75)
		iqr = q3 - q1
		low = q1 - 1.5 * iqr
		high = q3 + 1.5 * iqr
		flags = (numeric[col] < low) | (numeric[col] > high)
		outlier_cells += int(flags.sum())
		total_cells += int(numeric[col].notna().sum())
	if total_cells == 0:
		return 0.0
	return round((outlier_cells / total_cells) * 100, 2)


def quality_score(df: pd.DataFrame, missing_total: int, duplicate_rows: int) -> float:
	missing_pct = (missing_total / max(df.size, 1)) * 100
	duplicate_pct = (duplicate_rows / max(len(df), 1)) * 100
	outlier_pct = outlier_percentage(df)
	score = 100 - (missing_pct + duplicate_pct + outlier_pct)
	return round(max(0, min(100, score)), 2)


def reset_filters() -> None:
	for key in FILTER_KEYS:
		if key in st.session_state:
			del st.session_state[key]
	st.rerun()


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
	filtered = df.copy()
	active: dict[str, str] = {}

	cluster_col = column_or_none(filtered, ["cluster", "persona"])
	category_col = column_or_none(
		filtered,
		["preferred_order_category", "preferedordercat", "ordercategory", "category"],
	)
	region_col = column_or_none(filtered, ["region", "city_tier", "citytier", "city", "state"])
	amount_col = column_or_none(
		filtered,
		[
			"orderamount",
			"order_count",
			"ordercount",
			"cashback_amount",
			"cashbackamount",
		],
	)

	datetime_cols = detect_datetime_columns(filtered)
	date_col = datetime_cols[0] if datetime_cols else None

	st.sidebar.header("Filters")
	if st.sidebar.button("Reset Filters"):
		reset_filters()

	if cluster_col is not None:
		values = sorted(filtered[cluster_col].dropna().astype(str).unique().tolist())
		chosen = st.sidebar.multiselect(
			"Cluster / Persona",
			values,
			default=values,
			key="f_cluster",
		)
		if chosen and len(chosen) != len(values):
			filtered = filtered[filtered[cluster_col].astype(str).isin(chosen)]
			active["Cluster"] = ", ".join(chosen[:3]) + ("..." if len(chosen) > 3 else "")

	if category_col is not None:
		values = sorted(filtered[category_col].dropna().astype(str).unique().tolist())
		chosen = st.sidebar.multiselect(
			"Category",
			values,
			default=values,
			key="f_category",
		)
		if chosen and len(chosen) != len(values):
			filtered = filtered[filtered[category_col].astype(str).isin(chosen)]
			active["Category"] = ", ".join(chosen[:3]) + ("..." if len(chosen) > 3 else "")

	if region_col is not None:
		values = sorted(filtered[region_col].dropna().astype(str).unique().tolist())
		chosen = st.sidebar.multiselect(
			"Region",
			values,
			default=values,
			key="f_region",
		)
		if chosen and len(chosen) != len(values):
			filtered = filtered[filtered[region_col].astype(str).isin(chosen)]
			active["Region"] = ", ".join(chosen[:3]) + ("..." if len(chosen) > 3 else "")

	if date_col is not None and filtered[date_col].notna().any():
		min_date = filtered[date_col].min().date()
		max_date = filtered[date_col].max().date()
		selected = st.sidebar.date_input(
			"Date Range",
			value=(min_date, max_date),
			min_value=min_date,
			max_value=max_date,
			key="f_date_range",
		)
		if isinstance(selected, tuple) and len(selected) == 2:
			start_date, end_date = selected
			filtered = filtered[
				(filtered[date_col].dt.date >= start_date)
				& (filtered[date_col].dt.date <= end_date)
			]
			if (start_date != min_date) or (end_date != max_date):
				active["Date"] = f"{start_date} to {end_date}"

	if amount_col is not None:
		amount_series = pd.to_numeric(filtered[amount_col], errors="coerce")
		if amount_series.notna().any():
			low = float(amount_series.min())
			high = float(amount_series.max())
			selected_low, selected_high = st.sidebar.slider(
				f"{amount_col} Range",
				min_value=low,
				max_value=high,
				value=(low, high),
				key="f_numeric_range",
			)
			filtered = filtered[
				pd.to_numeric(filtered[amount_col], errors="coerce").between(
					selected_low, selected_high
				)
			]
			if (selected_low != low) or (selected_high != high):
				active[amount_col] = f"{selected_low:.2f} to {selected_high:.2f}"

	return filtered, active


def render_kpi_cards(df: pd.DataFrame, missing_total: int, duplicate_rows: int) -> None:
	churn_col = column_or_none(df, ["churn_flag", "churn"])
	complaint_col = column_or_none(df, ["complaint_flag", "complaintcount", "complain"])
	cashback_col = column_or_none(df, ["cashback_amount", "cashbackamount"])
	order_col = column_or_none(df, ["order_count", "ordercount"])

	churn_rate = (
		(pd.to_numeric(df[churn_col], errors="coerce").mean() * 100) if churn_col else 0.0
	)
	complaint_rate = (
		(pd.to_numeric(df[complaint_col], errors="coerce").mean() * 100)
		if complaint_col
		else 0.0
	)
	avg_cashback = (
		float(pd.to_numeric(df[cashback_col], errors="coerce").mean()) if cashback_col else 0.0
	)
	avg_orders = (
		float(pd.to_numeric(df[order_col], errors="coerce").mean()) if order_col else 0.0
	)

	score = quality_score(df, missing_total, duplicate_rows)

	st.markdown(
		f"""
		<div class='kpi-wrap'>
			<div class='kpi-card'><div class='kpi-title'>Rows</div><div class='kpi-value'>{len(df):,}</div></div>
			<div class='kpi-card'><div class='kpi-title'>Missing</div><div class='kpi-value'>{missing_total:,}</div></div>
			<div class='kpi-card'><div class='kpi-title'>Duplicates</div><div class='kpi-value'>{duplicate_rows:,}</div></div>
			<div class='kpi-card'><div class='kpi-title'>Churn Rate</div><div class='kpi-value'>{churn_rate:.2f}%</div></div>
			<div class='kpi-card'><div class='kpi-title'>Complaint Rate</div><div class='kpi-value'>{complaint_rate:.2f}%</div></div>
			<div class='kpi-card'><div class='kpi-title'>DQ Score</div><div class='kpi-value'>{score:.2f}/100</div></div>
		</div>
		""",
		unsafe_allow_html=True,
	)

	e1, e2 = st.columns(2)
	e1.metric("Avg Cashback", f"{avg_cashback:.2f}")
	e2.metric("Avg Orders", f"{avg_orders:.2f}")


def render_data_quality_section(df: pd.DataFrame) -> tuple[pd.DataFrame, int, pd.DataFrame]:
	st.markdown("<div class='section-title'>Data Quality Metrics</div>", unsafe_allow_html=True)
	missing_df, duplicate_rows, dtypes_df = data_quality_metrics(df)

	left, right = st.columns(2)
	with left:
		st.markdown("Top Missing Columns")
		fig_missing = px.bar(
			missing_df.head(10),
			x="column_name",
			y="missing_values",
			title="Top 10 Columns with Missing Values",
		)
		style_figure(fig_missing)
		fig_missing.update_layout(xaxis_title="Column", yaxis_title="Missing Values")
		st.plotly_chart(fig_missing, width="stretch")
	with right:
		st.markdown("Data Type Consistency")
		st.dataframe(dtypes_df, width="stretch", height=320)

	return missing_df, duplicate_rows, dtypes_df


def render_eda_section(df: pd.DataFrame) -> None:
	st.markdown("<div class='section-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
	numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

	if numeric_cols:
		a, b = st.columns(2)
		with a:
			target_col = st.selectbox("Numeric Distribution", numeric_cols, index=0)
			fig_hist = px.histogram(df, x=target_col, nbins=30, marginal="box")
			style_figure(fig_hist)
			st.plotly_chart(fig_hist, width="stretch")
		with b:
			fig_box = px.box(df[numeric_cols], points="outliers", title="Outlier Detection")
			style_figure(fig_box)
			st.plotly_chart(fig_box, width="stretch")

		fig_corr = px.imshow(
			df[numeric_cols].corr(numeric_only=True),
			text_auto=True,
			title="Correlation Heatmap",
		)
		style_figure(fig_corr)
		st.plotly_chart(fig_corr, width="stretch")
	else:
		st.info("No numeric columns available for EDA.")

	st.markdown("Relationship Views")
	coupon_col = column_or_none(df, ["coupon_used_count", "couponused"])
	cashback_col = column_or_none(df, ["cashback_amount", "cashbackamount"])
	churn_prob_col = column_or_none(df, ["churn_probability", "churn_flag", "churn"])
	cluster_col = column_or_none(df, ["cluster", "persona"])

	x1, x2 = st.columns(2)
	with x1:
		if coupon_col is not None:
			fig_coupon = px.histogram(df, x=coupon_col, nbins=20, title="Coupon Usage Frequency")
			style_figure(fig_coupon)
			st.plotly_chart(fig_coupon, width="stretch")
		else:
			st.info("CouponUsed column not found.")
	with x2:
		if cashback_col is not None and churn_prob_col is not None:
			fig_scatter = px.scatter(
				df,
				x=cashback_col,
				y=churn_prob_col,
				color=cluster_col,
				title=f"{cashback_col} vs {churn_prob_col}",
				opacity=0.75,
			)
			style_figure(fig_scatter)
			st.plotly_chart(fig_scatter, width="stretch")
		else:
			st.info("Need cashback and churn/churn_probability columns for scatter plot.")


def render_time_trends(df: pd.DataFrame) -> None:
	st.markdown("<div class='section-title'>Time Trends</div>", unsafe_allow_html=True)
	datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
	if not datetime_cols:
		st.info("No date/time column available for trends.")
		return

	date_col = datetime_cols[0]
	metric_candidates = [
		c
		for c in [
			column_or_none(df, ["churn_flag", "churn", "churn_probability"]),
			column_or_none(df, ["order_count", "ordercount"]),
			column_or_none(df, ["cashback_amount", "cashbackamount"]),
		]
		if c is not None
	]
	if not metric_candidates:
		st.info("No trend-ready numeric metric found.")
		return

	metric = st.selectbox("Trend Metric", metric_candidates, index=0)
	trend_df = (
		df[[date_col, metric]]
		.dropna()
		.assign(date=df[date_col].dt.date)
		.groupby("date")[metric]
		.mean()
		.reset_index()
	)
	trend_df["rolling_7"] = trend_df[metric].rolling(window=7, min_periods=1).mean()

	fig = px.line(trend_df, x="date", y=[metric, "rolling_7"], title=f"{metric} Trend")
	style_figure(fig)
	st.plotly_chart(fig, width="stretch")


def render_persona_tabs(df: pd.DataFrame) -> None:
	tabs = st.tabs(["Foundation", "Marketing", "Executive", "Support", "Product"])

	with tabs[0]:
		render_eda_section(df)

	with tabs[1]:
		st.subheader("Marketing View")
		coupon_col = column_or_none(df, ["coupon_used_count", "couponused"])
		category_col = column_or_none(df, ["preferred_order_category", "preferedordercat", "category"])
		if coupon_col and category_col:
			agg = (
				df.groupby(category_col)[coupon_col]
				.mean()
				.sort_values(ascending=False)
				.reset_index(name="avg_coupon_used")
			)
			fig_marketing = px.bar(agg, x=category_col, y="avg_coupon_used", title="Avg Coupon Usage by Category")
			style_figure(fig_marketing)
			st.plotly_chart(fig_marketing, width="stretch")
		else:
			st.info("Need coupon and category columns for marketing insights.")

	with tabs[2]:
		st.subheader("Executive View")
		churn_col = column_or_none(df, ["churn_flag", "churn"])
		if churn_col:
			churn_rate = pd.to_numeric(df[churn_col], errors="coerce").mean() * 100
			st.metric("Portfolio Churn Rate", f"{churn_rate:.2f}%")
		render_time_trends(df)

	with tabs[3]:
		st.subheader("Support View")
		complaint_col = column_or_none(df, ["complaint_flag", "complaintcount", "complain"])
		region_col = column_or_none(df, ["region", "city_tier", "citytier", "city", "state"])
		if complaint_col and region_col:
			support_df = (
				df.groupby(region_col)[complaint_col]
				.mean()
				.sort_values(ascending=False)
				.reset_index(name="avg_complaint")
			)
			fig_support = px.bar(support_df, x=region_col, y="avg_complaint", title="Complaint Intensity by Region")
			style_figure(fig_support)
			st.plotly_chart(fig_support, width="stretch")
		else:
			st.info("Need complaint and region columns for support insights.")

	with tabs[4]:
		st.subheader("Product View")
		device_col = column_or_none(df, ["preferred_login_device", "preferredlogindevice"])
		churn_col = column_or_none(df, ["churn_flag", "churn"])
		if device_col and churn_col:
			prod_df = (
				df.groupby(device_col)[churn_col]
				.mean()
				.sort_values(ascending=False)
				.reset_index(name="avg_churn")
			)
			fig_product = px.bar(prod_df, x=device_col, y="avg_churn", title="Churn by Preferred Login Device")
			style_figure(fig_product)
			st.plotly_chart(fig_product, width="stretch")
		else:
			st.info("Need device and churn columns for product insights.")


def render_active_filter_summary(active_filters: dict[str, str]) -> None:
	st.markdown("Active Filters")
	if not active_filters:
		st.caption("No filters applied.")
		return
	chips = "".join(
		[f"<span class='chip'><b>{k}</b>: {v}</span>" for k, v in active_filters.items()]
	)
	st.markdown(chips, unsafe_allow_html=True)


def render_data_preview(df: pd.DataFrame) -> None:
	st.markdown("<div class='section-title'>Filtered Raw Data Preview</div>", unsafe_allow_html=True)
	st.dataframe(df, width="stretch", height=320)
	st.download_button(
		label="Download Filtered Data (CSV)",
		data=df.to_csv(index=False).encode("utf-8"),
		file_name="filtered_dashboard_data.csv",
		mime="text/csv",
	)


def main() -> None:
	st.set_page_config(page_title="EDA & Data Quality Dashboard", layout="wide")
	inject_theme()
	st.markdown(
		"""
		<div class='hero'>
			<h2 style='margin:0;'>EDA & Data Quality Dashboard</h2>
			<p style='margin:6px 0 0 0; opacity:.95; color:#ffffff;'>Explore data reliability, patterns, and persona-level insights from a single control center.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

	default_source = _existing_default_source()
	source_type = st.sidebar.radio(
		"Data Source",
		["CSV", "SQLite"],
		index=0 if default_source == "CSV" else 1,
	)

	start = time.perf_counter()
	try:
		if source_type == "CSV":
			csv_path = st.sidebar.text_input("CSV Path", str(DEFAULT_CSV_PATH))
			dataset = load_csv_data(csv_path)
			source_label = csv_path
		else:
			sqlite_path = st.sidebar.text_input("SQLite DB Path", str(DEFAULT_SQLITE_PATH))
			dataset = load_sql_data(sqlite_path)
			source_label = f"{sqlite_path} | auto-table"
	except Exception as exc:  # noqa: BLE001
		st.error(f"Failed to load dataset: {exc}")
		st.stop()
	elapsed = time.perf_counter() - start

	if dataset.empty:
		st.warning("Loaded dataset is empty. Check source/query.")
		st.stop()

	filtered_df, active_filters = apply_filters(dataset)
	missing_df, duplicate_rows, _ = data_quality_metrics(filtered_df)

	render_kpi_cards(filtered_df, int(missing_df["missing_values"].sum()), duplicate_rows)
	st.caption(f"Loaded {len(dataset):,} rows from {source_label} in {elapsed:.2f}s")
	render_active_filter_summary(active_filters)

	st.divider()
	render_data_quality_section(filtered_df)
	st.divider()
	render_persona_tabs(filtered_df)
	st.divider()
	render_data_preview(filtered_df)


if __name__ == "__main__":
	main()
