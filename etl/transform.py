"""TRANSFORM stage for customer churn ETL pipeline.

This script loads raw data via extract_data(), applies cleaning and
standardization rules, validates quality, and saves Silver-layer output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from extract import extract_data


# Silver layer output path (Medallion Architecture)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
SILVER_OUTPUT_PATH = SILVER_DIR / "cleaned_data.csv"


# Important columns defined by business/data requirements
NUMERIC_COLUMNS = [
	"Tenure",
	"WarehouseToHome",
	"HourSpendOnApp",
	"NumberOfDeviceRegistered",
	"SatisfactionScore",
	"NumberOfAddress",
	"OrderAmountHikeFromlastYear",
	"CouponUsed",
	"OrderCount",
	"DaySinceLastOrder",
	"CashbackAmount",
]

CATEGORICAL_COLUMNS = [
	"PreferredLoginDevice",
	"PreferredPaymentMode",
	"Gender",
	"PreferedOrderCat",
	"MaritalStatus",
]


def print_diagnostics(df: pd.DataFrame, title: str) -> None:
	"""Print concise diagnostics for ETL monitoring."""
	print(f"\n--- {title} ---")
	print(f"Shape: {df.shape}")
	print("Dtypes:")
	print(df.dtypes)
	print("Missing values per column:")
	print(df.isnull().sum())


def coerce_numeric_columns(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
	"""Ensure configured numeric columns are numeric."""
	for col in numeric_columns:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
	return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
	"""Fill missing values: median for numeric, mode for categorical."""
	for col in NUMERIC_COLUMNS:
		if col in df.columns:
			median_value = df[col].median()
			df[col] = df[col].fillna(median_value)

	for col in CATEGORICAL_COLUMNS:
		if col in df.columns:
			mode_series = df[col].mode(dropna=True)
			if not mode_series.empty:
				df[col] = df[col].fillna(mode_series.iloc[0])

	return df


def apply_basic_outlier_filter(df: pd.DataFrame) -> pd.DataFrame:
	"""Remove unrealistic records based on rule-based thresholds."""
	if "OrderCount" in df.columns:
		df = df[df["OrderCount"] <= 30]
	if "HourSpendOnApp" in df.columns:
		df = df[df["HourSpendOnApp"] <= 15]
	return df


def validate_cleaned_dataset(df: pd.DataFrame) -> None:
	"""Validate cleaned data quality before save."""
	total_null_values = int(df.isnull().sum().sum())
	if total_null_values != 0:
		raise ValueError(
			f"Validation failed: cleaned dataset still has {total_null_values} null values."
		)


def clean_data() -> pd.DataFrame:
	"""Main transform workflow to produce Silver-layer cleaned dataset."""
	# 1) Load extracted dataset from EXTRACT stage
	df = extract_data()

	# 2) Print initial diagnostics
	print_diagnostics(df, "Initial Dataset Diagnostics")

	# 3) Remove duplicate rows
	before_dupes = len(df)
	df = df.drop_duplicates().copy()
	removed_dupes = before_dupes - len(df)
	print(f"\nRemoved duplicate rows: {removed_dupes}")

	# 4 & 5) Fix numeric dtypes and fill missing values
	df = coerce_numeric_columns(df, NUMERIC_COLUMNS)
	df = fill_missing_values(df)

	# 6) Outlier filtering
	before_filter = len(df)
	df = apply_basic_outlier_filter(df)
	filtered_rows = before_filter - len(df)
	print(f"Rows removed by outlier filtering: {filtered_rows}")

	# Optional step: keep CustomerID for future warehouse design
	# 7) Drop unnecessary columns only if needed (not dropping by default)

	# 8) Validate cleaned dataset quality
	validate_cleaned_dataset(df)

	# 9) Save Silver layer output
	SILVER_DIR.mkdir(parents=True, exist_ok=True)
	df.to_csv(SILVER_OUTPUT_PATH, index=False)

	# Final diagnostics and confirmation message
	print_diagnostics(df, "Cleaned Dataset Diagnostics")
	print("\nCleaned dataset saved successfully.")
	print(f"Rows: {df.shape[0]}")
	print(f"Columns: {df.shape[1]}")
	print(f"Saved to: {SILVER_OUTPUT_PATH}")

	return df


if __name__ == "__main__":
	clean_data()
