"""Build a Gold-layer Star Schema warehouse from Silver-layer cleaned data.

Medallion flow:
- Bronze: raw files
- Silver: cleaned dataset
- Gold: analytics-ready dimensional model in SQLite

This script creates all required dimensions and the fact table, validates
referential integrity, and persists tables to data/gold/warehouse.db.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV_PATH = PROJECT_ROOT / "data" / "silver" / "cleaned_data.csv"
WAREHOUSE_DB_PATH = PROJECT_ROOT / "data" / "gold" / "warehouse.db"

# Dimension attribute definitions
CUSTOMER_COLUMNS = ["CustomerID", "Gender", "MaritalStatus", "CityTier"]
DEVICE_COLUMNS = ["PreferredLoginDevice"]
PAYMENT_COLUMNS = ["PreferredPaymentMode"]
ORDER_CATEGORY_COLUMNS = ["PreferedOrderCat"]
TIME_COLUMNS = ["DaySinceLastOrder", "Tenure"]

# Fact measures required for analytical queries
FACT_MEASURE_COLUMNS = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
    "Churn",
]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return logger for warehouse build workflow."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("warehouse.build_warehouse")


logger = configure_logging()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def ensure_columns_exist(df: pd.DataFrame, required_columns: list[str], context: str) -> None:
    """Validate required columns exist before transformation."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {missing}")


def load_cleaned_dataset(input_path: Path = INPUT_CSV_PATH) -> pd.DataFrame:
    """Load the Silver-layer cleaned dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input cleaned dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    logger.info("Loaded cleaned dataset from %s", input_path)
    logger.info("Input dataset shape: rows=%d, cols=%d", *df.shape)
    return df


def create_dimension_table(
    source_df: pd.DataFrame,
    natural_key_columns: list[str],
    surrogate_key_column: str,
) -> pd.DataFrame:
    """Create a dimension table by deduplicating natural key attributes."""
    ensure_columns_exist(source_df, natural_key_columns, f"dimension {surrogate_key_column}")

    dim_df = source_df[natural_key_columns].drop_duplicates().reset_index(drop=True)
    dim_df.insert(0, surrogate_key_column, range(1, len(dim_df) + 1))
    return dim_df


def validate_dimension_table(dim_df: pd.DataFrame, surrogate_key_column: str, table_name: str) -> None:
    """Validate dimension table primary-key quality constraints."""
    if dim_df[surrogate_key_column].duplicated().any():
        raise ValueError(f"Duplicate primary keys found in {table_name}.{surrogate_key_column}")

    if dim_df[surrogate_key_column].isnull().any():
        raise ValueError(f"Null primary keys found in {table_name}.{surrogate_key_column}")

    if dim_df.isnull().any().any():
        null_counts = dim_df.isnull().sum()
        raise ValueError(
            f"Null values found in {table_name}: "
            + null_counts[null_counts > 0].to_dict().__repr__()
        )


def validate_foreign_keys(
    fact_df: pd.DataFrame,
    dim_df: pd.DataFrame,
    foreign_key: str,
    dim_table_name: str,
) -> None:
    """Validate that every fact FK value exists in the related dimension PK."""
    valid_values = set(dim_df[foreign_key].tolist())
    invalid_mask = ~fact_df[foreign_key].isin(valid_values)
    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        raise ValueError(
            f"Invalid foreign keys in fact_customer_behavior.{foreign_key} "
            f"not found in {dim_table_name}.{foreign_key}: {invalid_count}"
        )


def build_star_schema_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build star schema dimensions and fact table from cleaned source data."""
    logger.info("Building dimension tables.")

    # Build dimensions with surrogate keys.
    dim_customer = create_dimension_table(df, CUSTOMER_COLUMNS, "customer_key")
    dim_device = create_dimension_table(df, DEVICE_COLUMNS, "device_key")
    dim_payment = create_dimension_table(df, PAYMENT_COLUMNS, "payment_key")
    dim_order_category = create_dimension_table(df, ORDER_CATEGORY_COLUMNS, "category_key")

    # Optional time dimension: created when source columns are available.
    time_dimension_enabled = all(col in df.columns for col in TIME_COLUMNS)
    dim_time = pd.DataFrame()
    if time_dimension_enabled:
        dim_time = create_dimension_table(df, TIME_COLUMNS, "time_key")

    validate_dimension_table(dim_customer, "customer_key", "dim_customer")
    validate_dimension_table(dim_device, "device_key", "dim_device")
    validate_dimension_table(dim_payment, "payment_key", "dim_payment")
    validate_dimension_table(dim_order_category, "category_key", "dim_order_category")
    if time_dimension_enabled:
        validate_dimension_table(dim_time, "time_key", "dim_time")

    logger.info("Created dim_customer with %d rows", len(dim_customer))
    logger.info("Created dim_device with %d rows", len(dim_device))
    logger.info("Created dim_payment with %d rows", len(dim_payment))
    logger.info("Created dim_order_category with %d rows", len(dim_order_category))
    if time_dimension_enabled:
        logger.info("Created dim_time with %d rows", len(dim_time))

    logger.info("Building fact table with surrogate-key references.")

    # Build fact table by replacing dimension attributes with surrogate keys.
    fact_df = df.copy()
    fact_df = fact_df.merge(dim_customer, on=CUSTOMER_COLUMNS, how="left")
    fact_df = fact_df.merge(dim_device, on=DEVICE_COLUMNS, how="left")
    fact_df = fact_df.merge(dim_payment, on=PAYMENT_COLUMNS, how="left")
    fact_df = fact_df.merge(dim_order_category, on=ORDER_CATEGORY_COLUMNS, how="left")

    foreign_key_columns = ["customer_key", "device_key", "payment_key", "category_key"]

    if time_dimension_enabled:
        fact_df = fact_df.merge(dim_time, on=TIME_COLUMNS, how="left")
        foreign_key_columns.append("time_key")

    ensure_columns_exist(fact_df, FACT_MEASURE_COLUMNS, "fact table measures")

    fact_columns = foreign_key_columns + FACT_MEASURE_COLUMNS
    fact_customer_behavior = fact_df[fact_columns].copy()

    # Validate referential integrity assumptions before persistence.
    if len(fact_customer_behavior) != len(df):
        raise ValueError(
            "Fact row count does not match input row count. "
            f"fact={len(fact_customer_behavior)}, input={len(df)}"
        )

    null_fk_counts = fact_customer_behavior[foreign_key_columns].isnull().sum()
    if (null_fk_counts > 0).any():
        raise ValueError(
            "Null foreign keys detected in fact table: "
            + null_fk_counts[null_fk_counts > 0].to_dict().__repr__()
        )

    # Explicit FK validity checks against every dimension table.
    validate_foreign_keys(fact_customer_behavior, dim_customer, "customer_key", "dim_customer")
    validate_foreign_keys(fact_customer_behavior, dim_device, "device_key", "dim_device")
    validate_foreign_keys(fact_customer_behavior, dim_payment, "payment_key", "dim_payment")
    validate_foreign_keys(
        fact_customer_behavior,
        dim_order_category,
        "category_key",
        "dim_order_category",
    )
    if time_dimension_enabled:
        validate_foreign_keys(fact_customer_behavior, dim_time, "time_key", "dim_time")

    logger.info("Validation passed: fact row count and foreign keys are valid.")

    logger.info("Created fact_customer_behavior with %d rows", len(fact_customer_behavior))

    tables: Dict[str, pd.DataFrame] = {
        "dim_customer": dim_customer,
        "dim_device": dim_device,
        "dim_payment": dim_payment,
        "dim_order_category": dim_order_category,
        "fact_customer_behavior": fact_customer_behavior,
    }

    if time_dimension_enabled:
        tables["dim_time"] = dim_time

    return tables


def save_tables_to_sqlite(tables: Dict[str, pd.DataFrame], db_path: Path = WAREHOUSE_DB_PATH) -> None:
    """Persist all warehouse tables into SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving Star Schema tables to SQLite database.")

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        for table_name, table_df in tables.items():
            table_df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info("Saved %s to SQLite with %d rows", table_name, len(table_df))

    logger.info("Warehouse database saved at %s", db_path)


def build_data_warehouse() -> None:
    """Orchestrate end-to-end warehouse build from Silver to Gold."""
    logger.info("Starting warehouse build process (Silver -> Gold).")

    df = load_cleaned_dataset()
    tables = build_star_schema_tables(df)
    save_tables_to_sqlite(tables)

    logger.info("Warehouse build completed successfully.")


if __name__ == "__main__":
    build_data_warehouse()
