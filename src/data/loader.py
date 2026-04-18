from pathlib import Path

import pandas as pd


def load_customers(
	data_root: str | Path = "data/datathon-2026-round-1",
	parse_dates: bool = True,
) -> pd.DataFrame:
	csv_path = Path(data_root) / "master" / "customers.csv"
	date_cols = ["signup_date"] if parse_dates else None
	return pd.read_csv(csv_path, parse_dates=date_cols)


def load_products(
	data_root: str | Path = "data/datathon-2026-round-1",
	parse_dates: bool = True,
) -> pd.DataFrame:
	csv_path = Path(data_root) / "master" / "products.csv"
	return pd.read_csv(csv_path)


def load_promotions(
	data_root: str | Path = "data/datathon-2026-round-1",
	parse_dates: bool = True,
) -> pd.DataFrame:
	csv_path = Path(data_root) / "master" / "promotions.csv"
	date_cols = ["start_date", "end_date"] if parse_dates else None
	return pd.read_csv(csv_path, parse_dates=date_cols)


def load_orders(
	data_root: str | Path = "data/datathon-2026-round-1",
	parse_dates: bool = True,
) -> pd.DataFrame:
	base_path = Path(data_root) / "transaction"
	order_date_cols = ["order_date"] if parse_dates else None
	shipment_date_cols = ["ship_date", "delivery_date"] if parse_dates else None

	orders_df = pd.read_csv(base_path / "orders.csv", parse_dates=order_date_cols)
	payments_df = pd.read_csv(base_path / "payments.csv")
	shipments_df = pd.read_csv(base_path / "shipments.csv", parse_dates=shipment_date_cols)
	payments_df = payments_df.drop(columns=["payment_method"], errors="ignore")

	orders_with_payment = orders_df.merge(
		payments_df,
		on="order_id",
		how="left",
	)

	eligible_status = {"shipped", "delivered", "returned"}
	eligible_order_ids = orders_with_payment.loc[
		orders_with_payment["order_status"].isin(eligible_status),
		"order_id",
	]
	shipments_filtered = shipments_df[shipments_df["order_id"].isin(eligible_order_ids)]

	return orders_with_payment.merge(
		shipments_filtered,
		on="order_id",
		how="left",
	)


def load_order_items(
	data_root: str | Path = "data/datathon-2026-round-1",
	parse_dates: bool = True,
) -> pd.DataFrame:
	base_path = Path(data_root) / "transaction"
	returns_date_cols = ["return_date"] if parse_dates else None
	reviews_date_cols = ["review_date"] if parse_dates else None

	order_items_df = pd.read_csv(
		base_path / "order_items.csv",
		dtype={"promo_id_2": "string"},
	)
	returns_df = pd.read_csv(base_path / "returns.csv", parse_dates=returns_date_cols)
	reviews_df = pd.read_csv(base_path / "reviews.csv", parse_dates=reviews_date_cols)
	reviews_df = reviews_df.drop(columns=["customer_id"], errors="ignore")

	order_items_with_returns = order_items_df.merge(
		returns_df,
		on=["order_id", "product_id"],
		how="left",
	)

	return order_items_with_returns.merge(
		reviews_df,
		on=["order_id", "product_id"],
		how="left",
	)
 

def load_inventory(
	data_root: str | Path = "data/datathon-2026-round-1",
	parse_dates: bool = True,
) -> pd.DataFrame:
	csv_path = Path(data_root) / "operational" / "inventory.csv"
	date_cols = ["snapshot_date"] if parse_dates else None
	return pd.read_csv(csv_path, parse_dates=date_cols)


def load_web_traffic(
	data_root: str | Path = "data/datathon-2026-round-1",
	parse_dates: bool = True,
) -> pd.DataFrame:
	csv_path = Path(data_root) / "operational" / "web_traffic.csv"
	date_cols = ["date"] if parse_dates else None
	return pd.read_csv(csv_path, parse_dates=date_cols)


def load_sales(
	data_root: str | Path = "data/datathon-2026-round-1",
	parse_dates: bool = True,
) -> pd.DataFrame:
	csv_path = Path(data_root) / "analytical" / "sales.csv"
	date_cols = ["Date"] if parse_dates else None
	return pd.read_csv(csv_path, parse_dates=date_cols)
