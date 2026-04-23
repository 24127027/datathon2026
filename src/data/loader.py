from pathlib import Path

import pandas as pd


def load_customers(
	data_root: str | Path = "data/datathon-2026-round-1",
	parse_dates: bool = True,
) -> pd.DataFrame:
	csv_path = Path(data_root) / "master" / "customers.csv"
	geo_path = Path(data_root) / "master" / "geography.csv"
	date_cols = ["signup_date"] if parse_dates else None
	customers_df = pd.read_csv(csv_path, parse_dates=date_cols)
	geography_df = pd.read_csv(geo_path).drop(columns=["city"], errors="ignore")
	return customers_df.merge(geography_df, on="zip", how="left")


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


def load_orders(data_root: str | Path = "data/datathon-2026-round-1") -> pd.DataFrame:
	base_path = Path(data_root) / "transaction"
	orders_df = pd.read_csv(base_path / "orders.csv").rename(columns={"order_date": "date"})
	orders_df["date"] = pd.to_datetime(orders_df["date"], errors="coerce")
	payments_df = pd.read_csv(base_path / "payments.csv")
	shipments_df = pd.read_csv(
		base_path / "shipments.csv",
		parse_dates=["ship_date", "delivery_date"],
	)
	customers_df = pd.read_csv(
		Path(data_root) / "master" / "customers.csv",
		parse_dates=["signup_date"],
	)
	geography_df = pd.read_csv(
		Path(data_root) / "master" / "geography.csv"
	).drop(columns=["city"], errors="ignore")
	customers_df = customers_df.merge(geography_df, on="zip", how="left")
	customers_df = customers_df.drop(columns=["zip", "city"], errors="ignore")
	payments_df = payments_df.drop(columns=["payment_method"], errors="ignore")

	orders_with_customers = orders_df.merge(
		customers_df,
		on="customer_id",
		how="left",
	)

	orders_with_payment = orders_with_customers.merge(
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


def load_order_items(data_root: str | Path = "data/datathon-2026-round-1") -> pd.DataFrame:
	base_path = Path(data_root) / "transaction"
	order_items_df = pd.read_csv(
		base_path / "order_items.csv",
		dtype={"promo_id_2": "string"},
	)
	products_df = pd.read_csv(Path(data_root) / "master" / "products.csv")
	returns_df = pd.read_csv(base_path / "returns.csv", parse_dates=["return_date"])
	reviews_df = pd.read_csv(base_path / "reviews.csv", parse_dates=["review_date"])
	promotions_df = pd.read_csv(
		Path(data_root) / "master" / "promotions.csv",
		parse_dates=["start_date", "end_date"],
	)
	reviews_df = reviews_df.drop(columns=["customer_id"], errors="ignore")

	order_items_with_products = order_items_df.merge(
		products_df,
		on="product_id",
		how="left",
	)

	order_items_with_returns = order_items_with_products.merge(
		returns_df,
		on=["order_id", "product_id"],
		how="left",
	)
	order_items_with_reviews = order_items_with_returns.merge(
		reviews_df,
		on=["order_id", "product_id"],
		how="left",
	)

	promo_columns = [col for col in promotions_df.columns if col != "promo_id"]
	promo_for_id_1 = promotions_df.rename(
		columns={col: f"{col}_promo_1" for col in promo_columns}
	)
	promo_for_id_2 = promotions_df.rename(
		columns={
			"promo_id": "promo_id_2",
			**{col: f"{col}_promo_2" for col in promo_columns},
		}
	)

	with_promo_1 = order_items_with_reviews.merge(
		promo_for_id_1,
		on="promo_id",
		how="left",
	)

	return with_promo_1.merge(
		promo_for_id_2,
		on="promo_id_2",
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
	web_traffic_df = pd.read_csv(csv_path, parse_dates=date_cols)
	if parse_dates:
		mask = web_traffic_df["date"].dt.year != 2012
	else:
		mask = pd.to_datetime(web_traffic_df["date"], errors="coerce").dt.year != 2012
	return web_traffic_df.loc[mask].reset_index(drop=True)


def load_sales(
	data_root: str | Path = "data/datathon-2026-round-1",
	parse_dates: bool = True,
) -> pd.DataFrame:
	csv_path = Path(data_root) / "analytical" / "sales.csv"
	date_cols = ["Date"] if parse_dates else None
	sales_df = pd.read_csv(csv_path, parse_dates=date_cols)
	sales_df = sales_df.rename(columns={"Date": "date"})
	if parse_dates:
		mask = sales_df["date"].dt.year != 2012
	else:
		mask = pd.to_datetime(sales_df["date"], errors="coerce").dt.year != 2012
	return sales_df.loc[mask].reset_index(drop=True)
