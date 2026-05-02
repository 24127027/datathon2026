import pandas as pd
from src.data.loader import load_orders, load_order_items, load_sales

print("Loading data...")
o = load_orders()
oi = load_order_items()
s = load_sales()

print("-" * 40)
print("ANOMALY CHECK LIST:")

# 1. Total Revenue Check vs sales.csv
oi['calculated_rev'] = oi['quantity'] * oi['unit_price'] - oi['discount_amount']
print(f"1. Total computed revenue (qty*price - discount): {oi['calculated_rev'].sum():,.2f}")
print(f"   Total Sales.csv revenue: {s['Revenue'].sum():,.2f}")
print(f"   -> Difference: {abs(s['Revenue'].sum() - oi['calculated_rev'].sum()):,.2f}")

# 2. Refund check
print(f"2. Total returned amount (refund_amount): {oi['refund_amount'].sum():,.2f}")

# 3. Negative Quantities or Prices
print(f"3. Negative Quantity check -> {(oi['quantity'] < 0).sum()} rows have < 0 quantity.")
print(f"   Negative Unit Price check -> {(oi['unit_price'] < 0).sum()} rows have < 0 unit_price.")
print(f"   Negative Daily Revenue check -> {(s['Revenue'] < 0).sum()} days have < 0 revenue in sales.csv.")

# 4. Chronological dates (Delivery < Order Date)
o['date'] = pd.to_datetime(o['date'])
o['delivery_date'] = pd.to_datetime(o['delivery_date'], errors='coerce')
bad_dates = (o['delivery_date'] < o['date']).sum()
print(f"4. Delivery Date BEFORE Order Date -> {bad_dates} orders found with this anomaly.")

# 5. COGS > Revenue (Selling at a loss)
loss_items = (oi['cogs'] > oi['calculated_rev']).sum()
total_loss_amount = (oi['cogs'] - oi['calculated_rev'])[oi['cogs'] > oi['calculated_rev']].sum()
print(f"5. Items sold at a loss (COGS > Computed Revenue): {loss_items:,.0f} rows.")
print(f"   Total accumulated loss from these specific items: {total_loss_amount:,.2f}")

# 6. Duplicated Sales dates
dup_sales = s['date'].duplicated().sum()
print(f"6. Duplicated dates in sales.csv: {dup_sales} days.")

# 7. Unmatched Orders (Items that do not belong to an order, or orders with no items)
orphan_items = oi[~oi['order_id'].isin(o['order_id'])].shape[0]
orphan_orders = o[~o['order_id'].isin(oi['order_id'])].shape[0]
print(f"7. Orphan checks:")
print(f"   Order items with no matching order: {orphan_items}")
print(f"   Orders with no matching items: {orphan_orders}")
