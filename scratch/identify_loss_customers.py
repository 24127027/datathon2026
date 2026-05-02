import pandas as pd

print("Identifying Top Customers of Loss-Making/High-Volume Products...")

# 1. Load Data
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv')
prods = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

# 2. Identify Target Products (Loss-Makers)
items = items.merge(prods[['product_id', 'cogs', 'product_name']], on='product_id')
items['profit'] = (items['unit_price'] - items['discount_amount'] - items['cogs']) * items['quantity']

loss_maker_names = ['HanoiStreet UM-10', 'SaigonFlex UM-92', 'SaigonFlex UM-96', 'SaigonFlex UM-80']
loss_maker_ids = prods[prods['product_name'].isin(loss_maker_names)]['product_id'].tolist()

# Filter for these products only
loss_items = items[items['product_id'].isin(loss_maker_ids)].copy()
loss_items = loss_items.merge(orders[['order_id', 'customer_id']], on='order_id')

# 3. Aggregate by Customer
top_loss_customers = loss_items.groupby('customer_id').agg(
    total_loss_qty=('quantity', 'sum'),
    total_loss_amount=('profit', 'sum'),
    order_count=('order_id', 'nunique')
).reset_index()

# Sort by most units bought
top_loss_customers = top_loss_customers.sort_values('total_loss_qty', ascending=False)

# 4. Analysis of behavior
avg_qty_per_order = top_loss_customers['total_loss_qty'].sum() / top_loss_customers['order_count'].sum()
print(f"Average Loss-Items per Order across these customers: {round(avg_qty_per_order, 2)}")

# Rank Top 10
top_10 = top_loss_customers.head(10)
print("\n--- Top 10 Customers of Loss-Making Products ---")
print(top_10)

# Save for report
top_loss_customers.to_csv('reports/business_analysis/tables/top_loss_contributing_customers.csv', index=False)
