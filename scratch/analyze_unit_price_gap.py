import pandas as pd
import numpy as np

oi = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
products = pd.read_csv('data/datathon-2026-round-1/master/products.csv')
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', low_memory=False)

print(f'Total order_items: {len(oi):,}')
print(f'order_items columns: {oi.columns.tolist()}')
print(f'orders columns: {orders.columns.tolist()}')
print()

# Filter: no promotion applied
no_promo = oi[(oi['promo_id'].isna()) & (oi['promo_id_2'].isna())].copy()
print(f'No-promo order_items: {len(no_promo):,}')
print(f'No-promo with qty=1: {(no_promo["quantity"]==1).sum():,}')
print()

# Merge with products
merged = no_promo.merge(products[['product_id','price']], on='product_id', how='left')

# Compare unit_price vs price (all qty, no promo)
merged['diff'] = merged['unit_price'] - merged['price']
merged['pct_diff'] = (merged['diff'] / merged['price']) * 100

# qty=1 only
qty1 = merged[merged['quantity'] == 1].copy()

print(f'qty=1 no-promo rows: {len(qty1):,}')
print(f'Exact match (unit_price == price): {(qty1["diff"].abs() < 0.01).sum():,}')
print(f'Mismatch (|diff| >= 0.01): {(qty1["diff"].abs() >= 0.01).sum():,}')
print()
print('Pct diff stats (qty=1, no promo):')
print(qty1['pct_diff'].describe())
print()

bins = [0, 0.01, 1, 5, 10, 20, 50, 100, np.inf]
labels = ['<0.01%','0.01-1%','1-5%','5-10%','10-20%','20-50%','50-100%','>100%']
qty1['bucket'] = pd.cut(qty1['pct_diff'].abs(), bins=bins, labels=labels)
print('Distribution of |pct_diff| buckets (qty=1, no promo):')
print(qty1['bucket'].value_counts().sort_index())
print()

# Check if the gap is consistent per product_id (same product always has same ratio?)
print('\n=== Per-product unit_price variability (no promo, qty=1) ===')
per_product = qty1.groupby('product_id').agg(
    n=('unit_price', 'count'),
    price_ref=('price', 'first'),
    up_mean=('unit_price', 'mean'),
    up_std=('unit_price', 'std'),
    up_min=('unit_price', 'min'),
    up_max=('unit_price', 'max'),
    pct_diff_mean=('pct_diff', 'mean'),
).reset_index()
per_product['cv'] = per_product['up_std'] / per_product['up_mean']  # coeff of variation
print(per_product[per_product['n'] >= 5].sort_values('cv', ascending=False).head(20).to_string())
print()

# Sample mismatch rows
print('\n=== Sample mismatch rows (qty=1, no promo, |diff|>100) ===')
big_gap = qty1[qty1['diff'].abs() > 100].head(20)
print(big_gap[['order_id','product_id','quantity','unit_price','price','diff','pct_diff']].to_string())

# Check if unit_price = price * quantity (total price pattern)?
print('\n=== Check if unit_price = price * qty (total price hypothesis) ===')
no_promo2 = oi[(oi['promo_id'].isna()) & (oi['promo_id_2'].isna())].copy()
no_promo2 = no_promo2.merge(products[['product_id','price']], on='product_id', how='left')
no_promo2['expected_total'] = no_promo2['price'] * no_promo2['quantity']
no_promo2['diff_total'] = (no_promo2['unit_price'] - no_promo2['expected_total']).abs()
no_promo2['diff_unit'] = (no_promo2['unit_price'] - no_promo2['price']).abs()
# For which is unit_price closer: unit price or total?
no_promo2['closer_to_unit'] = no_promo2['diff_unit'] < no_promo2['diff_total']
print(f'unit_price closer to product price (unit): {no_promo2["closer_to_unit"].sum():,}')
print(f'unit_price closer to product price * qty: {(~no_promo2["closer_to_unit"]).sum():,}')

# Check price trend over time - maybe products.csv price is latest price, unit_price is historical
print('\n=== Check if unit_price / price ratio drifts over time ===')
orders_subset = orders[['order_id', 'order_date']].copy()
no_promo_time = oi[(oi['promo_id'].isna()) & (oi['promo_id_2'].isna())].copy()
no_promo_time = no_promo_time.merge(products[['product_id','price']], on='product_id', how='left')
no_promo_time = no_promo_time.merge(orders_subset, on='order_id', how='left')
no_promo_time['order_date'] = pd.to_datetime(no_promo_time['order_date'])
no_promo_time['year'] = no_promo_time['order_date'].dt.year
no_promo_time['pct_diff'] = (no_promo_time['unit_price'] - no_promo_time['price']) / no_promo_time['price'] * 100

yearly = no_promo_time.groupby('year')['pct_diff'].agg(['mean', 'median', 'std', 'count'])
print(yearly.to_string())
