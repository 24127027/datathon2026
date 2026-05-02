import pandas as pd
import numpy as np

oi = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
products = pd.read_csv('data/datathon-2026-round-1/master/products.csv')
promotions = pd.read_csv('data/datathon-2026-round-1/master/promotions.csv')
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', low_memory=False)

# ============================================================
# TEST A: Is products.price the MEAN of unit_price (no-promo) per product?
# If products.price = E[unit_price | no promo], then per-product mean(unit_price)
# should be very close to products.price.
# ============================================================
print("=" * 65)
print("TEST A: Is products.price = E[unit_price | no promo, qty=1]?")
print("=" * 65)

no_promo_q1 = oi[
    (oi['promo_id'].isna()) & (oi['promo_id_2'].isna()) & (oi['quantity'] == 1)
].copy()

per_product = no_promo_q1.groupby('product_id').agg(
    mean_up=('unit_price', 'mean'),
    n=('unit_price', 'count')
).reset_index()
per_product = per_product.merge(products[['product_id','price']], on='product_id', how='left')
per_product['pct_diff_mean_vs_price'] = (per_product['mean_up'] - per_product['price']) / per_product['price'] * 100

print(f"\nProducts with >= 5 no-promo, qty=1 observations: {(per_product['n']>=5).sum()}")
sub = per_product[per_product['n'] >= 5]
print(f"Std of (mean_unit_price - products.price) / products.price: {sub['pct_diff_mean_vs_price'].std():.4f}%")
print(f"Mean of above: {sub['pct_diff_mean_vs_price'].mean():.4f}%")
print(f"Max absolute deviation: {sub['pct_diff_mean_vs_price'].abs().max():.4f}%")
print(f"Within 0.5%: {(sub['pct_diff_mean_vs_price'].abs() < 0.5).mean()*100:.1f}%")
print(f"Within 1%:   {(sub['pct_diff_mean_vs_price'].abs() < 1.0).mean()*100:.1f}%")
print(f"Within 2%:   {(sub['pct_diff_mean_vs_price'].abs() < 2.0).mean()*100:.1f}%")

# ============================================================
# TEST B: For promo items, is unit_price = products.price * (1 - disc)?
# vs unit_price = products.price + noise (same as no-promo)?
# We already know residual std ≈ 2% either way — need to compare means.
# ============================================================
print("\n" + "=" * 65)
print("TEST B: For promo items — unit_price is pre- or post-discount?")
print("=" * 65)

promo_oi = oi[oi['promo_id'].notna()].copy()
promo_oi = promo_oi.merge(products[['product_id','price']], on='product_id', how='left')
promo_oi = promo_oi.merge(promotions[['promo_id','promo_type','discount_value']], on='promo_id', how='left')
pct_promo = promo_oi[promo_oi['promo_type'] == 'percentage'].copy()

# Mean of (unit_price / products.price) should be:
# - If pre-discount: ≈ 1.0 (same as no-promo)
# - If post-discount: ≈ (1 - disc%)
pct_promo['ratio_up_to_price'] = pct_promo['unit_price'] / pct_promo['price']

print("\nMean(unit_price / products.price) by promo discount level:")
promo_group = pct_promo.groupby('discount_value').agg(
    n=('ratio_up_to_price','count'),
    mean_ratio=('ratio_up_to_price','mean'),
    expected_if_postdisc=('discount_value', lambda x: 1 - x.iloc[0]/100),
    expected_if_predisc=('discount_value', lambda x: 1.0)
).reset_index()
promo_group['actual'] = promo_group['mean_ratio']
promo_group['err_postdisc'] = (promo_group['actual'] - promo_group['expected_if_postdisc']).abs()
promo_group['err_predisc'] = (promo_group['actual'] - 1.0).abs()
print(promo_group[['discount_value','n','actual','expected_if_postdisc','err_postdisc','err_predisc']].to_string())

# ============================================================
# TEST C: Check if products.price is a SINGLE FIXED value per product
# or derived from some aggregation of order data.
# If products.price is truly an EXTERNAL reference (MSRP/catalog),
# it should NOT exactly equal any specific order's unit_price.
# If it's an aggregate (mean/mode of unit_price), it should be "between" observed values.
# ============================================================
print("\n" + "=" * 65)
print("TEST C: Is products.price between min/max of observed unit_price?")
print("=" * 65)

all_no_promo = oi[(oi['promo_id'].isna()) & (oi['promo_id_2'].isna())].copy()
obs_range = all_no_promo.groupby('product_id').agg(
    min_up=('unit_price','min'),
    max_up=('unit_price','max'),
    mean_up=('unit_price','mean'),
    n=('unit_price','count')
).reset_index()
obs_range = obs_range.merge(products[['product_id','price']], on='product_id', how='left')
obs_range['price_within_range'] = (
    (obs_range['price'] >= obs_range['min_up']) & 
    (obs_range['price'] <= obs_range['max_up'])
)
obs_range['price_above_max'] = obs_range['price'] > obs_range['max_up']
obs_range['price_below_min'] = obs_range['price'] < obs_range['min_up']

print(f"\nProducts with >= 3 no-promo observations: {(obs_range['n']>=3).sum()}")
sub = obs_range[obs_range['n'] >= 3]
print(f"products.price within [min_up, max_up]: {sub['price_within_range'].sum():,} ({sub['price_within_range'].mean()*100:.1f}%)")
print(f"products.price above max_up: {sub['price_above_max'].sum():,}")
print(f"products.price below min_up: {sub['price_below_min'].sum():,}")
print()
print("Gap between products.price and mean_unit_price (no promo):")
sub = obs_range[obs_range['n'] >= 10].copy()
sub['gap_pct'] = (sub['price'] - sub['mean_up']) / sub['mean_up'] * 100
print(sub['gap_pct'].describe())

# ============================================================
# TEST D: Cross-check with payments — what does total revenue look like?
# If revenue = unit_price * qty - discount_amount, does it match payment amounts?
# ============================================================
print("\n" + "=" * 65)
print("TEST D: Revenue formula verification via payments")
print("=" * 65)

payments = pd.read_csv('data/datathon-2026-round-1/transaction/payments.csv', low_memory=False)
print("Payments columns:", payments.columns.tolist())
print(payments.head(3).to_string())
print()

# Compute order-level implied revenue from order_items
order_revenue = oi.groupby('order_id').apply(
    lambda g: (g['unit_price'] * g['quantity'] - g['discount_amount']).sum()
).reset_index()
order_revenue.columns = ['order_id', 'implied_revenue']

# Also try: revenue = unit_price * qty only
order_revenue_nodiscount = oi.groupby('order_id').apply(
    lambda g: (g['unit_price'] * g['quantity']).sum()
).reset_index()
order_revenue_nodiscount.columns = ['order_id', 'implied_revenue_nodiscount']

# Merge with payments
pay_total = payments.groupby('order_id').agg(
    total_paid=('amount', 'sum') if 'amount' in payments.columns else ('payment_amount', 'sum')
).reset_index()
print(f"Payment column used: {[c for c in payments.columns if 'amount' in c.lower() or 'pay' in c.lower()]}")

comp = order_revenue.merge(order_revenue_nodiscount, on='order_id')
comp = comp.merge(pay_total, on='order_id', how='inner')
comp['diff_withdisc'] = (comp['implied_revenue'] - comp['total_paid']).abs() / comp['total_paid'] * 100
comp['diff_nodisc'] = (comp['implied_revenue_nodiscount'] - comp['total_paid']).abs() / comp['total_paid'] * 100

print(f"\nOrders matched to payments: {len(comp):,}")
print(f"\nFormula: unit_price*qty - discount_amount vs payment:")
print(comp['diff_withdisc'].describe())
print(f"Within 1%: {(comp['diff_withdisc'] < 1).mean()*100:.1f}%")
print(f"\nFormula: unit_price*qty (no discount deducted) vs payment:")
print(comp['diff_nodisc'].describe())
print(f"Within 1%: {(comp['diff_nodisc'] < 1).mean()*100:.1f}%")
print()
print("Sample comparisons:")
print(comp[['order_id','implied_revenue','implied_revenue_nodiscount','total_paid']].head(10).to_string())
