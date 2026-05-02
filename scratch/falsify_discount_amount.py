import pandas as pd
import numpy as np

oi = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
products = pd.read_csv('data/datathon-2026-round-1/master/products.csv')
promotions = pd.read_csv('data/datathon-2026-round-1/master/promotions.csv')
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', low_memory=False)

# ============================================================
# DEEP DIVE: What is discount_amount relative to?
# It's NOT (price - unit_price), so what is it?
# Hypothesis: discount_amount = discount on total (unit_price * qty)?
#             or discount_amount per item = something else?
# ============================================================

promo_oi = oi[oi['promo_id'].notna()].copy()
promo_oi = promo_oi.merge(products[['product_id','price']], on='product_id', how='left')
promo_oi = promo_oi.merge(promotions[['promo_id','promo_type','discount_value']], on='promo_id', how='left')

pct_mask = promo_oi['promo_type'] == 'percentage'
sub_pct = promo_oi[pct_mask].copy()

# What is discount_amount / (price * qty)?
sub_pct['total_ref'] = sub_pct['price'] * sub_pct['quantity']
sub_pct['disc_over_total'] = sub_pct['discount_amount'] / sub_pct['total_ref'] * 100
sub_pct['disc_over_unit_price'] = sub_pct['discount_amount'] / sub_pct['unit_price'] * 100
sub_pct['disc_over_price'] = sub_pct['discount_amount'] / sub_pct['price'] * 100

# Expected discount rate
sub_pct['expected_disc_pct'] = sub_pct['discount_value']

print("=== discount_amount / (price*qty) stats ===")
print(sub_pct['disc_over_total'].describe())
print()
print("=== discount_amount / unit_price stats ===")
print(sub_pct['disc_over_unit_price'].describe())
print()

# Try: discount_amount = price * qty * discount_pct / 100?
sub_pct['expected_da_total'] = sub_pct['price'] * sub_pct['quantity'] * sub_pct['discount_value'] / 100
sub_pct['residual_da_total'] = (sub_pct['discount_amount'] - sub_pct['expected_da_total']).abs()
sub_pct['residual_pct_total'] = sub_pct['residual_da_total'] / sub_pct['expected_da_total'] * 100

print("=== discount_amount vs price*qty*disc% residual ===")
print(sub_pct['residual_pct_total'].describe())
within_5 = (sub_pct['residual_pct_total'] < 5).mean() * 100
print(f"Within 5%: {within_5:.1f}%")
print()

# Try: discount_amount = unit_price * qty * something?
sub_pct['expected_da_unitbased'] = sub_pct['unit_price'] * sub_pct['quantity'] * sub_pct['discount_value'] / 100
sub_pct['residual_da_unit'] = (sub_pct['discount_amount'] - sub_pct['expected_da_unitbased']).abs()
sub_pct['residual_pct_unit'] = sub_pct['residual_da_unit'] / sub_pct['expected_da_unitbased'] * 100

print("=== discount_amount vs unit_price*qty*disc% residual ===")
print(sub_pct['residual_pct_unit'].describe())
within_5_u = (sub_pct['residual_pct_unit'] < 5).mean() * 100
print(f"Within 5%: {within_5_u:.1f}%")
print()

# What if discount_amount is a TOTAL line discount (sum over all items in order)?
# Try looking at order-level discount
print("=== Sample rows to understand discount_amount structure ===")
cols = ['order_id','product_id','quantity','unit_price','price','discount_value',
        'discount_amount','expected_da_total','residual_da_total']
print(sub_pct[cols].head(20).to_string())
print()

# Key insight test: 
# If unit_price = price * (1 - disc%) + noise, then:
# revenue_per_line = unit_price * qty
# discount_amount should be the discount APPLIED (price*qty - unit_price*qty)
# Let's check: price*qty - unit_price*qty vs discount_amount
sub_pct['line_discount_implied'] = (sub_pct['price'] - sub_pct['unit_price']) * sub_pct['quantity']
sub_pct['residual_vs_implied'] = (sub_pct['discount_amount'] - sub_pct['line_discount_implied']).abs()
sub_pct['residual_pct_implied'] = sub_pct['residual_vs_implied'] / (sub_pct['price'] * sub_pct['quantity']) * 100

print("=== discount_amount vs (price - unit_price)*qty ===")
print(sub_pct['residual_pct_implied'].describe())
within_5_i = (sub_pct['residual_pct_implied'] < 5).mean() * 100
print(f"Within 5%: {within_5_i:.1f}%")
print()

# Final reality check: is discount_amount ALWAYS equal to price*qty*disc_pct?
# regardless of what unit_price is?
print("=== COMPARISON: which formula best explains discount_amount? ===")
print(f"Formula A: price*qty*disc%       -> within 5%: {within_5:.1f}%")
print(f"Formula B: unit_price*qty*disc%  -> within 5%: {within_5_u:.1f}%")
print(f"Formula C: (price-unit_price)*qty -> within 5%: {within_5_i:.1f}%")
print()

# Now check: does this mean unit_price and discount_amount are INDEPENDENT?
# i.e. discount_amount = f(price, qty, disc%) and unit_price = g(price, noise)
# separately, not linked?
# If so, then: actual_revenue = unit_price * qty - discount_amount ??? (double discount?)
# or: actual_revenue = unit_price * qty (and discount_amount is just metadata)?

# Let's check the exact formula by looking at min-residual cases
good_fit_mask = sub_pct['residual_pct_total'] < 1
print(f"Rows where discount_amount ≈ price*qty*disc% (within 1%): {good_fit_mask.sum():,} / {len(sub_pct):,}")
print()
print("Sample of GOOD FIT rows:")
print(sub_pct[good_fit_mask][cols].head(10).to_string())
print()
bad_fit_mask = sub_pct['residual_pct_total'] > 20
print(f"Rows where discount_amount strongly DEVIATES from price*qty*disc%: {bad_fit_mask.sum():,}")
print("Sample of BAD FIT rows:")
print(sub_pct[bad_fit_mask][cols].head(10).to_string())
