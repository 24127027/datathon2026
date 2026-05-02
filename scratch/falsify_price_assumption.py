import pandas as pd
import numpy as np

oi = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
products = pd.read_csv('data/datathon-2026-round-1/master/products.csv')
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', low_memory=False)
promotions = pd.read_csv('data/datathon-2026-round-1/master/promotions.csv')

# ============================================================
# TEST 1: Validate promo items — does unit_price = price * (1 - discount)?
# If TRUE, then unit_price is definitively the post-discount price,
# AND products.price is the PRE-discount base price used to compute discount.
# ============================================================
print("=" * 70)
print("TEST 1: Can we explain unit_price for promo items via discount math?")
print("=" * 70)

promo_oi = oi[oi['promo_id'].notna()].copy()
promo_oi = promo_oi.merge(products[['product_id', 'price']], on='product_id', how='left')
promo_oi = promo_oi.merge(promotions[['promo_id', 'promo_type', 'discount_value']], on='promo_id', how='left')

# For percentage discount: expected = price * (1 - discount_value/100)
# For fixed discount: expected = price - discount_value (per unit)
pct_mask = promo_oi['promo_type'] == 'percentage'
fix_mask = promo_oi['promo_type'] == 'fixed'

promo_oi.loc[pct_mask, 'expected_unit_price'] = (
    promo_oi.loc[pct_mask, 'price'] * (1 - promo_oi.loc[pct_mask, 'discount_value'] / 100)
)
promo_oi.loc[fix_mask, 'expected_unit_price'] = (
    promo_oi.loc[fix_mask, 'price'] - promo_oi.loc[fix_mask, 'discount_value']
)

promo_oi['residual_pct'] = (
    (promo_oi['unit_price'] - promo_oi['expected_unit_price'])
    / promo_oi['expected_unit_price'] * 100
)

print(f"\nPercentage-discount items: {pct_mask.sum():,}")
sub_pct = promo_oi[pct_mask]
print(f"  Std of residual (unit_price vs expected): {sub_pct['residual_pct'].std():.4f}%")
print(f"  Median residual: {sub_pct['residual_pct'].median():.4f}%")
print(f"  Within ±5%: {(sub_pct['residual_pct'].abs() <= 5).mean()*100:.1f}%")

print(f"\nFixed-discount items: {fix_mask.sum():,}")
sub_fix = promo_oi[fix_mask]
print(f"  Std of residual (unit_price vs expected): {sub_fix['residual_pct'].std():.4f}%")
print(f"  Median residual: {sub_fix['residual_pct'].median():.4f}%")
print(f"  Within ±5%: {(sub_fix['residual_pct'].abs() <= 5).mean()*100:.1f}%")

print("\nSample promo items (pct discount):")
cols = ['order_id','product_id','unit_price','price','discount_value','expected_unit_price','residual_pct']
print(sub_pct[cols].head(10).to_string())

# ============================================================
# TEST 2: For versioned products, does unit_price match
# the referenced product_id's price better than the other version?
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: For versioned products, does unit_price match its own product_id's price?")
print("=" * 70)

key_fields = ['product_name', 'category', 'segment', 'size', 'color']
groups = products.groupby(key_fields, dropna=False).filter(lambda x: len(x) > 1)
versioned_pids = set(groups['product_id'].tolist())

# Build mapping: product_id -> sibling product_ids and their prices
pid_to_siblings = {}
for _, grp in products[products['product_id'].isin(versioned_pids)].groupby(key_fields, dropna=False):
    pids = grp['product_id'].tolist()
    prices = dict(zip(grp['product_id'], grp['price']))
    for pid in pids:
        pid_to_siblings[pid] = {k: v for k, v in prices.items() if k != pid}

# For versioned order items (no promo, qty=1 for clean comparison)
oi_v = oi[
    (oi['product_id'].isin(versioned_pids)) &
    (oi['promo_id'].isna()) &
    (oi['promo_id_2'].isna()) &
    (oi['quantity'] == 1)
].copy()

oi_v = oi_v.merge(products[['product_id','price']], on='product_id', how='left')
oi_v['diff_own'] = (oi_v['unit_price'] - oi_v['price']).abs()

def min_sibling_diff(row):
    siblings = pid_to_siblings.get(row['product_id'], {})
    if not siblings:
        return np.nan
    return min(abs(row['unit_price'] - p) for p in siblings.values())

oi_v['diff_sibling'] = oi_v.apply(min_sibling_diff, axis=1)
oi_v['closer_to_own'] = oi_v['diff_own'] < oi_v['diff_sibling']

print(f"\nVersioned product order items (no promo, qty=1): {len(oi_v):,}")
print(f"  unit_price closer to OWN product_id's price: {oi_v['closer_to_own'].sum():,} ({oi_v['closer_to_own'].mean()*100:.1f}%)")
print(f"  unit_price closer to SIBLING product_id's price: {(~oi_v['closer_to_own']).sum():,} ({(~oi_v['closer_to_own']).mean()*100:.1f}%)")

# ============================================================
# TEST 3: Is the noise correlated with products.price magnitude?
# (multiplicative noise vs additive noise)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Is the noise multiplicative (% of price) or additive (fixed VND)?")
print("=" * 70)

no_promo = oi[(oi['promo_id'].isna()) & (oi['promo_id_2'].isna()) & (oi['quantity'] == 1)].copy()
no_promo = no_promo.merge(products[['product_id','price']], on='product_id', how='left')
no_promo['abs_noise'] = (no_promo['unit_price'] - no_promo['price']).abs()
no_promo['pct_noise'] = no_promo['abs_noise'] / no_promo['price'] * 100

# If multiplicative: abs_noise should correlate with price
# If additive: abs_noise should be roughly constant regardless of price
corr_mult = no_promo[['price','abs_noise']].corr().iloc[0,1]
corr_pct = no_promo[['price','pct_noise']].corr().iloc[0,1]
print(f"\nCorrelation(price, abs_noise): {corr_mult:.4f}  [high = multiplicative]")
print(f"Correlation(price, pct_noise): {corr_pct:.4f}  [near 0 = pct is price-independent]")

# Check std of absolute noise by price decile
no_promo['price_decile'] = pd.qcut(no_promo['price'], q=10, labels=False)
print("\nStd of absolute noise and % noise by price decile:")
print(no_promo.groupby('price_decile').agg(
    price_mean=('price','mean'),
    abs_noise_std=('abs_noise','std'),
    pct_noise_std=('pct_noise','std'),
    n=('price','count')
).to_string())

# ============================================================
# TEST 4: Does discount_amount in order_items equal (price - unit_price)?
# This would confirm whether discount_amount is relative to products.price
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: For promo items, does discount_amount = price - unit_price?")
print("=" * 70)

promo_items_check = oi[oi['promo_id'].notna()].copy()
promo_items_check = promo_items_check.merge(products[['product_id','price']], on='product_id', how='left')
promo_items_check['expected_discount'] = promo_items_check['price'] - promo_items_check['unit_price']
promo_items_check['discount_residual'] = (
    promo_items_check['discount_amount'] - promo_items_check['expected_discount']
).abs()
promo_items_check['discount_pct_residual'] = (
    promo_items_check['discount_residual'] / promo_items_check['price'] * 100
)

print(f"\nResidual of (discount_amount vs price - unit_price):")
print(promo_items_check['discount_pct_residual'].describe())
within_1pct = (promo_items_check['discount_pct_residual'] < 1).mean() * 100
print(f"\nWithin 1% of price: {within_1pct:.1f}%")
print("\nSample:")
print(promo_items_check[['product_id','unit_price','price','discount_amount','expected_discount','discount_residual']].head(15).to_string())
