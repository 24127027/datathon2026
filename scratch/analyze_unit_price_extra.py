import pandas as pd
import numpy as np

oi = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
products = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

# Check the multi-quantity no-promo case
no_promo = oi[(oi['promo_id'].isna()) & (oi['promo_id_2'].isna())].copy()
no_promo = no_promo.merge(products[['product_id','price']], on='product_id', how='left')
no_promo['pct_diff'] = (no_promo['unit_price'] - no_promo['price']) / no_promo['price'] * 100

print("=== Pct diff by quantity bucket (no promo) ===")
no_promo['qty_bucket'] = pd.cut(no_promo['quantity'], bins=[0,1,2,3,5,10,np.inf], labels=['1','2','3','4-5','6-10','>10'])
print(no_promo.groupby('qty_bucket')['pct_diff'].describe().to_string())
print()

# Key question: Is unit_price = total_price / quantity, or per-unit price?
# If unit_price is PER UNIT, then total = unit_price * quantity
# Compare: for qty>1, does unit_price * qty match some "round" number?
qty_gt1 = no_promo[no_promo['quantity'] > 1].copy()
qty_gt1['total_implied'] = qty_gt1['unit_price'] * qty_gt1['quantity']
qty_gt1['total_at_ref'] = qty_gt1['price'] * qty_gt1['quantity']
qty_gt1['pct_diff_total'] = (qty_gt1['total_implied'] - qty_gt1['total_at_ref']) / qty_gt1['total_at_ref'] * 100

print("=== For qty>1: pct_diff of (unit_price*qty) vs (price*qty) ===")
print(f"Std of pct_diff (unit_price vs price): {qty_gt1['pct_diff'].std():.4f}%")
print(f"Std of pct_diff (total_implied vs total_ref): {qty_gt1['pct_diff_total'].std():.4f}%")
print("Same since it's a ratio - the error is at the unit level")
print()

# Is discount_amount = 0 for ALL no-promo? Double check
print("=== Discount amount for no-promo items ===")
print(no_promo['discount_amount'].describe())
print(f"discount_amount > 0: {(no_promo['discount_amount'] > 0).sum()}")
print()

# Final: check with-promo to see if unit_price IS post-discount
has_promo = oi[oi['promo_id'].notna()].copy()
has_promo = has_promo.merge(products[['product_id','price']], on='product_id', how='left')
has_promo['pct_diff'] = (has_promo['unit_price'] - has_promo['price']) / has_promo['price'] * 100

print("=== For items WITH promo: pct_diff of unit_price vs product price ===")
print(has_promo['pct_diff'].describe())
print()
print("Percentiles:")
for p in [5,10,25,50,75,90,95]:
    print(f"  P{p}: {np.percentile(has_promo['pct_diff'].dropna(), p):.2f}%")
print()
print(f"unit_price < price (discount applied): {(has_promo['pct_diff'] < -1).mean()*100:.1f}%")
print(f"unit_price > price (markup?): {(has_promo['pct_diff'] > 1).mean()*100:.1f}%")
