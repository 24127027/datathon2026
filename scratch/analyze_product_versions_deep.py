import pandas as pd
import numpy as np

products = pd.read_csv('data/datathon-2026-round-1/master/products.csv')
oi = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)

# The anomalous case: LotusWear products with price ~ 20-70 VND (near zero)
# vs same product with price ~10,000-15,000 VND
lotus_weird = products[products['product_name'].str.startswith('LotusWear')].sort_values('product_id')
print("=== LotusWear products - all entries ===")
print(lotus_weird[['product_id','product_name','size','color','price','cogs']].to_string())
print()

# Identify products with suspiciously low price (< 500 VND, vs normal 5000-20000)
low_price = products[products['price'] < 500].copy()
print(f"=== Products with price < 500 VND: {len(low_price)} ===")
print(low_price[['product_id','product_name','category','price','cogs']].to_string())
print()

# Check if these "near-zero" price products actually appear in order_items
print("=== Order items for near-zero price products ===")
oi_low = oi[oi['product_id'].isin(low_price['product_id'])].copy()
print(f"Total order items referencing near-zero price products: {len(oi_low):,}")
if len(oi_low) > 0:
    oi_low_with_price = oi_low.merge(low_price[['product_id','price']], on='product_id')
    print(oi_low_with_price[['order_id','product_id','quantity','unit_price','price']].head(20).to_string())
print()

# Check product_id ranges for "batches"
# ID 1-535: first batch (no LotusWear dupes?)
# ID 280-399: LotusWear bad prices?
print("=== product_id range 270-395 ===")
batch1 = products[(products['product_id'] >= 270) & (products['product_id'] <= 400)]
print(batch1[['product_id','product_name','price','cogs']].to_string())
print()

# Check COGS reasonableness: is cogs/price ratio reasonable?
products['margin_pct'] = (products['price'] - products['cogs']) / products['price'] * 100
print("=== Margin distribution (all products) ===")
print(products['margin_pct'].describe())
print()
print("Products with margin < 0 (cogs > price):")
print(products[products['margin_pct'] < 0][['product_id','product_name','price','cogs','margin_pct']].to_string())
print()

# Final: what are the "true" duplicate groups?
# Separate: same product_name+size+color but different price
key_fields = ['product_name', 'category', 'segment', 'size', 'color']
groups = products.groupby(key_fields, dropna=False)

print("=== Summary of versioned products by price ratio ===")
version_data = []
for keys, grp in groups:
    if len(grp) > 1:
        grp_s = grp.sort_values('product_id')
        prices = grp_s['price'].tolist()
        ids = grp_s['product_id'].tolist()
        ratio = max(prices) / min(prices) if min(prices) > 0 else np.inf
        version_data.append({
            'product_name': keys[0],
            'size': keys[3],
            'color': keys[4],
            'n_versions': len(grp),
            'price_min': min(prices),
            'price_max': max(prices),
            'ratio_max_to_min': ratio,
            'ids': ids,
            'prices': prices
        })

vdf = pd.DataFrame(version_data)
print("Price ratio (max/min) distribution:")
print(vdf['ratio_max_to_min'].describe())
print()

# Categorize: "data error" (ratio > 100x) vs "legitimate price change" (ratio 0.5x - 2x)
vdf['is_data_error'] = vdf['ratio_max_to_min'] > 100
print(f"Groups with ratio > 100x (likely data errors): {vdf['is_data_error'].sum()}")
print(f"Groups with ratio <= 100x (legitimate price change): {(~vdf['is_data_error']).sum()}")
print()
print("Legitimate price changes (ratio <= 100x):")
legit = vdf[~vdf['is_data_error']].copy()
print(legit[['product_name','size','color','n_versions','price_min','price_max','ratio_max_to_min']].sort_values('ratio_max_to_min', ascending=False).head(30).to_string())
