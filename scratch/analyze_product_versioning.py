import pandas as pd
import numpy as np

products = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

print("=== products.csv overview ===")
print(f"Total rows: {len(products):,}")
print(f"Unique product_id: {products['product_id'].nunique():,}")
print()
print(products.dtypes)
print()

# Identify "duplicate" products: same everything except product_id and price
key_fields = ['product_name', 'category', 'segment', 'size', 'color']

groups = products.groupby(key_fields, dropna=False).agg(
    n_ids=('product_id', 'count'),
    product_ids=('product_id', list),
    prices=('price', list),
    cogs_vals=('cogs', list)
).reset_index()

dupes = groups[groups['n_ids'] > 1].copy()
print(f"Unique (name,cat,seg,size,color) combos: {len(groups):,}")
print(f"Combos appearing more than once (price-versioned): {len(dupes):,}")
print(f"Total products involved in versioning: {dupes['n_ids'].sum():,}")
print()
print("Distribution of version counts:")
print(dupes['n_ids'].value_counts().sort_index())
print()

# Show some examples
print("=== Examples of price-versioned products ===")
for _, row in dupes.head(10).iterrows():
    ids = sorted(zip(row['product_ids'], row['prices'], row['cogs_vals']), key=lambda x: x[0])
    print(f"\n{row['product_name']} | {row['category']} | {row['segment']} | {row['size']} | {row['color']}")
    for pid, price, cogs in ids:
        margin = (price - cogs) / price * 100
        print(f"  product_id={pid:4d}  price={price:>12,.2f}  cogs={cogs:>12,.2f}  margin={margin:.1f}%")

# Are the IDs sequential (i.e. higher ID = later/newer price)?
print("\n=== Is higher product_id = newer price (monotonic ID assignment)? ===")
dupes['min_id'] = dupes['product_ids'].apply(min)
dupes['max_id'] = dupes['product_ids'].apply(max)
dupes['price_at_min_id'] = dupes.apply(
    lambda r: r['prices'][r['product_ids'].index(r['min_id'])], axis=1
)
dupes['price_at_max_id'] = dupes.apply(
    lambda r: r['prices'][r['product_ids'].index(r['max_id'])], axis=1
)
dupes['price_went_up'] = dupes['price_at_max_id'] > dupes['price_at_min_id']
print(f"Higher ID has higher price (price increase): {dupes['price_went_up'].sum()} / {len(dupes)}")
print(f"Higher ID has lower price (price decrease): {(~dupes['price_went_up']).sum()} / {len(dupes)}")
print()

# Check price change magnitude
dupes['pct_change'] = (dupes['price_at_max_id'] - dupes['price_at_min_id']) / dupes['price_at_min_id'] * 100
print("Price change % (min_id -> max_id):")
print(dupes['pct_change'].describe())
print()

# Do the product_ids cluster in ranges? (e.g. batch 1: 1-500, batch 2: 501-1000, ...)
print("=== Product ID range analysis ===")
print(f"Min product_id: {products['product_id'].min()}")
print(f"Max product_id: {products['product_id'].max()}")
print(f"product_id gaps (discontinuities):")
ids_sorted = sorted(products['product_id'].unique())
gaps = [(ids_sorted[i], ids_sorted[i+1], ids_sorted[i+1]-ids_sorted[i]) 
        for i in range(len(ids_sorted)-1) if ids_sorted[i+1]-ids_sorted[i] > 1]
print(f"  Total gaps > 1: {len(gaps)}")
if gaps:
    for g in gaps[:20]:
        print(f"  After {g[0]} -> next {g[1]} (gap={g[2]})")
print()

# What fraction of order_items reference "old" vs "new" version of same product?
oi = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', low_memory=False)

# Map product_id to version group
id_to_group = {}
for _, row in dupes.iterrows():
    sorted_ids = sorted(row['product_ids'])
    for rank, pid in enumerate(sorted_ids):
        id_to_group[pid] = {
            'group_key': f"{row['product_name']}|{row['size']}|{row['color']}",
            'version': rank + 1,
            'total_versions': len(sorted_ids),
        }

oi_versioned = oi[oi['product_id'].isin(id_to_group.keys())].copy()
oi_versioned['version'] = oi_versioned['product_id'].map(lambda x: id_to_group[x]['version'])
oi_versioned['total_versions'] = oi_versioned['product_id'].map(lambda x: id_to_group[x]['total_versions'])

print(f"=== Order items referencing versioned products ===")
print(f"Total order_items: {len(oi):,}")
print(f"Items on versioned products: {len(oi_versioned):,} ({len(oi_versioned)/len(oi)*100:.1f}%)")
print()
print("By version rank (1=lowest product_id):")
print(oi_versioned.groupby('version').size())
print()

# Are different versions sold in different time periods?
oi_versioned = oi_versioned.merge(orders[['order_id','order_date']], on='order_id', how='left')
oi_versioned['order_date'] = pd.to_datetime(oi_versioned['order_date'])
oi_versioned['year'] = oi_versioned['order_date'].dt.year

print("=== Version usage by year ===")
print(oi_versioned.groupby(['year','version']).size().unstack(fill_value=0).to_string())
