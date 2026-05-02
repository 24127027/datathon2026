import pandas as pd
import numpy as np

oi = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
products = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

# Focus: no promo, qty=1
no_promo = oi[(oi['promo_id'].isna()) & (oi['promo_id_2'].isna())].copy()
qty1 = no_promo[no_promo['quantity'] == 1].copy()
qty1 = qty1.merge(products[['product_id','price']], on='product_id', how='left')
qty1['pct_diff'] = (qty1['unit_price'] - qty1['price']) / qty1['price'] * 100

# Hypothesis 1: Is pct_diff approximately uniform in [-10%, +10%]?
print("=== Hypothesis 1: Uniform noise in [-10%, 10%] ===")
within_10 = (qty1['pct_diff'].abs() <= 10).sum()
print(f"Within ±10%: {within_10:,} / {len(qty1):,} = {within_10/len(qty1)*100:.1f}%")
within_5 = (qty1['pct_diff'].abs() <= 5).sum()
print(f"Within ±5%: {within_5:,} / {len(qty1):,} = {within_5/len(qty1)*100:.1f}%")
print(f"Std of pct_diff: {qty1['pct_diff'].std():.4f}%")
print()

# Hypothesis 2: Is there a systematic direction (always higher or lower)?
print("=== Hypothesis 2: Systematic bias direction ===")
print(f"unit_price > product_price: {(qty1['pct_diff'] > 0).sum():,} ({(qty1['pct_diff'] > 0).mean()*100:.1f}%)")
print(f"unit_price < product_price: {(qty1['pct_diff'] < 0).sum():,} ({(qty1['pct_diff'] < 0).mean()*100:.1f}%)")
print(f"unit_price == product_price: {(qty1['pct_diff'].abs() < 0.01).sum():,}")
print()

# Hypothesis 3: Could unit_price be price*(1+noise) where noise~Uniform(-x,x)?
# std of Uniform(-a,a) = a/sqrt(3), so a = std*sqrt(3)
std = qty1['pct_diff'].std()
a = std * np.sqrt(3)
print(f"=== Hypothesis 3: Uniform noise model ===")
print(f"Implied noise range: ±{a:.2f}%")
# Check uniformity
percentiles = [5,10,25,50,75,90,95]
print("Percentiles of pct_diff:")
for p in percentiles:
    print(f"  P{p}: {np.percentile(qty1['pct_diff'], p):.4f}%")
print()

# Hypothesis 4: Maybe unit_price reflects negotiated price (customer-specific)
# Check if same product sold at consistently different prices to different orders
print("=== Hypothesis 4: Per-product pct_diff stability ===")
per_product = qty1.groupby('product_id').agg(
    n=('pct_diff','count'),
    mean_pct=('pct_diff','mean'),
    std_pct=('pct_diff','std')
).reset_index()
print(f"Products with consistent bias (|mean|>2% & std<2%): {((per_product['mean_pct'].abs()>2) & (per_product['std_pct']<2)).sum()}")
print(f"Products with random noise (|mean|<0.5% & std>1%): {((per_product['mean_pct'].abs()<0.5) & (per_product['std_pct']>1)).sum()}")
print(per_product[per_product['n']>=10].describe().to_string())
print()

# Hypothesis 5: Check if pct_diff follows a normal distribution (data entry noise?)
from scipy import stats
stat, pval = stats.normaltest(qty1['pct_diff'].dropna())
print(f"=== Hypothesis 5: Normality test ===")
print(f"K2 statistic={stat:.2f}, p-value={pval:.6f}")
print(f"(p<0.05 means NOT normal)")
print()

# Hypothesis 6: Check distribution shape - is it bimodal, uniform, or normal?
print("=== Hypothesis 6: Distribution shape check ===")
# Kurtosis: normal=0, uniform=-1.2
print(f"Kurtosis: {qty1['pct_diff'].kurtosis():.4f} (Normal=0, Uniform=-1.2)")
print(f"Skewness: {qty1['pct_diff'].skew():.4f}")
print()

# Histogram-like: count by 1% bins
edges = np.arange(-10, 11, 1)
counts, _ = np.histogram(qty1['pct_diff'], bins=edges)
print("1%-bin histogram of pct_diff:")
for i, c in enumerate(counts):
    label = f"[{edges[i]:+.0f}%,{edges[i+1]:+.0f}%)"
    bar = '#' * (c // 200)
    print(f"  {label}: {c:5d} {bar}")
print()

# Hypothesis 7: Are there repeating specific pct_diff values? (e.g. always +5%)
top_vals = qty1['pct_diff'].round(2).value_counts().head(20)
print("=== Hypothesis 7: Most common pct_diff values (rounded to 2dp) ===")
print(top_vals)
