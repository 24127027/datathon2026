import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Investigating the September Break Point...")

# 1. Load Data
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', parse_dates=['order_date'])
inv = pd.read_csv('data/datathon-2026-round-1/operational/inventory.csv')
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)

# 2. Analyze Monthly Trends (Normalized)
orders['month'] = orders['order_date'].dt.month
monthly_avg = orders.groupby('month').size() / orders['order_date'].dt.year.nunique()

# 3. Analyze September Specifics
# Factors to check: Stockout, Returns, Promo, Category Shift
sep_inv = inv[inv['month'] == 9].groupby('year').agg(
    avg_stockout=('stockout_days', 'mean'),
    avg_stock=('stock_on_hand', 'mean')
).mean()

all_inv = inv.groupby('month').agg(
    avg_stockout=('stockout_days', 'mean'),
    avg_stock=('stock_on_hand', 'mean')
)

# Category mix in September vs August and October
def get_cat_mix(month):
    m_orders = orders[orders['order_date'].dt.month == month]['order_id']
    m_items = items[items['order_id'].isin(m_orders)]
    m_items = m_items.merge(pd.read_csv('data/datathon-2026-round-1/master/products.csv')[['product_id', 'category']], on='product_id')
    return m_items['category'].value_counts(normalize=True)

aug_mix = get_cat_mix(8)
sep_mix = get_cat_mix(9)
oct_mix = get_cat_mix(10)

print("\n--- Monthly Comparison (Avg Orders) ---")
print(monthly_avg)

print("\n--- Inventory Health (September vs Others) ---")
print(all_inv)

print("\n--- Category Mix Transition ---")
print("August:\n", aug_mix)
print("September:\n", sep_mix)
print("October:\n", oct_mix)

# 4. Promo impact in September
sep_orders = orders[orders['order_date'].dt.month == 9]['order_id']
sep_promos = items[items['order_id'].isin(sep_orders)]['promo_id'].notna().mean()
aug_promos = items[items['order_id'].isin(orders[orders['order_date'].dt.month == 8]['order_id'])]['promo_id'].notna().mean()

print(f"\nPromo Share August: {aug_promos:.2%}")
print(f"Promo Share September: {sep_promos:.2%}")

# Visualization of the "Break"
plt.figure(figsize=(10, 5))
monthly_avg.plot(kind='line', marker='o', color='red', linewidth=2)
plt.axvline(x=9, color='gray', linestyle='--', label='September')
plt.title('Lượng đơn hàng trung bình theo tháng (2012-2022)')
plt.ylabel('Số lượng đơn hàng')
plt.xticks(range(1, 13))
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(r'c:\Users\mduy\source\repos\datathon2026\reports\business_analysis\figures\09_september_break_analysis.png')
