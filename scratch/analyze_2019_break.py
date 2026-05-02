import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Investigating the 2019 Structural Break...")

# 1. Load Data
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', parse_dates=['order_date'])
inv = pd.read_csv('data/datathon-2026-round-1/operational/inventory.csv')
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)

# 2. Yearly Aggregate Trends
orders['year'] = orders['order_date'].dt.year
yearly_orders = orders.groupby('year').size()

# 3. Analyze 2019 Specifics vs 2018 and 2020
yearly_inv = inv.groupby('year').agg(
    avg_stockout=('stockout_days', 'mean'),
    avg_stock=('stock_on_hand', 'mean'),
    total_received=('units_received', 'sum')
)

# Return rate check
returns = pd.read_csv('data/datathon-2026-round-1/transaction/returns.csv')
returns_with_date = returns.merge(orders[['order_id', 'year']], on='order_id')
yearly_returns = returns_with_date.groupby('year').size()
yearly_return_rate = yearly_returns / yearly_orders

print("\n--- Yearly Order Volume ---")
print(yearly_orders)

print("\n--- Inventory Health by Year ---")
print(yearly_inv)

print("\n--- Yearly Return Rates ---")
print(yearly_return_rate)

# Category mix in 2018 vs 2019 vs 2020
def get_cat_mix(year):
    y_orders = orders[orders['year'] == year]['order_id']
    y_items = items[items['order_id'].isin(y_orders)]
    y_items = y_items.merge(pd.read_csv('data/datathon-2026-round-1/master/products.csv')[['product_id', 'category']], on='product_id')
    return y_items['category'].value_counts(normalize=True)

mix_2018 = get_cat_mix(2018)
mix_2019 = get_cat_mix(2019)
mix_2020 = get_cat_mix(2020)

print("\n--- Category Mix Transition (2018-2020) ---")
print("2018:\n", mix_2018)
print("2019:\n", mix_2019)
print("2020:\n", mix_2020)

# 4. Visualization of the 2019 Break
plt.figure(figsize=(10, 5))
yearly_orders.plot(kind='line', marker='o', color='blue', linewidth=2)
plt.axvline(x=2019, color='red', linestyle='--', label='2019 Break')
plt.title('Tổng lượng đơn hàng theo năm (2012-2022)')
plt.ylabel('Số lượng đơn hàng')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(r'c:\Users\mduy\source\repos\datathon2026\reports\business_analysis\figures\yearly_order_break_2019.png')
