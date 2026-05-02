import pandas as pd

print("Comparing Daily Order Statistics (2018 vs 2019)...")

# 1. Load Data
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', parse_dates=['order_date'])
orders['year'] = orders['order_date'].dt.year

# 2. Daily Counts
daily_orders = orders.groupby(orders['order_date'].dt.date).size().reset_index(name='count')
daily_orders['order_date'] = pd.to_datetime(daily_orders['order_date'])
daily_orders['year'] = daily_orders['order_date'].dt.year

# 3. Stats by Year
stats = daily_orders.groupby('year')['count'].agg(['count', 'mean', 'std', 'max', 'sum']).reset_index()
print(stats[stats['year'].isin([2017, 2018, 2019, 2020])])

# 4. Check for Gaps
def check_gaps(year):
    y_data = daily_orders[daily_orders['year'] == year]
    expected_days = 365
    actual_days = y_data['order_date'].nunique()
    return actual_days

print(f"\nDays with data in 2018: {check_gaps(2018)}")
print(f"Days with data in 2019: {check_gaps(2019)}")
