import pandas as pd
import os

print("Recalculating standardized Turnover Index (Average of Monthly Turnovers)...")

# 1. Load Data
inv = pd.read_csv('data/datathon-2026-round-1/operational/inventory.csv')

# 2. Calculate Monthly Turnover for each snapshot
# Turnover = Units Sold in month / Stock on hand (at snapshot)
# We use max(stock, 1) to avoid div by zero
inv['monthly_turnover'] = inv['units_sold'] / inv['stock_on_hand'].clip(lower=1)

# 3. Aggregate by Product
turnover_stats = inv.groupby(['product_id', 'product_name', 'category', 'segment']).agg(
    avg_turnover=('monthly_turnover', 'mean'),
    avg_stock=('stock_on_hand', 'mean'),
    total_sold=('units_sold', 'sum')
).reset_index()

# 4. Identify Fast and Slow
# Fast: Highest average monthly turnover
fast_turnover = turnover_stats.sort_values('avg_turnover', ascending=False).head(10)
# Slow: Lowest average monthly turnover
slow_turnover = turnover_stats.sort_values('avg_turnover', ascending=True).head(10)

print("\nTop 10 Fast Turnover Products (Monthly Avg):")
print(fast_turnover[['product_name', 'avg_turnover', 'avg_stock']])

print("\nTop 10 Slow Turnover Products (Monthly Avg):")
print(slow_turnover[['product_name', 'avg_turnover', 'avg_stock']])

# Save for report
fast_turnover.to_csv('reports/business_analysis/tables/inventory_fast_turnover_v2.csv', index=False)
slow_turnover.to_csv('reports/business_analysis/tables/inventory_slow_turnover_v2.csv', index=False)
