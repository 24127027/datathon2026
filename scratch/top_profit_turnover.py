import pandas as pd
import os

print("Identifying Top Profitable Products and their Turnover Indices...")

# 1. Load Data
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
inv = pd.read_csv('data/datathon-2026-round-1/operational/inventory.csv')
prods = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

# 2. Calculate Profit per Product
items = items.merge(prods[['product_id', 'cogs', 'product_name', 'category', 'segment']], on='product_id')
items['total_profit'] = (items['unit_price'] - items['discount_amount'] - items['cogs']) * items['quantity']

top_profit_prods = items.groupby(['product_id', 'product_name', 'category', 'segment']).agg(
    total_actual_profit=('total_profit', 'sum'),
    total_qty_sold=('quantity', 'sum')
).reset_index()

# 3. Calculate Turnover Index
inv_stats = inv.groupby('product_id')['stock_on_hand'].mean().reset_index()
inv_stats.columns = ['product_id', 'avg_stock']

# Merge
final_stats = top_profit_prods.merge(inv_stats, on='product_id')
final_stats['turnover_index'] = final_stats['total_qty_sold'] / final_stats['avg_stock'].replace(0, 1)

# 4. Filter and Sort
top_10_profit = final_stats.sort_values('total_actual_profit', ascending=False).head(10)

# Format for output
top_10_profit['total_actual_profit_M'] = (top_10_profit['total_actual_profit'] / 1e6).round(2)
top_10_profit['turnover_index'] = top_10_profit['turnover_index'].round(2)
top_10_profit['avg_stock'] = top_10_profit['avg_stock'].round(1)

print("\nTop 10 Products by Total Actual Profit:")
print(top_10_profit[['product_name', 'segment', 'total_actual_profit_M', 'turnover_index', 'avg_stock']])

top_10_profit.to_csv('reports/business_analysis/tables/top_profit_with_turnover.csv', index=False)
