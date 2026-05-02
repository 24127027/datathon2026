import pandas as pd
import numpy as np

print("Evaluating Core Products and Identifying Hidden Stars...")

# 1. Load Data
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
inv = pd.read_csv('data/datathon-2026-round-1/operational/inventory.csv')
prods = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

# 2. Calculate Metrics per Product
# Profit per item: (unit_price - discount - cogs)
items = items.merge(prods[['product_id', 'cogs', 'product_name', 'category', 'segment']], on='product_id')
items['profit_per_unit'] = items['unit_price'] - items['discount_amount'] - items['cogs']
items['total_profit'] = items['profit_per_unit'] * items['quantity']
items['net_revenue'] = (items['unit_price'] - items['discount_amount']) * items['quantity']

prod_stats = items.groupby(['product_id', 'product_name', 'category', 'segment']).agg(
    total_qty=('quantity', 'sum'),
    total_actual_profit=('total_profit', 'sum'),
    total_net_revenue=('net_revenue', 'sum'),
    avg_margin_rate=('total_profit', lambda x: x.sum() / items.loc[x.index, 'net_revenue'].sum())
).reset_index()

# Turnover calculation
inv_stats = inv.groupby('product_id')['stock_on_hand'].mean().reset_index()
inv_stats.columns = ['product_id', 'avg_stock']

analysis = prod_stats.merge(inv_stats, on='product_id')
analysis['turnover'] = analysis['total_qty'] / analysis['avg_stock'].replace(0, 1)

# 3. Define Performance Score
# Score = Normalized Margin * Normalized Turnover
analysis['margin_score'] = (analysis['avg_margin_rate'] - analysis['avg_margin_rate'].min()) / (analysis['avg_margin_rate'].max() - analysis['avg_margin_rate'].min())
analysis['turnover_score'] = (analysis['turnover'] - analysis['turnover'].min()) / (analysis['turnover'].max() - analysis['turnover'].min())
analysis['perf_score'] = analysis['margin_score'] * analysis['turnover_score']

# 4. Current Heroes vs Potential
current_heroes = ['SaigonFlex UM-92', 'HanoiStreet UM-10', 'SaigonFlex UM-43', 'SaigonFlex UM-80', 'SaigonFlex UM-96']
current_stats = analysis[analysis['product_name'].isin(current_heroes)]

# Hidden Stars: High Margin (>30%), High Turnover (>20), not a current hero
hidden_stars = analysis[
    (~analysis['product_name'].isin(current_heroes)) & 
    (analysis['avg_margin_rate'] > 0.30) & 
    (analysis['turnover'] > 20)
].sort_values('total_actual_profit', ascending=False)

print("\n--- Current Hero Performance ---")
print(current_stats[['product_name', 'avg_margin_rate', 'turnover', 'total_actual_profit']])

print("\n--- Hidden Stars (High Margin + High Turnover) ---")
print(hidden_stars[['product_name', 'avg_margin_rate', 'turnover', 'total_actual_profit', 'segment']].head(10))

# Export for report
analysis.to_csv('reports/business_analysis/tables/product_hero_evaluation.csv', index=False)
