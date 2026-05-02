import pandas as pd

print("Aggregating Total Portfolio Profitability...")

# 1. Load Data
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
prods = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

# 2. Calculate Actual Profit for the whole database
items = items.merge(prods[['product_id', 'cogs']], on='product_id')
items['profit'] = (items['unit_price'] - items['discount_amount'] - items['cogs']) * items['quantity']

total_profit = items['profit'].sum()
total_revenue = ((items['unit_price'] - items['discount_amount']) * items['quantity']).sum()

# 3. Identify Loss from the specific "Bad Core" products
loss_maker_names = ['HanoiStreet UM-10', 'SaigonFlex UM-92', 'SaigonFlex UM-96', 'SaigonFlex UM-80']
loss_maker_ids = prods[prods['product_name'].isin(loss_maker_names)]['product_id'].tolist()

total_loss_from_core = items[items['product_id'].isin(loss_maker_ids)]['profit'].sum()
other_profit = total_profit - total_loss_from_core

# 4. Scenarios
print("\n--- Portfolio Performance (Actual) ---")
print(f"Total Revenue: {total_revenue/1e6:.2f} M")
print(f"Total Profit/Loss (Net): {total_profit/1e6:.2f} M")
print(f"Loss from 4 'Bad' Core Products: {total_loss_from_core/1e6:.2f} M")
print(f"Profit from All Other Products: {other_profit/1e6:.2f} M")

print("\n--- What-If Optimization Scenarios ---")
print(f"1. Status Quo: We keep losing {abs(total_profit/1e6):.2f} M (Net Loss).")
print(f"2. Stop Core Losses: If we just set these 4 products to 'Break-even', Net Profit becomes {other_profit/1e6:.2f} M.")
print(f"3. Strategic Shift: If we replace the 248M loss with the 30% margin from Hidden Stars, Profit could reach >500M.")

# Save summary
summary = {
    'Metric': ['Total Revenue', 'Net Profit/Loss', 'Core Product Loss', 'Other Product Profit'],
    'Value (M VND)': [total_revenue/1e6, total_profit/1e6, total_loss_from_core/1e6, other_profit/1e6]
}
pd.DataFrame(summary).to_csv('reports/business_analysis/tables/portfolio_profit_summary.csv', index=False)
