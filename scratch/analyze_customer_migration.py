import pandas as pd

print("Analyzing Customer Migration from Loss-Makers to Profit-Makers...")

# 1. Load Data
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', parse_dates=['order_date'])
prods = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

# 2. Setup Metrics
items = items.merge(prods[['product_id', 'cogs', 'product_name']], on='product_id')
items['profit'] = (items['unit_price'] - items['discount_amount'] - items['cogs']) * items['quantity']
items = items.merge(orders[['order_id', 'customer_id', 'order_date']], on='order_id')

loss_maker_names = ['HanoiStreet UM-10', 'SaigonFlex UM-92', 'SaigonFlex UM-96', 'SaigonFlex UM-80']
loss_maker_ids = set(prods[prods['product_name'].isin(loss_maker_names)]['product_id'])
profit_maker_ids = set(prods[(prods['price'] - prods['cogs']) / prods['price'] > 0.25]['product_id'])

# 3. Find First Purchase for each customer
cust_first_date = items.groupby('customer_id')['order_date'].min().reset_index()
cust_first_date.columns = ['customer_id', 'first_date']

items = items.merge(cust_first_date, on='customer_id')
first_purchases = items[items['order_date'] == items['first_date']]

# Customers whose FIRST purchase included a Loss-Maker
loss_starter_ids = first_purchases[first_purchases['product_id'].isin(loss_maker_ids)]['customer_id'].unique()

# 4. Check for Migration: Did they buy a Profit-Maker LATER?
def analyze_migration(cust_ids):
    later_items = items[(items['customer_id'].isin(cust_ids)) & (items['order_date'] > items['first_date'])]
    
    total_starters = len(cust_ids)
    converted_to_profit = later_items[later_items['product_id'].isin(profit_maker_ids)]['customer_id'].unique()
    num_converted = len(converted_to_profit)
    
    total_profit_from_group = items[items['customer_id'].isin(cust_ids)]['profit'].sum()
    avg_profit_per_head = total_profit_from_group / total_starters if total_starters > 0 else 0
    
    return {
        'Starter Group': 'Loss-Makers',
        'Total Customers': total_starters,
        'Converted to Profit-Makers later': num_converted,
        'Conversion Rate (%)': round(num_converted / total_starters * 100, 2) if total_starters > 0 else 0,
        'Avg Lifetime Profit': round(avg_profit_per_head, 2)
    }

migration_stats = analyze_migration(loss_starter_ids)
print("\n--- Migration Results ---")
print(migration_stats)

# Compare with Profit Starters
profit_starter_ids = first_purchases[first_purchases['product_id'].isin(profit_maker_ids)]['customer_id'].unique()
profit_stats = analyze_migration(profit_starter_ids)
profit_stats['Starter Group'] = 'Profit-Makers'
print(profit_stats)

# Save for report
pd.DataFrame([migration_stats, profit_stats]).to_csv('reports/business_analysis/tables/customer_migration_analysis.csv', index=False)
