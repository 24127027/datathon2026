import pandas as pd

print("Testing Loss Leader Hypothesis...")

# 1. Load Data
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv')
prods = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

# 2. Identify Loss-Makers
items = items.merge(prods[['product_id', 'cogs', 'product_name']], on='product_id')
items['profit'] = (items['unit_price'] - items['discount_amount'] - items['cogs']) * items['quantity']

loss_maker_names = ['HanoiStreet UM-10', 'SaigonFlex UM-92', 'SaigonFlex UM-96', 'SaigonFlex UM-80']
loss_maker_ids = prods[prods['product_name'].isin(loss_maker_names)]['product_id'].tolist()

# 3. Customer Segmentation
items_with_cust = items.merge(orders[['order_id', 'customer_id']], on='order_id')
loss_buyer_ids = items_with_cust[items_with_cust['product_id'].isin(loss_maker_ids)]['customer_id'].unique()

def get_metrics(cust_ids, group_name):
    group_items = items_with_cust[items_with_cust['customer_id'].isin(cust_ids)]
    group_orders = orders[orders['customer_id'].isin(cust_ids)]
    
    retention = group_orders.groupby('customer_id').size().mean()
    ltv_profit = group_items.groupby('customer_id')['profit'].sum().mean()
    
    non_loss_profit = group_items[~group_items['product_id'].isin(loss_maker_ids)]['profit'].sum()
    
    return {
        'Group': group_name,
        'Size': len(cust_ids),
        'Avg Orders': round(retention, 2),
        'Avg LTV Profit': round(ltv_profit, 2),
        'Total Non-Loss Profit': round(non_loss_profit, 2)
    }

all_cust_ids = orders['customer_id'].unique()
profit_only_ids = [cid for cid in all_cust_ids if cid not in loss_buyer_ids]

results = [
    get_metrics(loss_buyer_ids, 'Loss-Maker Buyers'),
    get_metrics(profit_only_ids, 'Profit-only Buyers')
]

print("\n--- Results ---")
print(pd.DataFrame(results))

loss_orders = items_with_cust[items_with_cust['product_id'].isin(loss_maker_ids)]['order_id'].unique()
basket_size = items_with_cust[items_with_cust['order_id'].isin(loss_orders)].groupby('order_id').size().mean()
print(f"\nAvg Basket Size for Loss-Maker Orders: {round(basket_size, 2)}")

pd.DataFrame(results).to_csv('reports/business_analysis/tables/loss_leader_test.csv', index=False)
