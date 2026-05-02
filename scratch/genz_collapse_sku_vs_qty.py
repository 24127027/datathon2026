import pandas as pd

print("Analyzing GenZ Segment Collapse: SKU Variety vs. Stock Volume...")

# 1. Load Data
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', parse_dates=['order_date'])
prods = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

orders['year'] = orders['order_date'].dt.year
items_with_year = items.merge(orders[['order_id', 'year']], on='order_id')
genz_pids = prods[prods['category'] == 'GenZ']['product_id'].tolist()

# 2. GenZ Specific Analysis
def get_genz_metrics(year):
    y_items = items_with_year[(items_with_year['year'] == year) & (items_with_year['product_id'].isin(genz_pids))]
    
    # Variety (Models)
    active_skus = y_items['product_id'].nunique()
    
    # New Models (SKUs)
    previous_skus = set(items_with_year[(items_with_year['year'] < year) & (items_with_year['product_id'].isin(genz_pids))]['product_id'])
    current_skus = set(y_items['product_id'])
    new_skus = len(current_skus - previous_skus)
    
    # Volume (Quantity)
    total_qty = y_items['quantity'].sum()
    
    # Depth (Units per SKU)
    units_per_sku = total_qty / active_skus if active_skus > 0 else 0
    
    return {
        'Year': year,
        'Active GenZ SKUs (Models)': active_skus,
        'New GenZ SKUs (New Models)': new_skus,
        'Total GenZ Quantity Sold': total_qty,
        'Avg Depth (Units/Model)': round(units_per_sku, 1)
    }

results = [get_genz_metrics(2018), get_genz_metrics(2019)]
print("\n--- GenZ Metrics Comparison: 2018 vs 2019 ---")
print(pd.DataFrame(results).T)

# Save for report
pd.DataFrame(results).to_csv('reports/business_analysis/tables/genz_collapse_deepdive.csv', index=False)
