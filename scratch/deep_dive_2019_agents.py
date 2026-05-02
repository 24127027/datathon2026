import pandas as pd
import numpy as np

print("Deep-Dive Investigation of the 2019 Structural Break...")

# 1. Load Data
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', parse_dates=['order_date'])
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
inv = pd.read_csv('data/datathon-2026-round-1/operational/inventory.csv')
prods = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

orders['year'] = orders['order_date'].dt.year
# Merge items with year for easier filtering
items_with_year = items.merge(orders[['order_id', 'year']], on='order_id')

# 2. Factor Analysis by Year (2018 vs 2019)
def get_year_metrics(year):
    y_orders = orders[orders['year'] == year]
    y_items = items_with_year[items_with_year['year'] == year]
    y_inv = inv[inv['year'] == year]
    
    # Supply
    units_received = y_inv['units_received'].sum()
    
    # Product Variety
    active_skus = y_items['product_id'].nunique()
    # New SKUs: present in this year but not in any previous year
    previous_skus = set(items_with_year[items_with_year['year'] < year]['product_id'])
    current_skus = set(y_items['product_id'])
    new_skus = len(current_skus - previous_skus)
    
    # Marketing
    promo_share = y_items['promo_id'].notna().mean()
    # Discount rate calculation
    total_revenue_potential = (y_items['unit_price'] * y_items['quantity']).sum()
    total_discount_given = (y_items['discount_amount'] * y_items['quantity']).sum()
    avg_discount_rate = total_discount_given / total_revenue_potential if total_revenue_potential > 0 else 0
    
    # Operational
    avg_stockout_days = y_inv['stockout_days'].mean()
    stockout_flag_rate = y_inv['stockout_flag'].mean()
    
    # Customer Behavior
    avg_basket_size = y_items.groupby('order_id')['quantity'].sum().mean()
    
    return {
        'Year': year,
        'Orders': len(y_orders),
        'Units Received': units_received,
        'Active SKUs': active_skus,
        'New SKUs Introduced': new_skus,
        'Promo Share (%)': round(promo_share * 100, 2),
        'Avg Discount (%)': round(avg_discount_rate * 100, 2),
        'Avg Stockout Days': round(avg_stockout_days, 2),
        'Stockout Incident Rate (%)': round(stockout_flag_rate * 100, 2),
        'Avg Basket Size': round(avg_basket_size, 2)
    }

results = [get_year_metrics(2018), get_year_metrics(2019)]
df_res = pd.DataFrame(results)

print("\n--- Comparative Metrics: 2018 vs 2019 ---")
print(df_res.T)

# 3. Category Impact
y18_items = items_with_year[items_with_year['year'] == 2018].merge(prods[['product_id', 'category']], on='product_id')
y19_items = items_with_year[items_with_year['year'] == 2019].merge(prods[['product_id', 'category']], on='product_id')

cat_18 = y18_items.groupby('category')['quantity'].sum()
cat_19 = y19_items.groupby('category')['quantity'].sum()
cat_change = ((cat_19 - cat_18) / cat_18 * 100).sort_values()

print("\n--- Quantity Change by Category (%) ---")
print(cat_change)

# 4. Check for Wholesaler Churn
cust_18 = orders[orders['year'] == 2018]['customer_id'].nunique()
cust_19 = orders[orders['year'] == 2019]['customer_id'].nunique()
print(f"\nUnique Customers 2018: {cust_18}")
print(f"Unique Customers 2019: {cust_19}")
print(f"Customer Growth: {(cust_19 - cust_18) / cust_18 * 100:.2f}%")

df_res.to_csv('reports/business_analysis/tables/break_2019_deep_dive.csv', index=False)
