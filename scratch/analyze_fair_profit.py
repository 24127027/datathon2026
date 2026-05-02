import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from calendar import monthrange

print("Analyzing Fair Profit Contribution (Accounted for Stockouts)...")
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
FIG_DIR = r'c:\Users\mduy\source\repos\datathon2026\reports\business_analysis\figures'

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()

# 1. Load Data
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', parse_dates=['order_date'])
products = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

# Merge to get dates and COGS
sales = items.merge(orders[['order_id', 'order_date']], on='order_id')
sales = sales.merge(products[['product_id', 'cogs', 'product_name', 'category']], on='product_id')

sales['year'] = sales['order_date'].dt.year
sales['month'] = sales['order_date'].dt.month

# Calculate Profit: (Unit Price * Qty) - (COGS * Qty)
sales['profit'] = (sales['unit_price'] - sales['cogs']) * sales['quantity']

# Aggregate Profit by Product, Year, Month
product_monthly_profit = sales.groupby(['product_id', 'product_name', 'category', 'year', 'month']).agg(
    actual_profit=('profit', 'sum'),
    actual_qty=('quantity', 'sum')
).reset_index()

# 2. Load Inventory Data
inv = pd.read_csv('data/datathon-2026-round-1/operational/inventory.csv')

# Merge Sales Profit with Inventory
df = product_monthly_profit.merge(inv[['product_id', 'year', 'month', 'stockout_days']], 
                                  on=['product_id', 'year', 'month'], how='left')

# Fill missing stockout days with 0 (assuming no stockout if not in inventory file for that period)
df['stockout_days'] = df['stockout_days'].fillna(0)

# 3. Calculate Fair Potential Profit
def get_days_in_month(row):
    try:
        return monthrange(int(row['year']), int(row['month']))[1]
    except:
        return 30

df['days_in_month'] = df.apply(get_days_in_month, axis=1)
df['active_days'] = df['days_in_month'] - df['stockout_days']
df['active_days'] = df['active_days'].clip(lower=1)

df['potential_profit'] = (df['actual_profit'] / df['active_days']) * df['days_in_month']
df['lost_profit'] = df['potential_profit'] - df['actual_profit']

# 4. Find Top Profitable Products (Overall Potential)
top_potential = df.groupby(['product_id', 'product_name', 'category']).agg(
    total_actual_profit=('actual_profit', 'sum'),
    total_potential_profit=('potential_profit', 'sum'),
    total_lost_profit=('lost_profit', 'sum'),
    avg_stockout_days=('stockout_days', 'mean')
).reset_index()

top_potential = top_potential.sort_values('total_potential_profit', ascending=False).head(10)

# 5. Profitability Trend (Top 5 Potential Products)
top5_ids = top_potential['product_id'].head(5).tolist()
trend = df[df['product_id'].isin(top5_ids)].copy()
trend['date'] = pd.to_datetime(trend[['year', 'month']].assign(day=1))

plt.figure(figsize=(12, 6))
for pid in top5_ids:
    p_data = trend[trend['product_id'] == pid].sort_values('date')
    p_name = p_data['product_name'].iloc[0]
    plt.plot(p_data['date'], p_data['potential_profit'] / 1e6, marker='o', label=p_name)

plt.title('Xu hướng Lợi nhuận tiềm năng của Top 5 Sản phẩm (Đã điều chỉnh theo Thiếu hàng)')
plt.ylabel('Lợi nhuận tiềm năng (Triệu VND)')
plt.xlabel('Thời gian')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
save_fig(os.path.join(FIG_DIR, '07_top_potential_profit_trend.png'))

# Save table for report
top_potential['total_actual_profit'] = (top_potential['total_actual_profit'] / 1e6).round(2)
top_potential['total_potential_profit'] = (top_potential['total_potential_profit'] / 1e6).round(2)
top_potential['total_lost_profit'] = (top_potential['total_lost_profit'] / 1e6).round(2)
top_potential.to_csv('reports/business_analysis/tables/top_potential_profit_skus.csv', index=False)

print("Fair profit analysis complete.")
