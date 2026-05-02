import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Analyzing High Turnover Products: Inventory vs. Sales vs. Promo Impact...")
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
FIG_DIR = r'c:\Users\mduy\source\repos\datathon2026\reports\business_analysis\figures'

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()

# 1. Load Data
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
inv = pd.read_csv('data/datathon-2026-round-1/operational/inventory.csv')
prods = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

# 2. Process Order Items for Promo & Margin
# Calculate Profit per item: (unit_price - discount - cogs) * quantity
items = items.merge(prods[['product_id', 'cogs', 'product_name', 'category']], on='product_id')
items['total_revenue'] = items['unit_price'] * items['quantity']
items['total_discount'] = items['discount_amount'] * items['quantity']
items['net_revenue'] = items['total_revenue'] - items['total_discount']
items['total_cogs'] = items['cogs'] * items['quantity']
items['total_profit'] = items['net_revenue'] - items['total_cogs']
items['has_promo'] = items['promo_id'].notna() | items['promo_id_2'].notna()

# Aggregate by Product
prod_stats = items.groupby(['product_id', 'product_name', 'category']).agg(
    total_qty=('quantity', 'sum'),
    avg_unit_price=('unit_price', 'mean'),
    avg_discount_rate=('total_discount', lambda x: x.sum() / items.loc[x.index, 'total_revenue'].sum()),
    promo_share=('has_promo', 'mean'),
    margin_rate=('total_profit', lambda x: x.sum() / items.loc[x.index, 'net_revenue'].sum())
).reset_index()

# 3. Process Inventory for Turnover & Stock Levels
# inv has: product_id, stock_on_hand, units_sold, year, month
inv_stats = inv.groupby('product_id').agg(
    avg_stock=('stock_on_hand', 'mean'),
    total_sold_inv=('units_sold', 'sum')
).reset_index()

# Merge all
analysis = prod_stats.merge(inv_stats, on='product_id')
# Calculate Turnover Proxy: Sold / Avg Stock
analysis['turnover'] = analysis['total_qty'] / analysis['avg_stock'].replace(0, 1)

# 4. Analysis Categories
# Define High Turnover as top 20%
turnover_threshold = analysis['turnover'].quantile(0.8)
analysis['turnover_cat'] = analysis['turnover'].apply(lambda x: 'High' if x >= turnover_threshold else 'Normal')

# 5. Visualization: Turnover vs Margin colored by Promo Share
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(data=analysis, x='turnover', y='margin_rate', 
                hue='avg_discount_rate', size='total_qty', 
                sizes=(20, 400), alpha=0.6, palette='coolwarm')
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.axvline(turnover_threshold, color='black', linestyle='--', alpha=0.5, label='Ngưỡng High Turnover')
plt.title('Mối quan hệ giữa Turnover, Biên lợi nhuận và Chiết khấu')
plt.xlabel('Chỉ số Turnover (Vòng quay)')
plt.ylabel('Biên lợi nhuận gộp (%)')
plt.legend(title='Tỷ lệ chiết khấu', bbox_to_anchor=(1.05, 1), loc='upper left')
save_fig(os.path.join(FIG_DIR, '10_turnover_margin_promo_scatter.png'))

# 6. Detailed Table for High Turnover, Low Margin
low_margin_high_turnover = analysis[(analysis['turnover_cat'] == 'High') & (analysis['margin_rate'] < 0.05)].sort_values('turnover', ascending=False)
low_margin_high_turnover.head(10).to_csv('reports/business_analysis/tables/high_turnover_low_margin_deepdive.csv', index=False)

# Summary of Findings
high_turn_stats = analysis.groupby('turnover_cat').agg(
    avg_stock=('avg_stock', 'mean'),
    avg_sales=('total_qty', 'mean'),
    avg_margin=('margin_rate', 'mean'),
    avg_promo_share=('promo_share', 'mean'),
    avg_discount=('avg_discount_rate', 'mean')
)
print(high_turn_stats)

high_turn_stats.to_csv('reports/business_analysis/tables/turnover_category_comparison.csv')
print("Deep-dive analysis complete.")
