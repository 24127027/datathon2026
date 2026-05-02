import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Analyzing Segment Classification: Price vs. Margin vs. Segment...")
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
FIG_DIR = r'c:\Users\mduy\source\repos\datathon2026\reports\business_analysis\figures'

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()

# 1. Load Master Data
prods = pd.read_csv('data/datathon-2026-round-1/master/products.csv')
prods['margin_rate'] = (prods['price'] - prods['cogs']) / prods['price']

# 2. Segment Profiling
segment_profile = prods.groupby('segment').agg(
    avg_price=('price', 'mean'),
    median_price=('price', 'median'),
    avg_margin=('margin_rate', 'mean'),
    min_price=('price', 'min'),
    max_price=('price', 'max'),
    product_count=('product_id', 'count')
).reset_index().sort_values('avg_price', ascending=False)

print("\nSegment Profiles (Master Data):")
print(segment_profile)

# 3. Visualization: Price and Margin distribution by Segment
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
sns.boxplot(data=prods, x='segment', y='price', palette='Set2')
plt.xticks(rotation=45)
plt.title('Phân bổ Giá theo Phân khúc')
plt.ylabel('Giá Niêm yết')

plt.subplot(1, 2, 2)
sns.boxplot(data=prods, x='segment', y='margin_rate', palette='Set3')
plt.xticks(rotation=45)
plt.title('Phân bổ Biên lợi nhuận theo Phân khúc')
plt.ylabel('Biên lợi nhuận gộp (%)')
plt.axhline(0, color='red', linestyle='--')

save_fig(os.path.join(FIG_DIR, '07_segment_price_margin_dist.png'))

# 4. Deep-dive into High Turnover vs Segment
# Need to link turnover from transaction data
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
inv = pd.read_csv('data/datathon-2026-round-1/operational/inventory.csv')

# Calculate turnover per product
sales_qty = items.groupby('product_id')['quantity'].sum().reset_index()
avg_stock = inv.groupby('product_id')['stock_on_hand'].mean().reset_index()
turnover = sales_qty.merge(avg_stock, on='product_id')
turnover['turnover_index'] = turnover['quantity'] / turnover['stock_on_hand'].replace(0, 1)

# Merge back to products
full_prod = prods.merge(turnover[['product_id', 'turnover_index']], on='product_id', how='left')

# Analyze which segment has the most High Turnover products
high_turn_threshold = full_prod['turnover_index'].quantile(0.8)
full_prod['is_high_turnover'] = full_prod['turnover_index'] >= high_turn_threshold

segment_turnover = full_prod.groupby('segment').agg(
    avg_turnover=('turnover_index', 'mean'),
    high_turnover_share=('is_high_turnover', 'mean'),
    avg_margin=('margin_rate', 'mean')
).reset_index().sort_values('avg_turnover', ascending=False)

print("\nTurnover Characteristics by Segment:")
print(segment_turnover)

segment_profile.to_csv('reports/business_analysis/tables/segment_definitions.csv', index=False)
segment_turnover.to_csv('reports/business_analysis/tables/segment_turnover_analysis.csv', index=False)
print("Analysis complete.")
