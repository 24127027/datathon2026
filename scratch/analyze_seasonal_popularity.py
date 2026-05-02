import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Analyzing seasonal product popularity...")
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
FIG_DIR = r'c:\Users\mduy\source\repos\datathon2026\reports\business_analysis\figures'

def save_fig(path):
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()

# Load data
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', parse_dates=['order_date'])
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
products = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

# Merge
df = items.merge(orders[['order_id', 'order_date']], on='order_id')
df = df.merge(products[['product_id', 'category', 'product_name']], on='product_id')
df['month'] = df['order_date'].dt.month

# Seasonal popularity by category
cat_seasonal = df.groupby(['month', 'category'])['quantity'].sum().reset_index()

# Find peak month for each category
top_cats = cat_seasonal.sort_values(['category', 'quantity'], ascending=[True, False]).drop_duplicates('category')
top_cats = top_cats.rename(columns={'month': 'Tháng cao điểm', 'category': 'Danh mục', 'quantity': 'Sản lượng đỉnh'})

# Top products by month (Popular items)
prod_seasonal = df.groupby(['month', 'product_name', 'category'])['quantity'].sum().reset_index()
# Get top product for each month
monthly_hero = prod_seasonal.sort_values(['month', 'quantity'], ascending=[True, False]).groupby('month').head(1)
monthly_hero = monthly_hero.rename(columns={'month': 'Tháng', 'product_name': 'Sản phẩm ưa chuộng nhất', 'category': 'Danh mục', 'quantity': 'Sản lượng'})

# Pivot for visualization
cat_pivot = cat_seasonal.pivot(index='month', columns='category', values='quantity')
cat_pivot = cat_pivot.fillna(0)

# Plotting
plt.figure(figsize=(12, 6))
sns.heatmap(cat_pivot.T, annot=False, cmap="YlGnBu", cbar_kws={'label': 'Sản lượng bán'})
plt.title('Bản đồ nhiệt: Mùa vụ ưa chuộng theo Danh mục sản phẩm')
plt.xlabel('Tháng')
plt.ylabel('Danh mục')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '07_seasonal_category_popularity.png'), dpi=170)
plt.close()

# Save tables
monthly_hero.to_csv('reports/business_analysis/tables/monthly_hero_products.csv', index=False)
top_cats.to_csv('reports/business_analysis/tables/category_peak_months.csv', index=False)

print("Seasonal analysis complete. Tables and heatmap generated.")
