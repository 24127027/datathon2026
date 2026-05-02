import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Analyzing year-over-year seasonal product popularity...")
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
FIG_DIR = r'c:\Users\mduy\source\repos\datathon2026\reports\business_analysis\figures'

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()

# Load data
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', parse_dates=['order_date'])
items = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
products = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

# Merge
df = items.merge(orders[['order_id', 'order_date']], on='order_id')
df = df.merge(products[['product_id', 'category']], on='product_id')
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month

# Seasonal popularity by category and year
yearly_cat_seasonal = df.groupby(['year', 'month', 'category'])['quantity'].sum().reset_index()

# Focus on last 4 years
recent_years = sorted(df['year'].unique())[-4:]
df_recent = yearly_cat_seasonal[yearly_cat_seasonal['year'].isin(recent_years)]

# Plotting: Faceted Heatmap
g = sns.FacetGrid(df_recent, col="year", col_wrap=2, height=4, aspect=1.5)
def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    pivot = data.pivot(index='category', columns='month', values='quantity').fillna(0)
    sns.heatmap(pivot, cmap="YlGnBu", **kwargs)

g.map_dataframe(draw_heatmap)
g.set_titles("{col_name}")
g.set_axis_labels("Month", "Category")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Seasonal Trends Over Years (Quantity by Category)', fontsize=14)

save_fig(os.path.join(FIG_DIR, '07_yearly_seasonal_trends.png'))

# Check consistency of top categories per month across years
top_cats_monthly = yearly_cat_seasonal.sort_values(['year', 'month', 'quantity'], ascending=[True, True, False]).groupby(['year', 'month']).head(1)
top_cats_monthly.to_csv('reports/business_analysis/tables/monthly_top_categories_by_year.csv', index=False)

print("Year-over-year seasonal analysis complete.")
