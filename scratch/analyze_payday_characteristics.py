import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Analyzing characteristics of Payday Peak orders (25-31)...")
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

df = orders.copy()
df['day'] = df['order_date'].dt.day
df['is_payday'] = df['day'] >= 25

# 1. Payment Method Comparison
pay_comp = df.groupby(['is_payday', 'payment_method']).size().unstack(level=0)
pay_comp_pct = pay_comp.div(pay_comp.sum(axis=0), axis=1) * 100

# 2. AOV comparison
order_rev = items.assign(rev=lambda x: x['unit_price'] * x['quantity']).groupby('order_id')['rev'].sum().reset_index()
# Convert order_id to the same type for merging
order_rev['order_id'] = order_rev['order_id'].astype(str)
df['order_id'] = df['order_id'].astype(str)

df = df.merge(order_rev, on='order_id', how='left')

stats = df.groupby('is_payday').agg(
    avg_aov=('rev', 'mean')
).reset_index()

print("Payday Stats:")
print(stats)
print("Payment Method Pct:")
print(pay_comp_pct)

# Plotting
plt.figure(figsize=(10, 6))
pay_comp_pct.plot(kind='bar', ax=plt.gca(), color=['#94AECC', '#E15759'])
plt.title('So sánh Phương thức thanh toán: Ngày thường vs Kỳ nhận lương (25-31)')
plt.ylabel('Tỷ lệ %')
plt.xlabel('Phương thức thanh toán')
plt.legend(['Ngày thường (1-24)', 'Kỳ nhận lương (25-31)'], loc='upper right')
plt.xticks(rotation=0)
save_fig(os.path.join(FIG_DIR, '10_payday_payment_comparison.png'))

pay_comp_pct.to_csv('reports/business_analysis/tables/payday_payment_comparison.csv')
stats.to_csv('reports/business_analysis/tables/payday_stats_comparison.csv', index=False)

print("Analysis complete.")
