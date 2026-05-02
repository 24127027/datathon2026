import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Updating order distribution chart with CLEARER legends...")
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
FIG_DIR = r'c:\Users\mduy\source\repos\datathon2026\reports\business_analysis\figures'

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()

# Load data
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', parse_dates=['order_date'])
orders['day_of_month'] = orders['order_date'].dt.day

# 1. Distribution by Day of Month
plt.figure(figsize=(12, 6))
day_counts = orders.groupby('day_of_month').size().reset_index(name='order_count')
sns.barplot(data=day_counts, x='day_of_month', y='order_count', color='#4E79A7')

# Add clearer vertical lines and annotations
plt.axvline(x=4, color='red', linestyle='--', linewidth=2, label='Giới hạn nhận lương đầu tháng (Ngày 5)')
plt.axvline(x=24, color='green', linestyle='--', linewidth=2, label='Bắt đầu kỳ lương cuối tháng (Ngày 25)')

plt.title('Phân bổ Đơn hàng theo Ngày trong Tháng (Salary Cycle Analysis)', fontsize=14)
plt.xlabel('Ngày trong tháng')
plt.ylabel('Số lượng đơn hàng')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

save_fig(os.path.join(FIG_DIR, '10_order_distribution_day_of_month.png'))
print("Chart updated with legend.")
