import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

print("Analyzing correlation with Vietnamese Lunar Calendar (Approximate)...")
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
FIG_DIR = r'c:\Users\mduy\source\repos\datathon2026\reports\business_analysis\figures'

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()

# 1. Load Order Data
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', parse_dates=['order_date'])

# 2. Lunar Day Approximation Logic
# Reference: Jan 23, 2012 was Lunar 1/1 (New Moon).
# Julian Day for 2012-01-23 is 2455950.
# Average Lunar Month length: 29.53059 days.

def get_julian_day(date):
    return date.to_julian_date()

ref_jd = 2455950.0
lunar_month = 29.53059

orders['jd'] = orders['order_date'].apply(get_julian_day)
orders['lunar_cycle_pos'] = (orders['jd'] - ref_jd) % lunar_month
# Convert to day of month (1 to 30)
orders['lunar_day'] = orders['lunar_cycle_pos'].apply(lambda x: int(round(x)) + 1)
orders.loc[orders['lunar_day'] > 30, 'lunar_day'] = 1 # Wrap around

# 3. Aggregate Sales by Lunar Day
lunar_dist = orders.groupby('lunar_day').size().reset_index(name='order_count')
lunar_dist['avg_per_day'] = lunar_dist['order_count'] / orders['order_date'].nunique() * 30 # Normalized

# 4. Visualization
plt.figure(figsize=(12, 6))
sns.barplot(data=lunar_dist, x='lunar_day', y='order_count', color='teal')
plt.axvline(x=0, color='red', linestyle='--', label='Đầu tháng (Mùng 1)')
plt.axvline(x=14, color='orange', linestyle='--', label='Rằm (15)')
plt.title('Phân bổ Đơn hàng theo Ngày Âm lịch (Xấp xỉ)')
plt.xlabel('Ngày Âm lịch')
plt.ylabel('Tổng số đơn hàng')
plt.legend()
save_fig(os.path.join(FIG_DIR, '10_lunar_day_distribution.png'))

# 5. Check for Specific Lunar Months (Tet)
# Tet is around Lunar Day 1 of Month 1.
# We can find Tet dates precisely for the years.
tet_dates = {
    2012: '2012-01-23', 2013: '2013-02-10', 2014: '2014-01-31', 
    2015: '2015-02-19', 2016: '2016-02-08', 2017: '2017-01-28',
    2018: '2018-02-16', 2019: '2019-02-05', 2020: '2020-01-25',
    2021: '2021-02-12', 2022: '2022-02-01'
}

# Calculate average orders around Tet (15 days before to 5 days after)
tet_results = []
for year, date_str in tet_dates.items():
    tet_dt = pd.to_datetime(date_str)
    start = tet_dt - pd.Timedelta(days=20)
    end = tet_dt + pd.Timedelta(days=10)
    
    mask = (orders['order_date'] >= start) & (orders['order_date'] <= end)
    period_orders = orders[mask].copy()
    period_orders['days_to_tet'] = (period_orders['order_date'] - tet_dt).dt.days
    tet_results.append(period_orders)

tet_df = pd.concat(tet_results)
tet_daily = tet_df.groupby('days_to_tet').size().reset_index(name='order_count')

plt.figure(figsize=(10, 5))
plt.plot(tet_daily['days_to_tet'], tet_daily['order_count'], marker='s', color='darkred', linewidth=2)
plt.axvline(x=0, color='gold', linestyle='-', label='Ngày mùng 1 Tết')
plt.fill_between(tet_daily['days_to_tet'], tet_daily['order_count'], alpha=0.2, color='red')
plt.title('Xu hướng đơn hàng xung quanh kỳ nghỉ Tết Nguyên Đán (Trung bình 2012-2022)')
plt.xlabel('Số ngày trước/sau Tết')
plt.ylabel('Số lượng đơn hàng')
plt.grid(True, alpha=0.3)
plt.legend()
save_fig(os.path.join(FIG_DIR, '10_tet_seasonality.png'))

print("Lunar analysis complete.")
