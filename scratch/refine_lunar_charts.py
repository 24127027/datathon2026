import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

print("Refining Lunar and Tet charts for clarity...")
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
FIG_DIR = r'c:\Users\mduy\source\repos\datathon2026\reports\business_analysis\figures'

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()

# 1. Load Order Data
orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv', parse_dates=['order_date'])

# 2. Lunar Day Approximation
ref_jd = 2455950.0
lunar_month = 29.53059
orders['jd'] = orders['order_date'].apply(lambda x: x.to_julian_date())
orders['lunar_day'] = ((orders['jd'] - ref_jd) % lunar_month).round().astype(int) + 1
orders.loc[orders['lunar_day'] > 30, 'lunar_day'] = 1

# 3. Lunar Day Distribution - Deviation from Average
lunar_counts = orders['lunar_day'].value_counts().sort_index()
avg_orders = lunar_counts.mean()
lunar_dev = ((lunar_counts - avg_orders) / avg_orders * 100).reset_index()
lunar_dev.columns = ['lunar_day', 'deviation']

plt.figure(figsize=(14, 6))
# Highlight 1 and 15
colors = ['#E15759' if x in [1, 15] else '#4E79A7' for x in lunar_dev['lunar_day']]
ax = sns.barplot(data=lunar_dev, x='lunar_day', y='deviation', palette=colors)

# Add labels
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                    textcoords='offset points')

plt.axhline(0, color='black', linewidth=1)
plt.title('Mức độ biến động đơn hàng theo Ngày Âm lịch (So với trung bình tháng)')
plt.ylabel('Tỷ lệ chênh lệch (%)')
plt.xlabel('Ngày Âm lịch')
plt.ylim(lunar_dev['deviation'].min() - 5, lunar_dev['deviation'].max() + 8)
save_fig(os.path.join(FIG_DIR, '10_lunar_day_distribution.png'))

# 4. Tet Seasonality - Normalized
tet_dates = {
    2012: '2012-01-23', 2013: '2013-02-10', 2014: '2014-01-31', 
    2015: '2015-02-19', 2016: '2016-02-08', 2017: '2017-01-28',
    2018: '2018-02-16', 2019: '2019-02-05', 2020: '2020-01-25',
    2021: '2021-02-12', 2022: '2022-02-01'
}

tet_results = []
for year, date_str in tet_dates.items():
    tet_dt = pd.to_datetime(date_str)
    mask = (orders['order_date'] >= tet_dt - pd.Timedelta(days=30)) & (orders['order_date'] <= tet_dt + pd.Timedelta(days=15))
    period_orders = orders[mask].copy()
    period_orders['days_to_tet'] = (period_orders['order_date'] - tet_dt).dt.days
    tet_results.append(period_orders)

tet_df = pd.concat(tet_results)
tet_daily = tet_df.groupby('days_to_tet').size().reset_index(name='order_count')
avg_tet_period = tet_daily['order_count'].mean()
tet_daily['deviation'] = (tet_daily['order_count'] - avg_tet_period) / avg_tet_period * 100

plt.figure(figsize=(14, 6))
plt.plot(tet_daily['days_to_tet'], tet_daily['deviation'], color='#D62728', marker='o', linewidth=2.5, markersize=8)
plt.fill_between(tet_daily['days_to_tet'], tet_daily['deviation'], -50, alpha=0.1, color='red')

# Annotations
plt.axvline(x=0, color='gold', linestyle='--', linewidth=2, label='Mùng 1 Tết')
plt.axvline(x=-7, color='orange', linestyle=':', label='23 tháng Chạp (Ông Táo)')
plt.annotate('Đỉnh mua sắm Tết', xy=(-5, tet_daily.loc[tet_daily['days_to_tet']==-5, 'deviation'].values[0]),
             xytext=(-25, 40), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Chạm đáy', xy=(1, tet_daily.loc[tet_daily['days_to_tet']==1, 'deviation'].values[0]),
             xytext=(5, -60), arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('Hành vi mua sắm xung quanh kỳ nghỉ Tết Nguyên Đán (Độ lệch %)')
plt.ylabel('Thay đổi so với trung bình (%)')
plt.xlabel('Số ngày trước/sau Tết')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
save_fig(os.path.join(FIG_DIR, '10_tet_seasonality.png'))

print("Charts refined with clear labels and percentage deviations.")
