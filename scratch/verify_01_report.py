import pandas as pd
import numpy as np
from src.data.loader import load_orders, load_order_items, load_web_traffic, load_inventory, load_sales

def main():
    print("Loading data...")
    orders = load_orders()
    order_items = load_order_items()
    web_traffic = load_web_traffic()
    sales = load_sales()
    products = pd.read_csv('data/datathon-2026-round-1/master/products.csv')

    print("Data loaded. Computing Section 1: Yearly Revenue...")
    
    # order_items from loader already has product joined
    merged = pd.merge(orders, order_items, on='order_id', how='inner')
    
    # Calculate revenue, COGS, Profit, Margin
    merged['revenue'] = merged['quantity'] * merged['unit_price']
    merged['total_cogs_item'] = merged['quantity'] * merged['cogs']
    merged['profit'] = merged['revenue'] - merged['total_cogs_item']
    
    merged['date'] = pd.to_datetime(merged['date'])
    merged['year'] = merged['date'].dt.year
    
    yearly_stats = merged.groupby('year').agg(
        total_revenue=('revenue', 'sum'),
        total_cogs=('total_cogs_item', 'sum'),
        total_profit=('profit', 'sum')
    )
    yearly_stats['margin'] = yearly_stats['total_profit'] / yearly_stats['total_revenue']
    
    web_traffic['date'] = pd.to_datetime(web_traffic['date'])
    web_traffic['year'] = web_traffic['date'].dt.year
    yearly_traffic = web_traffic.groupby('year').agg(
        total_sessions=('sessions', 'sum')
    )
    
    yearly_orders = merged.groupby(['year', 'order_id']).size().reset_index().groupby('year').size()
    
    yearly = yearly_stats.join(yearly_traffic)
    yearly['orders'] = yearly_orders
    yearly['conversion'] = yearly['orders'] / yearly['total_sessions']
    yearly['aov'] = yearly['total_revenue'] / yearly['orders']
    yearly['revenue_per_session'] = yearly['total_revenue'] / yearly['total_sessions']
    
    print("\n=== Section 1: Yearly Stats ===")
    print(yearly[['total_revenue', 'total_cogs', 'total_profit', 'margin', 'conversion', 'aov', 'revenue_per_session']].to_string())
    
    print("\n=== Section 2: 2018 vs 2019 ===")
    if 2018 in yearly.index and 2019 in yearly.index:
        for col in ['total_revenue', 'total_sessions', 'conversion', 'aov', 'revenue_per_session']:
            val18 = yearly.loc[2018, col]
            val19 = yearly.loc[2019, col]
            pct_change = (val19 - val18) / val18 * 100
            print(f"{col}: 2018={val18:.2f}, 2019={val19:.2f}, Change={pct_change:.2f}%")
            
    print("\n=== Section 3: August (Odd vs Even) ===")
    august_data = merged[merged['date'].dt.month == 8].copy()
    august_data['is_even'] = august_data['year'] % 2 == 0
    
    august_stats = august_data.groupby(['year', 'is_even']).agg(
        revenue=('revenue', 'sum'),
        cogs=('total_cogs_item', 'sum')
    ).reset_index()
    
    # Calculate daily average for each year, then average across years? Or total / days?
    # Let's do daily average by dividing total revenue in August for that group by (number of years in group * 31)
    # Actually, a simpler grouping by is_even might not match exactly how daily average is computed. Let's look at total days.
    aug_even_years = august_data[august_data['is_even']]['year'].nunique()
    aug_odd_years = august_data[~august_data['is_even']]['year'].nunique()
    
    aug_daily = august_data.groupby('is_even').agg(
        revenue=('revenue', 'sum'),
        cogs=('total_cogs_item', 'sum'),
        profit=('profit', 'sum')
    )
    aug_daily.loc[True, 'daily_revenue'] = aug_daily.loc[True, 'revenue'] / (aug_even_years * 31)
    aug_daily.loc[True, 'daily_cogs'] = aug_daily.loc[True, 'cogs'] / (aug_even_years * 31)
    aug_daily.loc[False, 'daily_revenue'] = aug_daily.loc[False, 'revenue'] / (aug_odd_years * 31)
    aug_daily.loc[False, 'daily_cogs'] = aug_daily.loc[False, 'cogs'] / (aug_odd_years * 31)
    aug_daily['margin'] = aug_daily['profit'] / aug_daily['revenue']
    
    print(aug_daily[['daily_revenue', 'daily_cogs', 'margin']])
    
    print("\n=== Section 4: Quarters ===")
    merged['quarter'] = merged['date'].dt.quarter
    q_stats = merged.groupby('quarter').agg(
        total_revenue=('revenue', 'sum'),
        total_cogs=('total_cogs_item', 'sum'),
        total_profit=('profit', 'sum')
    )
    q_stats['margin'] = q_stats['total_profit'] / q_stats['total_revenue']
    print(q_stats)

    print("\n=== Data Anomalies ===")
    print("Missing values in merged data:", merged.isnull().sum())
    print("Negative quantities?", (merged['quantity'] < 0).sum())
    print("Negative prices?", (merged['unit_price'] < 0).sum())
    
if __name__ == '__main__':
    main()