import pandas as pd

orders = pd.read_csv('data/datathon-2026-round-1/transaction/orders.csv')
orders['order_date'] = pd.to_datetime(orders['order_date'])

def get_median_gap(df):
    df = df.copy()
    df = df.sort_values(['customer_id', 'order_date'])
    df['gap'] = df.groupby('customer_id')['order_date'].diff().dt.days
    return df['gap'].median()

print(f"All orders: {get_median_gap(orders)}")
print(f"Delivered only: {get_median_gap(orders[orders['order_status'] == 'delivered'])}")
print(f"Excluding cancelled: {get_median_gap(orders[orders['order_status'] != 'cancelled'])}")
print(f"Excluding cancelled and returned: {get_median_gap(orders[~orders['order_status'].isin(['cancelled', 'returned'])])}")
