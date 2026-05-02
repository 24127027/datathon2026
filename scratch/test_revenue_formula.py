import pandas as pd
import numpy as np

oi = pd.read_csv('data/datathon-2026-round-1/transaction/order_items.csv', low_memory=False)
payments = pd.read_csv('data/datathon-2026-round-1/transaction/payments.csv', low_memory=False)

rev_A = oi.groupby('order_id').apply(
    lambda g: (g['unit_price'] * g['quantity'] - g['discount_amount']).sum(),
    include_groups=False
).reset_index()
rev_A.columns = ['order_id', 'rev_A']

rev_B = oi.groupby('order_id').apply(
    lambda g: (g['unit_price'] * g['quantity']).sum(),
    include_groups=False
).reset_index()
rev_B.columns = ['order_id', 'rev_B']

pay = payments.groupby('order_id')['payment_value'].sum().reset_index()
pay.columns = ['order_id', 'paid']

comp = rev_A.merge(rev_B, on='order_id').merge(pay, on='order_id', how='inner')
comp['err_A'] = (comp['rev_A'] - comp['paid']).abs() / comp['paid'] * 100
comp['err_B'] = (comp['rev_B'] - comp['paid']).abs() / comp['paid'] * 100

print(f'Orders matched: {len(comp):,}')
print()
print('Formula A (unit_price*qty - discount_amount) vs payment_value:')
print(comp['err_A'].describe())
pct_A = (comp['err_A'] < 1).mean() * 100
print(f'Within 1%: {pct_A:.1f}%')
print()
print('Formula B (unit_price*qty only) vs payment_value:')
print(comp['err_B'].describe())
pct_B = (comp['err_B'] < 1).mean() * 100
print(f'Within 1%: {pct_B:.1f}%')
print()
print('Sample:')
print(comp[['order_id', 'rev_A', 'rev_B', 'paid']].head(10).to_string())
