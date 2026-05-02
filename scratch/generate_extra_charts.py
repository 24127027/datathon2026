import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Generating additional charts for report sections...")
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
FIG_DIR = r'c:\Users\mduy\source\repos\datathon2026\reports\business_analysis\figures'

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()

# 1. Order Status Distribution
# Data from report: Delivered (79.87%), Cancelled (9.19%), Returned (5.59%), Others (5.35%)
status_data = pd.DataFrame({
    'Status': ['Thành công', 'Đã hủy', 'Hoàn hàng', 'Khác'],
    'Percentage': [79.87, 9.19, 5.59, 5.35]
})
plt.figure(figsize=(8, 6))
plt.pie(status_data['Percentage'], labels=status_data['Status'], autopct='%1.1f%%', 
        colors=['#4E79A7', '#E15759', '#F28E2B', '#76B7B2'], startangle=140, explode=(0.05, 0, 0, 0))
plt.title('Phân bổ Trạng thái Đơn hàng')
save_fig(os.path.join(FIG_DIR, '10_order_status_pie.png'))

# 2. Payment Method Distribution
# Data from report: Credit Card (55.08%), Paypal (15.00%), COD (14.94%), Apple Pay (10.01%), Bank Transfer (4.97%)
payment_data = pd.DataFrame({
    'Method': ['Credit Card', 'Paypal', 'COD', 'Apple Pay', 'Bank Transfer'],
    'Percentage': [55.08, 15.00, 14.94, 10.01, 4.97]
})
plt.figure(figsize=(10, 5))
sns.barplot(data=payment_data, x='Method', y='Percentage', palette='Blues_r')
plt.title('Phân bổ Phương thức Thanh toán')
plt.ylabel('Tỷ lệ (%)')
plt.xlabel('Phương thức')
save_fig(os.path.join(FIG_DIR, '10_payment_method_bar.png'))

# 3. Top 5 Products by Revenue
# Data from report: SaigonFlex UM-92 (380.47), HanoiStreet UM-10 (327.51), SaigonFlex UM-43 (325.01), SaigonFlex UM-80 (256.55), SaigonFlex UM-96 (241.00)
top_prods = pd.DataFrame({
    'Product': ['SaigonFlex UM-92', 'HanoiStreet UM-10', 'SaigonFlex UM-43', 'SaigonFlex UM-80', 'SaigonFlex UM-96'],
    'Revenue': [380.47, 327.51, 325.01, 256.55, 241.00]
})
plt.figure(figsize=(10, 6))
sns.barplot(data=top_prods, y='Product', x='Revenue', palette='viridis')
plt.title('Top 5 Sản phẩm đóng góp Doanh thu cao nhất')
plt.xlabel('Doanh thu (Triệu VND)')
plt.ylabel('Sản phẩm')
save_fig(os.path.join(FIG_DIR, '07_top_5_products_revenue.png'))

print("Additional charts generated.")
