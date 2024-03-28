# in_time = [7, 11, 16]
# out_time = [10, 15, 18]

# clock = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# time = []
# # Khoi tao lich thoi gian da co order(1) và ko có order(0)
# def init(clock, in_time, out_time):
#     for i in clock: 
#         time.append(0)
#     for i in range(0, len(in_time)):
#         for j in clock:
#             if (j > in_time[i]) and (j <= out_time[i]):
#                 time[j] = 1

# init(clock, in_time, out_time)
# print(time)

# # Check xem don hang them vao co ai dat chua
# def check_time(a, b, time):
#     for i in range(0, len(time)):
#         if a < clock[i] <= b:
#             if time[i] == 1:
#                 return False
#     return True

# print(check_time(10,11,time))


# a= "12:15"
# h,p = a.split(":")
# print((int(h)*100+int(p))/100)

#####################################################################################################
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import Counter

dataAll = ['Quang noodles', 'Fish ball vermicelli', 'Nem lui', 'Hue beef noodle soup', 'Goi spring rolls', 'Fish sauce hot pot', 'Pot pot rice', 'Bread', 'Mussels with rice paper', 'Rice paper rolls', 'Beef noodles']

dataOrder = {
    'product_name': ['Quang noodles', 'Bun with fish balls', 'Nem lui', 'Quang noodles', 'Bun with fish balls', 'Bun with beef noodles', 'Nem lui', 'Quang noodles', 'Bun with fish balls', 'Spring rolls'],
    'order_date': ['2023-11-01', '2023-11-01', '2023-11-02', '2023-11-02', '2023-11-03', '2023-11-03', '2023-11-04', '2023-11-04', '2023-11-05', '2023-11-05']
}

df = pd.DataFrame(dataOrder)
df['order_date'] = pd.to_datetime(df['order_date'])

# Sắp xếp DataFrame theo 'order_date' giảm dần
df = df.sort_values(by='order_date', ascending=False)

# Lấy 5 sản phẩm được đặt nhiều nhất
top_products = [item[0] for item in Counter(df['product_name']).most_common(5)]
print(top_products)

check = 'Beef noodle'
all_data = dataAll + [check]

# Tạo ma trận TF-IDF cho các sản phẩm loại bỏ từ dừng vietnamese
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(all_data)

# Sử dụng linear_kernel để tính toán độ tương đồng giữa "Mì Quảng" và tất cả các sản phẩm khác
cosine_similarities = linear_kernel(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
print(cosine_similarities)

# Xây dựng danh sách các cặp (tên món ăn, độ tương đồng cosine)
similarities_with_names = list(zip(dataAll, cosine_similarities))

# In ra danh sách các món ăn và độ tương đồng cosine tương ứng
for name, similarity in similarities_with_names:
    print(f"Độ tương đồng giữa 'mì quảng' và '{name}': {similarity}")