from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. داده‌های نمونه (محصولات و توضیحات)
products = pd.DataFrame({
    'product_id': [1, 2, 3, 4, 5],
    'product_name': ["Laptop", "Smartphone", "Headphones", "Camera", "Smartwatch"],
    'description': [
        "A high-performance laptop with 16GB RAM and 512GB SSD.",
        "A smartphone with a great camera and long-lasting battery.",
        "Noise-cancelling headphones with excellent sound quality.",
        "A DSLR camera with 24MP resolution and 4K video recording.",
        "A smartwatch with fitness tracking and heart rate monitoring."
    ]
})

# 2. بردارسازی توضیحات با استفاده از TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(products['description'])

# 3. محاسبه شباهت کسینوسی
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# نمایش ماتریس شباهت
print("Similarity Matrix:")
print(similarity_matrix)

# 4. تعریف تابع توصیه‌گر
def recommend_products(product_id, similarity_matrix, products, top_n=3):
    product_idx = products[products['product_id'] == product_id].index[0]
    similarity_scores = list(enumerate(similarity_matrix[product_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_products = [products.iloc[i[0]]['product_name'] for i in similarity_scores[1:top_n+1]]
    return top_products

# 5. آزمایش سیستم توصیه‌گر
product_id_to_recommend = 1
recommended = recommend_products(product_id_to_recommend, similarity_matrix, products)
print(f"Products recommended for Product ID {product_id_to_recommend}: {recommended}")

# --- برنامه 7: پیش‌بینی مصرف برق ---
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. بارگذاری داده‌ها
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip"
data = pd.read_csv(data_url, sep=';', parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True, low_memory=False, na_values=['?'], index_col='datetime')
data.fillna(method='ffill', inplace=True)

# 2. انتخاب ویژگی‌ها و متغیر هدف
power_data = data['Global_active_power'].astype('float32')
power_data = power_data.resample('D').mean()

# 3. آماده‌سازی داده‌ها
scaler = MinMaxScaler()
power_scaled = scaler.fit_transform(power_data.values.reshape(-1, 1))

X, y = [], []
for i in range(30, len(power_scaled)):
    X.append(power_scaled[i-30:i, 0])
    y.append(power_scaled[i, 0])
X, y = np.array(X), np.array(y)

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. ساخت مدل
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# 5. ارزیابی مدل
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

# 6. مصورسازی نتایج
plt.figure(figsize=(14, 8))
plt.plot(y_test, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='orange')
plt.title("Actual vs Predicted Power Consumption")
plt.xlabel("Time")
plt.ylabel("Power Consumption (scaled)")
plt.legend()
plt.show()
