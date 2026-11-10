"""BUỔI 1 - Huấn luyện và đọc cây quyết định trên bộ dữ liệu Iris."""
# Bước 1. Nạp thư viện
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Bước 2. Đọc dữ liệu
dataset = pd.read_csv('iris.data.csv', header=None)
dataset.columns = ['sepal length in cm', 'sepal width in cm',
                   'petal length in cm', 'petal width in cm', 'species']
dataset = dataset.dropna()
print('Kích thước:', dataset.shape)
print(dataset.head())

# Bước 3. Mã hóa nhãn: đổi tên loài thành 0, 1, 2
factor = pd.factorize(dataset['species'])
dataset['species'] = factor[0]
definitions = factor[1]
print('Các nhãn:', definitions)

# Bước 4. Tách đặc trưng X và nhãn y
feature_names = list(dataset.columns[0:4])
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# Bước 5. Chia tập huấn luyện (75%) và tập kiểm tra (25%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21)
print(f'Train: {len(X_train)} mẫu | Test: {len(X_test)} mẫu')

# Bước 6. Huấn luyện cây quyết định
classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
classifier.fit(X_train, y_train)
print('Độ sâu của cây:', classifier.get_depth())
print('Số nút lá:', classifier.get_n_leaves())

# Bước 7. Đánh giá trên tập kiểm tra
y_pred = classifier.predict(X_test)
ten_loai = dict(zip(range(3), definitions))
y_test_name = np.vectorize(ten_loai.get)(y_test)
y_pred_name = np.vectorize(ten_loai.get)(y_pred)

print('\nMa trận nhầm lẫn:')
print(pd.crosstab(y_test_name, y_pred_name,
                  rownames=['Thực tế'], colnames=['Dự đoán']))
print('Accuracy:', round(accuracy_score(y_test, y_pred), 3))
print('\nBáo cáo phân loại:')
print(classification_report(y_test_name, y_pred_name))

# Bước 8. Xem cấu trúc cây: in luật và vẽ hình
print('\nCấu trúc cây dạng luật:')
print(export_text(classifier, feature_names=feature_names))

plt.figure(figsize=(14, 8))
plot_tree(classifier, feature_names=feature_names,
          class_names=list(definitions), filled=True, rounded=True,
          fontsize=10)
plt.savefig('cay_quyet_dinh.png', dpi=150, bbox_inches='tight')
print('Đã lưu hình cây vào cay_quyet_dinh.png')

# Bước 9. Kiểm chứng entropy và độ lợi thông tin ở nút gốc
def entropy(nhan):
    p = np.bincount(nhan) / len(nhan)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) + 0.0

feat = classifier.tree_.feature[0]
thr = classifier.tree_.threshold[0]
trai = y_train[X_train[:, feat] <= thr]
phai = y_train[X_train[:, feat] > thr]

H_goc = entropy(y_train)
H_sau = (len(trai) * entropy(trai) + len(phai) * entropy(phai)) / len(y_train)

print(f'\nNút gốc tách theo: {feature_names[feat]} <= {thr:.2f}')
print(f'Số mẫu mỗi lớp ở gốc: {np.bincount(y_train).tolist()}')
print(f'Entropy nút gốc (tự tính) : {H_goc:.4f}')
print(f'Entropy nút gốc (sklearn): {classifier.tree_.impurity[0]:.4f}')
print(f'Nhánh trái: {len(trai)} mẫu, entropy = {entropy(trai):.4f}')
print(f'Nhánh phải: {len(phai)} mẫu, entropy = {entropy(phai):.4f}')
print(f'Độ lợi thông tin = {H_goc:.4f} - {H_sau:.4f} = {H_goc - H_sau:.4f}')

# Bước 10. Độ quan trọng đặc trưng và lưu mô hình
print('\nĐộ quan trọng đặc trưng:')
for ten, diem in zip(feature_names, classifier.feature_importances_):
    print(f'{ten}: {diem:.3f}')

joblib.dump(classifier, 'model.pkl')
joblib.dump(list(definitions), 'labels.pkl')
print('Đã lưu model.pkl, labels.pkl')

# Kiểm tra nhanh
model = joblib.load('model.pkl')
labels = joblib.load('labels.pkl')
sample_pred = labels[model.predict([[5.1, 3.5, 1.4, 0.2]])[0]]
print('Dự đoán mẫu thử [5.1, 3.5, 1.4, 0.2]:', sample_pred)