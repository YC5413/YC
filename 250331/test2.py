import numpy as np


#Q1:建立一個 6x6 的矩陣，並用 1 圍住矩陣，內部全為 0。
a = np.zeros((6,6))
a[0, :] = 1
a[-1, :] = 1
a[:, 0] = 1
a[:, -1] = 1
print(a)
print("\n")


#Q2:建立一個 3x3 矩陣，計算該矩陣的行列式和反矩陣。
import numpy as np
matrix = np.array([[3, 2, 1],
                   [1, 1, 1],
                   [2, 3, 4]])
# 計算行列式
determinant = np.linalg.det(matrix)
print("行列式:", determinant)
# 判斷是否為奇異矩陣
if not np.isclose(determinant, 0):
    inverse_matrix = np.linalg.inv(matrix)
    print("反矩陣:")
    print(inverse_matrix)
else:
    print("此矩陣不可逆，無法計算反矩陣。")



#Q3:生成一個隨機陣列，對其執行標準化 (即將數據轉換為均值為 0、標準差為 1 的形式)。
data = np.random.rand(5, 5)
print("原始數據：")
print(data)

# 計算均值和標準差
mean = np.mean(data)
std = np.std(data)

# 標準化
standardized_data = (data - mean) / std
print("\n標準化後：")
print(standardized_data)


#Q4:創建兩個矩陣，使用 NumPy 進行矩陣乘法，並計算它們的特徵值和特徵向量。
a = np.random.rand(3, 3)
b = np.random.rand(3, 3)
# 矩陣乘法
C = np.dot(a, b)
print("矩陣乘法結果：")
print(C)
# 計算特徵值和特徵向量
eigenvalues, eigenvectors = np.linalg.eig(C)
print("\n特徵值：", eigenvalues)
print("特徵向量：")
print(eigenvectors)


#Q5:將一個一維陣列重塑為 3x4 矩陣，並取出矩陣的第 2 列。
array = np.arange(1, 13)

# 重塑為 3x4 矩陣
matrix = array.reshape(3, 4)
print("重塑後的矩陣：")
print(matrix)

# 取出第 2 列（索引從 0 開始，因此索引 1 是第 2 列）
second_column = matrix[:, 1]
print("\n第 2 列：", second_column)

