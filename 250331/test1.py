import numpy as np


#Q1:使用 NumPy 建立一個包含數字 1 到 10 的一維陣列。
a = np.arange(1,11,1)
print(a)
print("\n")


#Q2:建立一個 3x3 的全零矩陣。
b = np.zeros((3,3))
print(b)
print("\n")


#Q3:建立一個 4x4 的單位矩陣（對角線為 1，其餘為 0）。
c =np.zeros((4,4))
for i in range(0,4):
    c[i][i] = 1
print(c)
print("\n")


#Q4:使用 np.random.rand() 建立一個 5x5 的隨機陣列，並找出其中的最大值與最小值。

d = np.random.rand(5,5)
max = 0
min = 1
for i in range(0,5):
    for j in range(0,5):
        if d[i][j]> max:
            max = d[i][j]
        elif d[i][j]< min:
            min = d[i][j]
print(d)
print("\n")
print(f"max = {round(max,4)}, min = {round(min,4)}\n")


#Q5:使用 np.arange() 建立 0 到 20 之間間隔為 2 的陣列。
e = np.arange(0,21,2)
print(e)
print("\n")