import numpy as np
import matplotlib.pyplot as plt

try:
    K = int(input('Введите число K: '))
    N = int(input('Введите число N большее 3: '))
except ValueError:
    print('Ошибка неправильного ввода данных.\n'
          'Повторите попытку, необходимо ввести целочисленные данные.')
    exit()

if N < 4:
    print('Введено неверное количество строк (столбцов) квадратной матрицы.\n')
    exit()

# Задаем половину размера матрицы
N = N // 2

# Генерация подматриц
b = np.random.randint(-10, 10, (N, N))
print('Подматрица b: \n', b, '\n')

c = np.random.randint(-10, 10, (N, N))
print('Подматрица c: \n', c, '\n')

d = np.random.randint(-10, 10, (N, N))
print('Подматрица d: \n', d, '\n')

e = np.random.randint(-10, 10, (N, N))
print('Подматрица e: \n', e, '\n')

A = np.vstack([np.hstack([b, c]), np.hstack([d, e])])
print('Матрица A: \n')
print(A, '\n')

# Копирование
F = A.copy()

count = 0
prodNumbers = 1
for i in e:
    for j in i[1::2]:
        if j > K:
            count += 1

for i in e[::2]:
    for j in i:
        prodNumbers *= j

if count > prodNumbers:
    F[:, N:] = F[::-1, N:]
else:
    F[:N, :] = np.hstack((F[:N, N:], F[:N, :N]))

print('Матрица F: \n')
print(F, '\n')

print('Матрица A: \n')
print(A, '\n')

# Диагональные элементы матрицы - главная диагональ матрицы
if np.linalg.det(A) > np.diagonal(F).sum():
    A_trans = np.transpose(A)
    F_inv = np.linalg.inv(F)
    result = A * A_trans - K * F_inv
else:
    A_inv = np.linalg.inv(A)
    F_trans = np.transpose(F)
    G = np.tril(A)
    result = (A_inv + G - F_trans) * K

print('Результат вычисления выражения: \n')
print(result)

# Растровое представление
figFirst = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(F[:N, :N], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 2)
plt.imshow(F[:N, N:], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 3)
plt.imshow(F[N:, :N], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 4)
plt.imshow(F[N:, N:], cmap='rainbow', interpolation='bilinear')
plt.show()

# Представление двумерных графиков
figSecond = plt.figure()
plt.subplot(2, 2, 1)
plt.plot(F[:N, :N])
plt.subplot(2, 2, 2)
plt.plot(F[:N, N:])
plt.subplot(2, 2, 3)
plt.plot(F[N:, :N])
plt.subplot(2, 2, 4)
plt.plot(F[N:, N:])
plt.show()
