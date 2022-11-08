import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as st
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# Количество точек соседей
k = 7


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def knn(x_train, y, x_input, k):
    op_labels = []

    for item in test_set_x:
        # Массив для хранения расстояний
        point_dist = []

        # Просматриваем все обучающие данные
        for j in range(len(train_set_x)):
            distances = euclidean_distance(np.array(train_set_x[j, :]), item)
            point_dist.append(distances)

        # Преобразование в массив для удобства работы
        point_dist = np.array(point_dist)

        # Сортировка массива с сохранением индекса
        # Сохранение первых K точек данных
        dist = np.argsort(point_dist)[:k]

        # Метки K точек данных в header [Protein, Vegetable, Fruit, Berry]
        labels = y[dist]
        # print('Labels values: ')
        # print(labels)

        # Мода на частоту
        # label = stats.mode(labels, keepdims=True)
        # label = label[0][0]
        label = st.mode(labels)

        # print('Moda labels: ')
        # print(label)
        # print()

        op_labels.append(label)

    return op_labels


def knn_with_scikit_learn(X, y):
    print('\nKNN with scikit-learn')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # Предсказывание
    predictions = model.predict(X_test)

    return X_train, X_test, y_train, y_test, predictions


####################################################################
# Данные и переменные
dataset = pd.read_csv('data.csv')

f1 = dataset['Sweetness'].values
f2 = dataset['Crunch'].values

X = np.array(list(zip(f1, f2)), dtype=np.float64)
y = dataset['Product category'].values

# [Sweetness Crunch]
# Тренировочные данные - 70% и тестовые данные - 30%
train_set_x = X[:70]
test_set_x = X[70:]

train_set_y = y[:70]
test_set_y = y[70:]

# distances = np.zeros((len(test_set_x), len(train_set_x)))
# dist_sort = []
# for i in range(len(test_set_x)):
#     for j in range(len(train_set_x)):
#         distances[i][j] = np.linalg.norm(test_set_x[i] - train_set_x[j])

####################################################################
print('\nThe first experiment\n')
# # Обучение методом KNN БЕЗ ИСПОЛЬЗОВАНИЯ библиотеки scikit-learn
y_predictions = knn(train_set_x, train_set_y, test_set_x, 7)

# Вывод статистики качества прогнозирования
print('Statistics knn-method without libraries')
print(classification_report(test_set_y, y_predictions))
print(confusion_matrix(test_set_y, y_predictions))

####################################################################
# Обучение методом KNN С ИСПОЛЬЗОВАНИЕМ библиотеки scikit-learn
X_train, X_test, y_train, y_test, y_pred = knn_with_scikit_learn(X, y)

# Вывод статистики качества прогнозирования
print('Statistics knn-method with scikit-learn')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

####################################################################
# График разброса
y_pred_color = [elem.replace("Fruit", "red", 1)
                .replace("Vegetable", "blue", 1)
                .replace("Protein", "black", 1) for elem in y_predictions]

y_pred_color_with_scikit_learn = [elem.replace("Fruit", "red", 1)
                                  .replace("Vegetable", "blue", 1)
                                  .replace("Protein", "black", 1) for elem in y_pred]

f, ax = plt.subplots(2, 1, figsize=(8, 8))

ax[0].scatter(test_set_x[:, 0], test_set_x[:, 1], c=y_pred_color, s=50)
ax[0].set_title('Test data without libraries')  # заголовок для Axes

ax[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred_color_with_scikit_learn, s=50)
ax[1].set_title('Test data with scikit-learn')  # заголовок для Axes

plt.show()

####################################################################
# Новые данные
# Добавлен дополнительный класс Berry
dataset_new = pd.read_csv('data_new.csv')

f1 = dataset_new['Sweetness'].values
f2 = dataset_new['Crunch'].values

X_new = np.array(list(zip(f1, f2)), dtype=np.float64)
y_new = dataset_new['Product category'].values

# [Sweetness Crunch]
# Тренировочные данные - 70% и тестовые данные - 30%
train_set_x = X_new[:84]
test_set_x = X_new[84:]

train_set_y = y_new[:84]
test_set_y = y_new[84:]

####################################################################
print('\nThe second experiment\n')
# Обучение методом KNN БЕЗ ИСПОЛЬЗОВАНИЯ библиотеки scikit-learn
y_predictions_new = knn(train_set_x, train_set_y, test_set_x, k)

# Вывод статистики качества прогнозирования
print('Statistics knn-method without libraries')
print(classification_report(test_set_y, y_predictions_new))
print(confusion_matrix(test_set_y, y_predictions_new))

####################################################################
# Обучение методом KNN С ИСПОЛЬЗОВАНИЕМ библиотеки scikit-learn
X_train_new, X_test_new, y_train_new, y_test_new, y_pred_new = knn_with_scikit_learn(X_new, y_new)

# Вывод статистики качества прогнозирования
print('Statistics knn-method with scikit-learn')
print(classification_report(y_test_new, y_pred_new))
print(confusion_matrix(y_test_new, y_pred_new))

####################################################################
# График разброса
y_pred_color_new = [elem.replace("Fruit", "red", 1)
                    .replace("Vegetable", "blue", 1)
                    .replace("Protein", "black", 1)
                    .replace("Berry", "green", 1) for elem in y_predictions_new]

y_pred_color_with_scikit_learn_new = [elem.replace("Fruit", "red", 1)
                                      .replace("Vegetable", "blue", 1)
                                      .replace("Protein", "black", 1)
                                      .replace("Berry", "green", 1) for elem in y_pred_new]

fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].scatter(test_set_x[:, 0], test_set_x[:, 1], c=y_pred_color_new, s=100)
ax1[0].set_title('Test data without libraries')  # заголовок для Axes

ax1[1].scatter(X_test_new[:, 0], X_test_new[:, 1], c=y_pred_color_with_scikit_learn_new, s=100)
ax1[1].set_title('Test data with scikit-learn')  # заголовок для Axes

plt.show()
