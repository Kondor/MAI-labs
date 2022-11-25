# https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
#
# Wine Quality Dataset
#
# Input variables (based on physicochemical tests):
# 1 - fixed acidity
# 2 - volatile acidity
# 3 - citric acid
# 4 - residual sugar
# 5 - chlorides
# 6 - free sulfur dioxide
# 7 - total sulfur dioxide
# 8 - density
# 9 - pH
# 10 - sulphates
# 11 - alcohol
#
# Output variable (based on sensory data):
# 12 - quality (score between 0 and 10)


import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# Метод k-Nearest Neighbors
def knn(X_train_knn, X_test_knn, y_train_knn, y_test_knn):
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_knn, y_train_knn)
    y_pred_knn = knn_model.predict(X_test_knn)

    # Статистика
    print('Report showing the main classification metrics')
    print(classification_report(y_test_knn, y_pred_knn, zero_division=0))

    print('A matrix of inaccuracies for evaluating accuracy of the classification')
    print(confusion_matrix(y_test_knn, y_pred_knn))

    print('Score the X-train with Y-train is : ', knn_model.score(X_train_knn, y_train_knn))
    print('Score the X-test  with Y-test  is : ', knn_model.score(X_test_knn, y_test_knn))
    print('Evaluation of the Model K-nearest neighbors: accuracy score ', accuracy_score(y_test_knn, y_pred_knn))

    # Графическое отображение наиболее важных признаков
    #  6 - free sulfur dioxide
    #  7 - total sulfur dioxide
    # 11 - alcohol
    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    ax[0].scatter(X_test_knn[:, 5], X_test_knn[:, 6], c=y_pred_knn)
    ax[0].set_title('Dependence of the free sulfur dioxide on total sulfur dioxide')

    ax[1].scatter(X_test_knn[:, 5], X_test_knn[:, 10], c=y_pred_knn)
    ax[1].set_title('Dependence of the free sulfur dioxide on alcohol')

    ax[2].scatter(X_test_knn[:, 6], X_test_knn[:, 10], c=y_pred_knn)
    ax[2].set_title('Dependence of the total sulfur dioxide on alcohol')
    plt.show()

    # Графическое отображение тестовых значений
    target_var1 = pd.crosstab(index=y_test_knn, columns='% observations')
    plt.pie(target_var1['% observations'], labels=target_var1['% observations'].index, autopct='%.0f%%')
    plt.title('Array of test values')
    plt.show()

    # Графическое отображение предсказанных значений
    target_var2 = pd.crosstab(index=y_pred_knn, columns='% observations')
    plt.pie(target_var2['% observations'], labels=target_var2['% observations'].index, autopct='%.0f%%')
    plt.title('Array of predicted values')
    plt.show()

    # Совпадение тестовых данных и предсказания
    plt.plot(y_test_knn, label='Test')
    plt.plot(y_pred_knn, label='Predication')
    plt.legend()
    plt.show()

    quality_interval = max(y_test_knn) - min(y_test_knn)
    plt.title('Quality from {} to {}'.format(min(y_test_knn), max(y_test_knn)))
    plt.hist(y_test_knn, bins=quality_interval)
    plt.show()


# Метод Support Vector Classification
def svc(X_train_svc, X_test_svc, y_train_svc, y_test_svc):
    svc_model = SVC(kernel='poly', degree=10)
    svc_model.fit(X_train_svc, y_train_svc)
    y_pred_svc = svc_model.predict(X_test_svc)

    # Статистика
    print('Report showing the main classification metrics')
    print(classification_report(y_test_svc, y_pred_svc, zero_division=0))

    print('A matrix of inaccuracies for evaluating accuracy of the classification')
    print(confusion_matrix(y_test_svc, y_pred_svc))

    print('Score the X-train with Y-train is : ', svc_model.score(X_train_svc, y_train_svc))
    print('Score the X-test  with Y-test  is : ', svc_model.score(X_test_svc, y_test_svc))
    print('Evaluation of the Model Support Vector Classification: accuracy score ', accuracy_score(y_test_svc, y_pred_svc))

    # Графическое отображение наиболее важных признаков
    #  6 - free sulfur dioxide
    #  7 - total sulfur dioxide
    # 11 - alcohol
    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    ax[0].scatter(X_test_svc[:, 5], X_test_svc[:, 6], c=y_pred_svc)
    ax[0].set_title('Dependence of the free sulfur dioxide on total sulfur dioxide')

    ax[1].scatter(X_test_svc[:, 5], X_test_svc[:, 10], c=y_pred_svc)
    ax[1].set_title('Dependence of the free sulfur dioxide on alcohol')

    ax[2].scatter(X_test_svc[:, 6], X_test_svc[:, 10], c=y_pred_svc)
    ax[2].set_title('Dependence of the total sulfur dioxide on alcohol')
    plt.show()

    # Графическое отображение тестовых значений
    target_var1 = pd.crosstab(index=y_test_svc, columns='% observations')
    plt.pie(target_var1['% observations'], labels=target_var1['% observations'].index, autopct='%.0f%%')
    plt.title('Array of test values')
    plt.show()

    # Графическое отображение предсказанных значений
    target_var2 = pd.crosstab(index=y_pred_svc, columns='% observations')
    plt.pie(target_var2['% observations'], labels=target_var2['% observations'].index, autopct='%.0f%%')
    plt.title('Array of predicted values')
    plt.show()

    # Совпадение тестовых данных и предсказания
    plt.plot(y_test_svc, label='Test')
    plt.plot(y_pred_svc, label='Predication')
    plt.legend()
    plt.show()

    quality_interval = max(y_test_svc) - min(y_test_svc)
    plt.title('Quality from {} to {}'.format(min(y_test_svc), max(y_test_svc)))
    plt.hist(y_test_svc, bins=quality_interval)
    plt.show()


# Метод Decision Tree Classifier
def dtc(X_train_t, X_test_t, y_train_t, y_test_t):
    Tree_model = DecisionTreeClassifier(max_depth=12)
    Tree_model.fit(X_train_t, y_train_t)
    y_pred_t = Tree_model.predict(X_test_t)

    # Статистика
    print('Report showing the main classification metrics')
    print(classification_report(y_test_t, y_pred_t, zero_division=0))

    print('A matrix of inaccuracies for evaluating accuracy of the classification')
    print(confusion_matrix(y_test_t, y_pred_t))

    print('Score the X-train with Y-train is : ', Tree_model.score(X_train_t, y_train_t))
    print('Score the X-test  with Y-test  is : ', Tree_model.score(X_test_t, y_test_t))
    print('Evaluation of the Model Decision Tree Classifier: accuracy score ', accuracy_score(y_test_t, y_pred_t))

    # Графическое отображение наиболее важных признаков
    #  6 - free sulfur dioxide
    #  7 - total sulfur dioxide
    # 11 - alcohol
    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    ax[0].scatter(X_test_t[:, 5], X_test_t[:, 6], c=y_pred_t)
    ax[0].set_title('Dependence of the free sulfur dioxide on total sulfur dioxide')

    ax[1].scatter(X_test_t[:, 5], X_test_t[:, 10], c=y_pred_t)
    ax[1].set_title('Dependence of the free sulfur dioxide on alcohol')

    ax[2].scatter(X_test_t[:, 6], X_test_t[:, 10], c=y_pred_t)
    ax[2].set_title('Dependence of the total sulfur dioxide on alcohol')
    plt.show()

    # Графическое отображение тестовых значений
    target_var1 = pd.crosstab(index=y_test_t, columns='% observations')
    plt.pie(target_var1['% observations'], labels=target_var1['% observations'].index, autopct='%.0f%%')
    plt.title('Array of test values')
    plt.show()

    # Графическое отображение предсказанных значений
    target_var2 = pd.crosstab(index=y_pred_t, columns='% observations')
    plt.pie(target_var2['% observations'], labels=target_var2['% observations'].index, autopct='%.0f%%')
    plt.title('Array of predicted values')
    plt.show()

    # Совпадение тестовых данных и предсказания
    plt.plot(y_test_t, label='Test')
    plt.plot(y_pred_t, label='Predication')
    plt.legend()
    plt.show()

    quality_interval = max(y_test_t) - min(y_test_t)
    plt.title('Quality from {} to {}'.format(min(y_test_t), max(y_test_t)))
    plt.hist(y_test_t, bins=quality_interval)
    plt.show()


# Метод Gaussian Naive Bayes
def naive_bayes(X_train_n, X_test_n, y_train_n, y_test_n):
    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(X_train_n, y_train_n)
    y_pred_n = naive_bayes_model.predict(X_test_n)

    # Статистика
    print('Report showing the main classification metrics')
    print(classification_report(y_test_n, y_pred_n, zero_division=0))

    print('A matrix of inaccuracies for evaluating accuracy of the classification')
    print(confusion_matrix(y_test_n, y_pred_n))

    print('Score the X-train with Y-train is : ', naive_bayes_model.score(X_train_n, y_train_n))
    print('Score the X-test  with Y-test  is : ', naive_bayes_model.score(X_test_n, y_test_n))
    print('Evaluation of the Model Naive Bayes: accuracy score ', accuracy_score(y_test_n, y_pred_n))

    # Графическое отображение наиболее важных признаков
    #  6 - free sulfur dioxide
    #  7 - total sulfur dioxide
    # 11 - alcohol
    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    ax[0].scatter(X_test_n[:, 5], X_test_n[:, 6], c=y_pred_n)
    ax[0].set_title('Dependence of the free sulfur dioxide on total sulfur dioxide')

    ax[1].scatter(X_test_n[:, 5], X_test_n[:, 10], c=y_pred_n)
    ax[1].set_title('Dependence of the free sulfur dioxide on alcohol')

    ax[2].scatter(X_test_n[:, 6], X_test_n[:, 10], c=y_pred_n)
    ax[2].set_title('Dependence of the total sulfur dioxide on alcohol')
    plt.show()

    # Графическое отображение тестовых значений
    target_var1 = pd.crosstab(index=y_test_n, columns='% observations')
    plt.pie(target_var1['% observations'], labels=target_var1['% observations'].index, autopct='%.0f%%')
    plt.title('Array of test values')
    plt.show()

    # Графическое отображение предсказанных значений
    target_var2 = pd.crosstab(index=y_pred_n, columns='% observations')
    plt.pie(target_var2['% observations'], labels=target_var2['% observations'].index, autopct='%.0f%%')
    plt.title('Array of predicted values')
    plt.show()

    # Совпадение тестовых данных и предсказания
    plt.plot(y_test_n, label='Test')
    plt.plot(y_pred_n, label='Predication')
    plt.legend()
    plt.show()

    quality_interval = max(y_test_n) - min(y_test_n)
    plt.title('Quality from {} to {}'.format(min(y_test_n), max(y_test_n)))
    plt.hist(y_test_n, bins=quality_interval)
    plt.show()


# Набор данных
dataset = pd.read_csv('wine_quality_dataset.csv')

# Содержимое dataset
print('\nViewing five rows')
print(dataset.head())

# Вывод размерности
print('\nShape the DataSet : ', dataset.shape)

# Количество пропущенных значений
print('\nThe number of missing values in the entire Dataset')
print(dataset.isnull().sum())

# Описательная статистика
print('\nStatics')
print(dataset.describe().round(2))

# Удаление столбца Id
dataset.drop(columns='Id', inplace=True)

# Уникальные значения качества (quality)
print('\nThe value quality : ', dataset['quality'].unique())

# Группирование по значениям quality
ave_qu = dataset.groupby('quality').mean()
print('\nGroup by quality')
print(ave_qu.round(2))

# f1 = dataset['free sulfur dioxide'].values
# f2 = dataset['total sulfur dioxide'].values
# f3 = dataset['alcohol'].values
# X = np.array(list(zip((f1+f2+f3), f3)))

# Формирования набора данных
X = dataset.drop(columns='quality').values
y = dataset['quality'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('\nThe size of the arrays')
print('X train : ', X_train.shape)
print('X test  : ', X_test.shape)
print('Y train : ', y_train.shape)
print('Y test  : ', y_test.shape)

print('\nMethod K-nearest neighbors\n')
knn(X_train, X_test, y_train, y_test)

print('\nMethod Support Vector Classification\n')
svc(X_train, X_test, y_train, y_test)

print('\nMethod Decision Tree Classifier\n')
dtc(X_train, X_test, y_train, y_test)

print('\nMethod Naive Bayes\n')
naive_bayes(X_train, X_test, y_train, y_test)
