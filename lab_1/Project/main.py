# Дан список стран и языков на которых говорят в этой стране в формате
# <Название Страны> : <язык1> <язык2> <язык3> ... в файле file.txt.
# На ввод задается N - длина списка и список языков.
# Для каждого языка укажите, в каких странах на нем говорят

# На ввод подаем количество языком, затем вводим каждый язык с новой строчки
# Пример:
# 3
# русский
# английский
# немецкий

dict_main = {}
with open('file.txt', 'r', encoding='utf-8') as file:
    for line in file:
        key, *value = line.split(':')
        dict_main[key] = value
        # print(key, ':', ' '.join(str(x) for x in dict_main[key]))

dict_tmp = {}
country = []

for _ in range(int(input())):
    language = input().strip()
    for key, value in dict_main.items():
        for val in value:
            if language in val:
                country.append(key)
                break

    dict_tmp[language] = country
    country = []

for key, value in dict_tmp.items():
    for val in value:
        print(val, end=' ')
    print()
