import pandas as pd
import numpy as np
import random
import csv
from faker import Faker
import matplotlib.pyplot as plt
from matplotlib import style


# identifier, name, sex, birthYear, startingWorkYear, department, position, salary, completedProjectsCount
fake = Faker()
identifier = 0
structure_departament_position = {
    "Strategy": ["CEO ", "CTO", "CIO"],
    "Marketing": ["Marketing Manager", "Business Development Manager"],
    "Development": ["Product Manager", "Business Analyst", "Architect", "Tech Lead", "Team Lead",
                    "Developer", "DevOps", "Data Science"],
    "Testing": ["Tester", "Team Lead Testing"],
    "Resource": ["Resource Manager", "Recruiter", "HR Manager", "Vendor Manager"]
}


def get_id():
    return identifier


def get_name():
    return fake.name()


def capitalize(str):
    return str.capitalize()


def get_sex():
    return fake.simple_profile()['sex']


def get_birth_year():
    return fake.year()


def get_starting_work_year(birth_year):
    return int(birth_year) + 20


def get_department():
    return random.choice(list(structure_departament_position))


def get_position(departament):
    return random.choice(structure_departament_position[departament])


def get_salary():
    return round(random.uniform(50000, 750000), 2)


def get_completed_projects_count():
    return random.randint(5, 15)


# identifier, name, sex, birthYear, startingWorkYear, department, position, salary, completedProjectsCount
def generate_data():
    departament = get_department()
    birth_year = get_birth_year()
    return [get_id(), get_name(), get_sex(), birth_year, get_starting_work_year(birth_year), departament,
            get_position(departament), get_salary(), get_completed_projects_count()]


def analyze_numpy():
    with open('data.csv', 'r') as csvFile:
        list_rows = [list(currentRow) for currentRow in csv.reader(csvFile)]

    array_rows = np.array(list_rows)

    numbers = [int(item) for item in array_rows[1:, 0]]
    min_number = np.min(numbers)
    max_number = np.max(numbers)
    sum_number = np.sum(numbers)
    average_number = np.average(numbers)
    disp_number = np.var(numbers)
    standard_deviation_number = np.std(numbers)
    median_number = np.median(numbers)

    salary = [float(item) for item in array_rows[1:, 7]]
    min_salary = np.min(salary)
    max_salary = np.max(salary)
    sum_salary = np.sum(salary)
    average_salary = np.average(salary)
    disp_salary = np.var(salary)
    standard_deviation_salary = np.std(salary)
    median_salary = np.median(salary)

    projects = [int(item) for item in array_rows[1:, 8]]
    min_project = np.min(projects)
    max_project = np.max(projects)
    sum_project = np.sum(projects)
    average_project = np.average(projects)
    disp_project = np.var(projects)
    standard_deviation_project = np.std(projects)
    median_project = np.median(projects)

    gender = [item for item in array_rows[1:, 2]]
    arr_gender_f = np.sum(gender.count('F'))
    arr_gender_m = np.sum(gender.count('M'))

    names = [str(item) for item in array_rows[1:, 1]]
    name = np.unique(names)

    print('\n------------------')
    print('Numpy analyze\n')

    print('For identifier:')
    print('Min: ', min_number)
    print('Max: ', max_number)
    print('Sum: ', sum_number)
    print('Average: ', average_number)
    print('Disp: ', disp_number)
    print('Standard deviation: ', standard_deviation_number)
    print('Median: ', median_number)
    print()

    print('For salary:')
    print('Min: ', min_salary)
    print('Max: ', max_salary)
    print('Sum: ', sum_salary)
    print('Average: ', average_salary)
    print('Disp: ', disp_salary)
    print('Standard deviation: ', standard_deviation_salary)
    print('Median: ', median_salary)
    print()

    print('For projects:')
    print('Min: ', min_project)
    print('Max: ', max_project)
    print('Sum: ', sum_project)
    print('Average: ', average_project)
    print('Disp: ', disp_project)
    print('Standard deviation: ', standard_deviation_project)
    print('Median: ', median_project)
    print()

    print('For gender:')
    print('Count female: ', arr_gender_f)
    print('Count male: ', arr_gender_m)
    print()

    print('For names:')
    print('Unique: ', name.size/len(names))
    print()


def analyze_pandas():
    data = pd.read_csv('data.csv')

    numbers = data['identifier']
    min_number = numbers.min()
    max_number = numbers.max()
    sum_number = numbers.sum()
    average_number = numbers.mean()
    disp_number = numbers.var()
    standard_deviation_number = numbers.std()
    median_number = numbers.median()

    salary = data['salary']
    min_salary = salary.min()
    max_salary = salary.max()
    sum_salary = salary.sum()
    average_salary = salary.mean()
    disp_salary = salary.var()
    standard_deviation_salary = salary.std()
    median_salary = salary.median()

    projects = data['completedProjectsCount']
    min_project = projects.min()
    max_project = projects.max()
    sum_project = projects.sum()
    average_project = projects.mean()
    disp_project = projects.var()
    standard_deviation_project = projects.std()
    median_project = projects.median()

    gender = data['sex']
    arr_gender_f = gender.value_counts()['F']
    arr_gender_m = gender.value_counts()['M']

    names = data['name']
    name = names.unique()

    print('\n------------------')
    print('Pandas analyze\n')

    print('For identifier:')
    print('Min: ', min_number)
    print('Max: ', max_number)
    print('Sum: ', sum_number)
    print('Average: ', average_number)
    print('Disp: ', disp_number)
    print('Standard deviation: ', standard_deviation_number)
    print('Median: ', median_number)
    print()

    print('For salary:')
    print('Min: ', min_salary)
    print('Max: ', max_salary)
    print('Sum: ', sum_salary)
    print('Average: ', average_salary)
    print('Disp: ', disp_salary)
    print('Standard deviation: ', standard_deviation_salary)
    print('Median: ', median_salary)
    print()

    print('For projects:')
    print('Min: ', min_project)
    print('Max: ', max_project)
    print('Sum: ', sum_project)
    print('Average: ', average_project)
    print('Disp: ', disp_project)
    print('Standard deviation: ', standard_deviation_project)
    print('Median: ', median_project)
    print()

    print('For gender:')
    print('Count female: ', arr_gender_f)
    print('Count male: ', arr_gender_m)
    print()

    print('For names:')
    print('Unique: ', name.size/names.size)
    print()

    # Graph
    plt.figure(figsize=(10, 7), dpi=80)
    x = data['completedProjectsCount']
    y = data['salary']
    plt.bar(x, y, color='blue', edgecolor='black',
            linewidth=1)
    plt.title("Dependence of the number of projects on salary")
    plt.ylabel('Salary')
    plt.xlabel('Number of projects')
    plt.show()

    data_gender = [data['sex'].value_counts()['F'], data['sex'].value_counts()['M']]
    man_woman = ['Woman', 'Man']
    explode = (0.1, 0)
    fig1, ax1 = plt.subplots()
    ax1.pie(data_gender, explode=explode, labels=man_woman, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.show()

    plt.figure(figsize=(10, 7), dpi=80)
    style.use('ggplot')
    x = data['salary']
    y = data['department']
    plt.scatter(x, y, color='g')
    plt.title('Dependence of the number of department on salary')
    plt.xlabel('Salary')
    plt.ylabel('Department')
    plt.show()


with open('data.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=",", lineterminator="\r")
    writer.writerow(['identifier', 'name', 'sex', 'birthYear', 'startingWorkYear', 'department', 'position', 'salary',
                     'completedProjectsCount'])
    for i in range(1, 1001):
        identifier = i
        writer.writerow(generate_data())

analyze_numpy()
analyze_pandas()
