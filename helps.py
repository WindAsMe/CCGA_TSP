import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt


def read_matrix(file_path):
    matrix = []
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_double = [float(i) for i in row]
            matrix.append(row_double)
    return matrix


def read_tsp(file_path, skip):
    df = pd.read_csv(file_path, sep=" ", skiprows=skip, header=None, encoding='utf8')
    city_x = np.array(df[1][0:-1])
    city_y = np.array(df[2][0:-1])
    points = list(zip(city_x, city_y))
    return points


def K_Nearest(data, sub_size):
    size = len(data)
    num = int(size / sub_size)
    rest = size
    if size % sub_size != 0:
        num += 1
    clusters = np.zeros(size)
    flag = 0
    for i in range(size):
        if clusters[i] != 0:
            continue
        flag += 1
        clusters[i] = flag
        dis = [float("inf")] * size
        for j in range(i+1, size):
            if clusters[j] == 0:
                dis[j] = Dis(data[i], data[j])

        sort_index = np.argsort(dis)
        if rest >= sub_size:
            sort_index = sort_index[0:sub_size-1]
        else:
            sort_index = sort_index[0:rest-1]
        for ind in sort_index:
            clusters[ind] = flag
        rest -= sub_size
    return clusters


def Dis(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def draw_city_clustering(data, label, name):
    font_title = {'size': 18}
    plt.title(name, font_title)
    category = max(label)
    for i in range(1, int(category+1)):
        sub_data = []
        for j in range(len(data)):
            if label[j] == i:
                sub_data.append(data[j])
        sub_data = np.array(sub_data)
        plt.scatter(sub_data[:, 0], sub_data[:, 1])
    plt.show()


def divide_cities(cities, label):
    category = int(max(label))
    sub_cities = []
    sub_cities_num = []
    for i in range(category):
        sub_cities.append([])
        sub_cities_num.append([])

    for i in range(len(cities)):
        sub_cities[int(label[i] - 1)].append(cities[i])
        sub_cities_num[int(label[i] - 1)].append(i)
    return sub_cities, sub_cities_num


def sub_num_real_num(sub_route, real_route):
    Route = []
    for var in sub_route:
        Route.append(real_route[var])
    return Route


def write_result(path, data):
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def draw_tour(Route, name, dis=None):
    Route = np.array(Route)
    plt.plot(Route[:, 0], Route[:, 1], 'o--')
    font_title = {'size': 18}
    plt.title("Elite of" + name, font_title)
    x_title = {'size': 14}
    if dis is not None:
        plt.xlabel("Distance =" + str(dis), x_title)
    plt.legend()
    plt.show()


