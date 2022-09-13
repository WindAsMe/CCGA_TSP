import numpy as np

import helps
from EAs.PSO_templet import PSO_TSP
from EAs import PSO
from scipy import spatial


def CC_PSO_exe(cities, NIND, Max_iter, name):
    label = helps.K_Nearest(cities, 50)
    # helps.draw_city_clustering(cities, label, name)
    sub_cities, sub_cities_num = helps.divide_cities(cities, label)
    Route = []
    for i in range(len(sub_cities)):
        best_Dis, sub_route, trace = PSO.PSO_exe(sub_cities[i], NIND, int(0.2 * Max_iter))
        sub_route.pop()
        route = helps.sub_num_real_num(sub_route, sub_cities_num[i])
        Route.extend(route)

    global distance_matrix
    distance_matrix = spatial.distance.cdist(cities, cities, metric='euclidean')
    best_Dis, Route, trace = PSO_elite_exe(cities, NIND, int(0.8 * Max_iter), Route)
    Route = list(Route)
    Route.append(Route[0])
    return best_Dis, Route, np.array(trace)[:, 0]


def PSO_elite_exe(places, NIND, Max_iter, elite_x):
    num_points = len(places)
    # 执行蚁群(PSO)算法
    PSO = PSO_TSP(func=evaluate, n_dim=num_points, elite_x=elite_x, size_pop=NIND, max_iter=Max_iter)

    # 结果输出
    Route, best_Dis, trace = PSO.run()
    return best_Dis, Route, trace


def decrease(trace):
    for i in range(1, len(trace)):
        if trace[i] > trace[i-1]:
            trace[i] = trace[i-1]
    return trace


def evaluate(routine):
    num_points = len(routine)
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


def print_route(best_points):
    result_cur_best = []
    for i in best_points:
        result_cur_best += [i]
    for i in range(len(result_cur_best)):
        result_cur_best[i] += 1
    result_path = result_cur_best
    result_path.append(result_path[0])
    return result_path