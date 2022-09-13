import numpy as np
from scipy import spatial
from sko.PSO import PSO_TSP


def decrease(trace):
    for i in range(1, len(trace)):
        if trace[i] > trace[i-1]:
            trace[i] = trace[i-1]
    return trace


def evaluate(routine):
    num_points, = routine.shape
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


def PSO_exe(places, NIND, Max_iter):
    num_points = len(places)
    global distance_matrix
    distance_matrix = spatial.distance.cdist(places, places, metric='euclidean')
    # 执行PSO算法
    pso = PSO_TSP(func=evaluate, n_dim=num_points, size_pop=NIND, max_iter=Max_iter)

    # 结果输出
    Route, best_Dis, trace = pso.run()
    Route = list(Route)
    for i in range(len(Route)):
        Route[i] -= 1
    Route.append(Route[0])
    return best_Dis, Route, decrease(np.array(trace)[:, 0])

