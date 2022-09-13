import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from sko.ACA import ACA_TSP
from matplotlib.ticker import FormatStrFormatter
from time import perf_counter


def decrease(trace):
    for i in range(1, len(trace)):
        if trace[i] > trace[i-1]:
            trace[i] = trace[i-1]
    return trace


def cal_total_distance(routine):
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


def ACO_exe(places, NIND, Max_iter):
    num_points = len(places)
    global distance_matrix
    distance_matrix = spatial.distance.cdist(places, places, metric='euclidean')
    aca = ACA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=NIND, max_iter=Max_iter, distance_matrix=distance_matrix)

    Route, best_Dis, trace = aca.run()
    return [best_Dis], Route, decrease(trace)
