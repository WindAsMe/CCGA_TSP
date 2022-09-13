import time

import numpy as np
import os
import pandas as pd
import helps
from EAs import GA, PSO, CC_GA, ACO


if __name__ == "__main__":
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/Data/tsp_data/"


    # tsp_files = os.listdir(data_path)
    tsp_files = ['lim963.tsp', 'dca1389.tsp', 'fra1488.tsp', 'xit1083.tsp', 'dja1436.tsp', 'rbv1583.tsp', 'dkg813.tsp',
                 'dka1376.tsp', 'icw1483.tsp', 'pbd984.tsp']
    names = ['lim963', 'dca1389', 'fra1488', 'xit1083', 'dja1436', 'rbv1583', 'dkg813', 'dka1376', 'icw1483', 'pbd984']
    # print(tsp_files)
    scales = [963, 1389, 1488, 1083, 1436, 1583, 813, 1376, 483, 984]
    skip = 8
    trial_runs = 30
    NIND = 100
    Max_iter = 1000
    for run in range(trial_runs):
        # for i in range(len(tsp_files)):
        for inst in range(1):
            GA_best_path = os.path.dirname(os.path.abspath(__file__)) + "/Data/GA/best_Dis/" + names[inst] + ".csv"
            GA_route_path = os.path.dirname(os.path.abspath(__file__)) + "/Data/GA/Route/" + names[inst] + ".csv"
            GA_trace_path = os.path.dirname(os.path.abspath(__file__)) + "/Data/GA/trace/" + names[inst] + ".csv"

            PSO_best_path = os.path.dirname(os.path.abspath(__file__)) + "/Data/PSO/best_Dis/" + names[inst] + ".csv"
            PSO_route_path = os.path.dirname(os.path.abspath(__file__)) + "/Data/PSO/Route/" + names[inst] + ".csv"
            PSO_trace_path = os.path.dirname(os.path.abspath(__file__)) + "/Data/PSO/trace/" + names[inst] + ".csv"

            ACO_best_path = os.path.dirname(os.path.abspath(__file__)) + "/Data/ACO/best_Dis/" + names[inst] + ".csv"
            ACO_route_path = os.path.dirname(os.path.abspath(__file__)) + "/Data/ACO/Route/" + names[inst] + ".csv"
            ACO_trace_path = os.path.dirname(os.path.abspath(__file__)) + "/Data/ACO/trace/" + names[inst] + ".csv"

            CCGA_best_path = os.path.dirname(os.path.abspath(__file__)) + "/Data/CCGA/best_Dis/" + names[inst] + ".csv"
            CCGA_route_path = os.path.dirname(os.path.abspath(__file__)) + "/Data/CCGA/Route/" + names[inst] + ".csv"
            CCGA_trace_path = os.path.dirname(os.path.abspath(__file__)) + "/Data/CCGA/trace/" + names[inst] + ".csv"

            cities = helps.read_tsp(data_path + tsp_files[inst], skip)

            """Conventional GA and PSO"""
            # time_GA_start = time.time()
            best_Dis_GA, Route_GA, trace_GA = GA.GA_exe(cities, NIND, Max_iter)
            helps.write_result(GA_best_path, best_Dis_GA)
            helps.write_result(GA_route_path, Route_GA)
            helps.write_result(GA_trace_path, trace_GA)
            # time_GA_end = time.time()
            # print("GA run time: ", time_GA_end - time_GA_start)

            # time_PSO_start = time.time()
            best_Dis_PSO, Route_PSO, trace_PSO = PSO.PSO_exe(cities, NIND, Max_iter)
            helps.write_result(PSO_best_path, best_Dis_PSO)
            helps.write_result(PSO_route_path, Route_PSO)
            helps.write_result(PSO_trace_path, trace_PSO)
            # time_PSO_end = time.time()
            # print("PSO run time: ", time_PSO_end - time_PSO_start)

            best_Dis_ACO, Route_ACO, trace_ACO = ACO.ACO_exe(cities, NIND, Max_iter)
            helps.write_result(ACO_best_path, best_Dis_ACO)
            helps.write_result(ACO_route_path, Route_ACO)
            helps.write_result(ACO_trace_path, trace_ACO)

            """Cooperative Coevolution"""
            # time_CCGA_start = time.time()
            # best_Dis_CCGA, Route_CCGA, trace_CCGA = CC_GA.CC_GA_exe(cities, NIND, Max_iter, names[inst])
            # helps.write_result(CCGA_best_path, best_Dis_CCGA)
            # helps.write_result(CCGA_route_path, Route_CCGA)
            # helps.write_result(CCGA_trace_path, trace_CCGA)
            # time_CCGA_end = time.time()
            # print("CCGA run time: ", time_CCGA_end - time_CCGA_start)

