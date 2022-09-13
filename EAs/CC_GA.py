import geatpy as ea
from EAs import GA
from EAs.Problem import MyProblem
from EAs.templet.GA_templet import soea_SEGA_templet
import helps
import numpy as np


def CC_GA_exe(cities, NIND, Max_iter, name):
    label = helps.K_Nearest(cities, 50)
    # helps.draw_city_clustering(cities, label, name)
    sub_cities, sub_cities_num = helps.divide_cities(cities, label)
    Route = []
    for i in range(len(sub_cities)):
        best_Dis, sub_route, trace = GA.GA_exe(sub_cities[i], NIND, int(0.2 * Max_iter))
        route = helps.sub_num_real_num(sub_route, sub_cities_num[i])
        route.pop()
        Route.extend(route)
    best_Dis, Route, trace = GA_elite_exe(cities, NIND, Max_iter, Route[1:])
    return best_Dis, Route, trace


def GA_elite_exe(places, NIND, Max_iter, elite):
    """================================实例化问题对象============================"""
    problem = MyProblem(places)  # 生成问题对象
    """==================================种群设置=============================="""
    Encoding = 'P'  # 编码方式，采用排列编码
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    prophetPop = ea.Population(Encoding, Field, 1)
    prophetPop.Chrom = np.array([elite])
    """================================算法参数设置============================="""
    myAlgorithm = soea_SEGA_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = Max_iter  # 最大进化代数
    myAlgorithm.mutOper.Pm = 0.8  # 变异概率
    myAlgorithm.logTras = 0  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = False  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）

    """===========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run(prophetPop)  # 执行算法模板，得到最优个体以及最后一代种群

    """==================================输出结果=============================="""
    return BestIndi.ObjV[0], np.hstack([0, BestIndi.Phen[0, :], 0]), myAlgorithm.trace['f_best']






