import geatpy as ea
from EAs import GA
from EAs.Problem import MyProblem
from EAs.templet.GA_templet import soea_SEGA_templet
import helps
import numpy as np
from PtrNet.dataset import DataGenerator
from PtrNet.actor import Actor
import tensorflow as tf


def CC_GA_exe(cities, NIND, Max_iter, config, actor, name):
    label = helps.K_Nearest(cities, 20)
    # helps.draw_city_clustering(cities, label, name)
    sub_cities, sub_cities_num = helps.divide_cities(cities, label)
    Route = []

    for i in range(len(sub_cities)):
        if len(sub_cities[i]) == 20:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                training_set = DataGenerator(config, sub_cities[i])

                # Get test data
                input_batch = training_set.test_batch()
                feed = {actor.input_: input_batch}

                # Sample solutions
                positions, _, _, _ = sess.run([actor.positions, actor.reward, actor.train_step1, actor.train_step2],
                                          feed_dict=feed)
                sub_route = positions[0]
        else:
            best_Dis, sub_route, trace = GA.GA_exe(sub_cities[i], NIND, 10)
        sub_route = list(sub_route)
        sub_route.pop()
        route = helps.sub_num_real_num(sub_route, sub_cities_num[i])
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






