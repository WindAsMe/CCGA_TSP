import geatpy as ea
from EAs.Problem import MyProblem
import numpy as np
from EAs.templet.GA_templet import soea_SEGA_templet


def GA_exe(places, NIND, Max_iter):
    """================================实例化问题对象============================"""
    problem = MyProblem(places)  # 生成问题对象
    """==================================种群设置=============================="""
    Encoding = 'P'  # 编码方式，采用排列编码
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = soea_SEGA_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = Max_iter  # 最大进化代数
    myAlgorithm.mutOper.Pm = 0.8  # 变异概率
    myAlgorithm.logTras = 0  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = False  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）

    """===========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群

    """==================================输出结果=============================="""
    return BestIndi.ObjV[0], np.hstack([0, BestIndi.Phen[0, :], 0]), myAlgorithm.trace['f_best']

