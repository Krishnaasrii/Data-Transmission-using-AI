import numpy as np
from random import randint, uniform
import SwarmPackagePy
from SwarmPackagePy import intelligence


class ABC(intelligence.sw):
    """
    Artificial Bee Algorithm
    """

    def __init__(self, n, function, lb, ub, dimension, iteration):
        """
        :param n: number of agents
        :param function: test function
        :param lb: lower limits for plot axes
        :param ub: upper limits for plot axes
        :param dimension: space dimension
        :param iteration: number of iterations
        """

        super(ABC, self).__init__()

        self.__function = function

        self.__agents = n  #np.random.uniform(lb, ub, (n, dimension))
        self._points(self.__agents)

        Pbest = self.__agents[np.array([function(x) for x in self.__agents]).argmin()]
        Gbest = Pbest

        if len(n) <= 10:
            count = n - n // 2, 1, 1, 1
        else:
            a = len(n) // 10
            b = 5
            c = (n - a * b - a) // 2
            d = 2
            count = a, b, c, d

        for t in range(iteration):

            fitness = [function(x) for x in self.__agents]
            sort_fitness = [function(x) for x in self.__agents]
            sort_fitness.sort()
            sort_fitness = np.asarray(sort_fitness)

            best = [self.__agents[i] for i in
                    [fitness.index(x) for x in sort_fitness[:count[0]]]]
            selected = [self.__agents[i]
                        for i in [fitness.index(x)
                                  for x in sort_fitness[1:5]]]
            
        self._set_Gbest(Gbest)

    def __new(self, l, c, lb, ub):

        bee = []
        for i in l:
            new = [self.__neighbor(i, lb, ub) for k in range(c)]
            bee += new
        bee += l

        return bee

    def __neighbor(self, who, lb, ub):

        neighbor = np.array(who) + uniform(-1, 1) * (
            np.array(who) - np.array(
                self.__agents[randint(0, len(self.__agents) - 1)]))
        neighbor = np.clip(neighbor, lb, ub)

        return list(neighbor)
