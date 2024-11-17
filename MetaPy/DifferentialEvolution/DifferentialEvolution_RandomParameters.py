import random

from DifferentialEvolution import DifferentialEvolution

class DifferentialEvolution_RandomParameters(DifferentialEvolution):
    def __init__(self, objectiveFunction, initializeIndividual):
        """
            Class for Differential Evolution Metaheuristic where 
            their parameters (scaling factor and crossover rate)
            randomly change each iteration
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals

            Based on DE/rand/1/bin
        """
        super().__init__(objectiveFunction, initializeIndividual)

    def __call__(self, iterations, populationSize):
        """
            Method to search optimal solution for objective function
            -- iterations : Amount of iterations
            -- populationSize : Parameter NP. Size of solutions population

            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the population at each iteration
        """
        return super().__call__(iterations, populationSize, 0, 0)
    
    def diffevol_IterativeSearch(self, iteration):
        self.scalingFactor = random.random()
        self.crossoverRate = random.random()
        super().diffevol_IterativeSearch(iteration)