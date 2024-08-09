from typing import Any


class GreyWolfOptimizer:
    def __init__(self,objectiveFunction,initializeIndividual):
        """
            Class for Grey Wolf Optimizer Metaheuristic
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals

            Based on GWO
        """
        self.objectiveFunction = objectiveFunction
        self.initializeIndividual = initializeIndividual
    
    def __call__(self, iterations:int ,populationSize:int):
        import numpy as np
        self.populationSize = populationSize
        self.GWO_InitializePopulation()
        vector_a = np.full(len(self.population[0]),2,dtype=np.float64)
        functionDecreasedVector_a = lambda iteration: vector_a - 2*iteration/(iterations-1)
        for iteration in range(iteration):
            vectorA = 
            vectorC = 

    def GWO_InitializePopulation(self):
        """
            Method to initialize population and fitnessValuesPopulation attributes  
        """
        import numpy as np
        self.population = np.array([self.initializeIndividual() for _ in range(self.populationSize)])
        self.fitnessValuesPopulation = np.array([self.objectiveFunction(individual) for individual in self.population])