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
        self.GWO_solutionsAlphaBettaDelta()
        vector_a = np.full(len(self.population[0]),2,dtype=np.float64)
        for iteration in range(iteration+1):
            vectorA = 
            vectorC =  
            self.GWO_decreaseVector_a(vector_a,iteration,iterations)
            
    def GWO_initializePopulation(self):
        """
            Method to initialize population and fitnessValuesPopulation attributes  
        """
        import numpy as np
        self.population = np.array([self.initializeIndividual() for _ in range(self.populationSize)])
        self.fitnessValuesPopulation = np.array([self.objectiveFunction(individual) for individual in self.population])

    def GWO_decreaseVector_a(self,vector_a,iteration,iterations):
        """
            Method to linearly decrease the vector a's entries 
        """
        import numpy as np
        if iteration == iterations-1:
                vector_a = np.full(len(self.population[0]),0,dtype=np.float64)
        else:
            vector_a -= 2/iterations

    def GWO_solutionsAlphaBettaDelta(self):
        """
            Method to initialize alpha (best), betta (second) and delta (third) solutions
        """
        import numpy as np
        self.indexAlphaBettaDelta = np.array([0,0,0])
        for indexSolution , fitnessValueSolution in enumerate(self.fitnessValuesPopulation):
            if fitnessValueSolution >= self.fitnessValuesPopulation(self.indexAlphaBettaDelta[0]):
                self.indexAlphaBettaDelta[0] , self.indexAlphaBettaDelta[1] , self.indexAlphaBettaDelta[2] = indexSolution , self.indexAlphaBettaDelta[0] , self.indexAlphaBettaDelta[1]
            elif fitnessValueSolution >= self.fitnessValuesPopulation(self.indexAlphaBettaDelta[1]):
                self.indexAlphaBettaDelta[1] , self.indexAlphaBettaDelta[2] = indexSolution , self.indexAlphaBettaDelta[1]
            elif fitnessValueSolution > self.fitnessValuesPopulation(self.indexAlphaBettaDelta[2]):
                self.indexAlphaBettaDelta[2] = indexSolution