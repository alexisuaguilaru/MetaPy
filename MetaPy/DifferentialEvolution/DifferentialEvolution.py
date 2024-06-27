class DifferentialEvolution:
    def __init__(self,objectiveFunction,initializeIndividual):
        """
            Class for Differential Evolution Metaheuristic
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals

            Based on DE/rand/1/bin
        """
        self.objectiveFunction = objectiveFunction
        self.initializeIndividual = initializeIndividual

    def __call__(self,iterations:int,populationSize:int,scalingFactor:float,crossoverRate:float):
        """
            Method to search optimal solution for objective function
            -- iterations : Amount of iterations
            -- populationSize : Parameter NP. Size of solutions population
            -- scalingFactor : Parameter F. Scaling factor for difference vector 
            -- crossoverRate : Parameter Cr. Crossover rate for crossover operation

            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the population at each iteration
        """
        self.populationSize = populationSize
        self.scalingFactor = scalingFactor
        self.crossoverRate = crossoverRate
        self.SnapshotsSaved = []
        self.diffevol_InitializePopulation()
        self.optimalIndividual , self.optimalValue = self.diffevol_BestOptimalFound()
        for iteration in range(iterations):
            self.diffevol_IterativeSearch(iteration)
        return self.optimalIndividual , self.SnapshotsSaved

    def diffevol_InitializePopulation(self) -> None:
        """
            Method to initialize population and fitnessValuesPopulation attributes  
        """
        import numpy as np
        self.population = np.array([self.initializeIndividual() for _ in range(self.populationSize)])
        self.fitnessValuesPopulation = np.array([self.objectiveFunction(individual) for individual in self.population])

    def diffevol_IterativeSearch(self,iteration:int) -> None:
        """
            Method to search optimal solutions iteratively 
            -- iteration : Number of iteration
        """
        populationSize = self.populationSize
        for indexIndividual in range(populationSize):
            mutatedIndividual = self.diffevol_MutationOperation()
            crossoverIndividual = self.diffevol_CrossoverOperation(indexIndividual,mutatedIndividual)
            if (fitnessValue:=self.objectiveFunction(crossoverIndividual)) <= self.fitnessValuesPopulation[indexIndividual]:
                self.population[indexIndividual] = crossoverIndividual
                self.fitnessValuesPopulation[indexIndividual] = fitnessValue
                if self.fitnessValuesPopulation[indexIndividual] < self.optimalValue:
                    self.optimalValue = self.fitnessValuesPopulation[indexIndividual]
                    self.optimalIndividual = self.population[indexIndividual]
        self.diffevol_SnapshotPopulation(iteration)

    def diffevol_MutationOperation(self):
        """
            Method to apply Differential Evolution Mutation Operation to a random individual
        """
        from random import sample
        randomIndex_1 , randomIndex_2 , randomIndex_3 = sample(range(self.populationSize),k=3)
        randomIndividual_1 , randomIndividual_2 , randomIndividual_3 = self.population[randomIndex_1] , self.population[randomIndex_2] , self.population[randomIndex_3]
        return  randomIndividual_1 + self.scalingFactor*(randomIndividual_2 - randomIndividual_3)

    def diffevol_CrossoverOperation(self,indexIndividual:int,mutatedIndividual):
        """
            Method to apply Differential Evolution Crossover Operation
            -- indexIndividual : Individual's index to be crossover
            -- mutantIndividual : Individual to be crossover
        """
        from random import random , randrange
        from copy import deepcopy
        individual = self.population[indexIndividual]
        crossoverIndividual = deepcopy(individual)
        indexMutated = randrange(0,len(crossoverIndividual))
        for index , componentMutatedIndividual in enumerate(mutatedIndividual):
            if (random() <= self.crossoverRate) or (index == indexMutated):
                crossoverIndividual[index] = componentMutatedIndividual
        return crossoverIndividual

    def diffevol_SnapshotPopulation(self,iteration:int):
        """
            Method to save a snapshot of the population at iteration-st
            -- iteration : Number of iteration 
        """
        from copy import deepcopy
        self.SnapshotsSaved.append((iteration,deepcopy(self.population),self.optimalValue))

    def diffevol_BestOptimalFound(self):
        """
            Method to return the best optimal found's individual and function value
        """
        bestIndexIndividual = 0
        for indexIndividual in range(1,self.populationSize):
            if self.fitnessValuesPopulation[indexIndividual] < self.fitnessValuesPopulation[indexIndividual]:
                bestIndexIndividual = indexIndividual
        return self.population[bestIndexIndividual] , self.fitnessValuesPopulation[indexIndividual]