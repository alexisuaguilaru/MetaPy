class DifferentialEvolution:
    def __init__(self,objectiveFunction:function,initializeIndividual:function):
        """
            Class for Differential Evolution Metaheuristic
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals
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
        """
        self.populationSize = populationSize
        self.scalingFactor = scalingFactor
        self.crossoverRate = crossoverRate
        self.diffevol_InitializePopulation()
        for iteration in range(iterations):
            self.diffevol_IterativeSearch(iteration)
        return self.diffevol_OptimalSolution(populationSize)

    def diffevol_InitializePopulation(self):
        populationSize = self.populationSize
        self.population = [self.initializeIndividual() for _ in range(populationSize)]
        self.fitnessValuesPopulation = [self.objectiveFunction(*individual) for individual in self.population]

    def diffevol_IterativeSearch(self,iteration:int):
        populationSize = self.populationSize
        for indexIndividual in range(populationSize):
            mutantIndividual = self.diffevol_MutantOperation(indexIndividual)
            crossoverIndividual = self.diffevol_CrossoverOperation(indexIndividual,mutantIndividual)
            if (fitnessValue:=self.objectiveFunction(*crossoverIndividual)) <= self.fitnessValuesPopulation[indexIndividual]:
                self.population[indexIndividual] = crossoverIndividual
                self.fitnessValuesPopulation[indexIndividual] = fitnessValue

    def diffevol_MutantOperation(self,indexIndividual:int):
        """
            Method to apply Differential Mutant Operation to an individual
        """
        pass

    def diffevol_CrossoverOperation(self,indexIndividual:int,mutantIndividual):
        """
            Method to apply Differential Crossover Operation 
        """
        pass
    
    def diffevol_OptimalSolution(self,populationSize:int):
        """
            Method to return the optimal solution found
        """
        pass