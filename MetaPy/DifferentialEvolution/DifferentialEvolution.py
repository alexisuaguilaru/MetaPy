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
        return self.diffevol_OptimalSolution()

    def diffevol_InitializePopulation(self) -> None:
        """
            Method to initialize population and fitnessValuesPopulation attributes  
        """
        populationSize = self.populationSize
        self.population = [self.initializeIndividual() for _ in range(populationSize)]
        self.fitnessValuesPopulation = [self.objectiveFunction(*individual) for individual in self.population]

    def diffevol_IterativeSearch(self,iteration:int) -> None:
        """
            Method to search optimal solutions
            -- iteration : Number of iteration
        """
        populationSize = self.populationSize
        for indexIndividual in range(populationSize):
            mutantIndividual = self.diffevol_MutationOperation(indexIndividual)
            crossoverIndividual = self.diffevol_CrossoverOperation(indexIndividual,mutantIndividual)
            if (fitnessValue:=self.objectiveFunction(*crossoverIndividual)) <= self.fitnessValuesPopulation[indexIndividual]:
                self.population[indexIndividual] = crossoverIndividual
                self.fitnessValuesPopulation[indexIndividual] = fitnessValue

    def diffevol_MutationOperation(self,indexIndividual:int):
        """
            Method to apply Differential Evolution Mutation Operation to an individual
            -- indexIndividual : Individual's index to be mutated
        """
        pass

    def diffevol_CrossoverOperation(self,indexIndividual:int,mutantIndividual):
        """
            Method to apply Differential Evolution Crossover Operation
            -- indexIndividual : Individual's index to be crossover
            -- mutantIndividual : Individual to be crossover
        """
        pass
    
    def diffevol_OptimalSolution(self):
        """
            Method to return the optimal solution found
        """
        pass