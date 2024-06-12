class DifferentialEvolution:
    def __init__(self,objectiveFunction:function,initializeIndividual:function):
        """
            Class for Differential Evolution Metaheuristic
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals

            -- population : Solutions population
            -- fitnessValuesPopulation : Fitness values population
        """
        self.objectiveFunction = objectiveFunction
        self.initializeIndividual = initializeIndividual
        self.population = []
        self.fitnessValuesPopulation = []

    def __call__(self,iterations:int,populationSize:int,scalingFactor:float,crossoverRate:float):
        """
            Method to search optimal solution for objective function
            -- iterations : Amount of iterations
            -- populationSize : Parameter NP. Size of solutions population
            -- scalingFactor : Parameter F. Scaling factor for difference vector 
            -- crossoverRate : Parameter Cr. Crossover rate for crossover operation
        """
        objectiveFunction = self.objectiveFunction
        self.population = [self.initializeIndividual() for _ in range(populationSize)]
        self.fitnessValuesPopulation = [objectiveFunction(*individual) for individual in self.population]
        for iteration in range(iterations):
            for indexIndividual in range(populationSize):
                mutantIndividual = self._differentialMutantOperation(scalingFactor,indexIndividual)
                crossoverIndividual = self._differentialCrossoverOperation(crossoverRate,indexIndividual,mutantIndividual)
                if (fitnessValue:=objectiveFunction(*crossoverIndividual)) <= self.fitnessValuesPopulation[indexIndividual]:
                    self.population[indexIndividual] = crossoverIndividual
                    self.fitnessValuesPopulation[indexIndividual] = fitnessValue
        return self._optimalSolution(populationSize)

    def _differentialMutantOperation(self,scalingFactor:float,indexIndividual:int):
        """
            Private method to apply Differential Mutant Operation to an individual
        """
        pass

    def _differentialCrossoverOperation(self,crossoverRate:float,indexIndividual:int,mutantIndividual):
        """
            Private method to apply Differential Crossover Operation 
        """
        pass
    
    def _optimalSolution(self,populationSize:int):
        """
            Private method to return the optimal solution found
        """
        pass