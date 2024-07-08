from DifferentialEvolution import DifferentialEvolution

class DifferentialEvolution_ReductionPopulation(DifferentialEvolution):
    def __init__(self, objectiveFunction, initializeIndividual):
        """
            Class for Differential Evolution Metaheuristic Variant which use
            K-Means Algorithm for improving solutions
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals

            Based on DE/rand/1/bin
        """
        super().__init__(objectiveFunction, initializeIndividual)
    
    def __call__(self, populationSize:int , scalingFactor:float , crossoverRate:float) -> tuple:
        """
            Method to search optimal solution for objective function
            -- populationSize : Parameter NP. Size of solutions population
            -- scalingFactor : Parameter F. Scaling factor for difference vector 
            -- crossoverRate : Parameter Cr. Crossover rate for crossover operation
            No iteration parameter is required because ends when the population size is equal to 1
            
            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the population at each iteration
        """
        self.populationSize = populationSize
        self.scalingFactor = scalingFactor
        self.crossoverRate = crossoverRate
        self.SnapshotsSaved = []
        self.diffevol_InitializePopulation()
        self.optimalIndividual , self.optimalValue = self.diffevol_BestOptimalFound()
        iteration = 0
        while self.population > 1:
            if iteration%10 == 0 and iteration != 0:
                self.diffevol_redpop_ReductionPopulation()
                self.diffevol_redpop_IncreasePopulation()
            self.diffevol_IterativeSearch(iteration)
            iteration += 1
        return self.optimalIndividual , self.SnapshotsSaved

    def diffevol_redpop_ReductionPopulation(self):
        """
            Method to reduce population size using k-means
        """
        pass

    def diffevol_redpop_IncreasePopulation(self):
        """
            Method to generate new solutions based on best solutions
            at each cluster
        """
        pass