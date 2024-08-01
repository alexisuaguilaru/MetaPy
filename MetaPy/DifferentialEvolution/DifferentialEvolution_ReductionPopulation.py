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
    
    def __call__(self, iterations:int , populationSize:int , scalingFactor:float , crossoverRate:float) -> tuple:
        """
            Method to search optimal solution for objective function
            -- iterations : Amount of iterations
            -- populationSize : Parameter NP. Size of solutions population
            -- scalingFactor : Parameter F. Scaling factor for difference vector 
            -- crossoverRate : Parameter Cr. Crossover rate for crossover operation
            No iteration parameter is required because ends when the population size is equal to 1
            
            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the population at each iteration
        """
        from math import sqrt , ceil
        self.populationSize = populationSize
        self.numberClusters = ceil(sqrt(populationSize))
        self.scalingFactor = scalingFactor
        self.crossoverRate = crossoverRate
        self.SnapshotsSaved = []
        self.diffevol_InitializePopulation()
        self.optimalIndividual , self.optimalValue = self.diffevol_BestOptimalFound()
        for iteration in range(iterations):
            if iteration%20 == 0 and iteration != 0:
                self.diffevol_redpop_ReductionPopulation()
                self.diffevol_redpop_IncreasePopulation()
            self.diffevol_IterativeSearch(iteration)
            iteration += 1
        return self.optimalIndividual , self.SnapshotsSaved

    def diffevol_redpop_ReductionPopulation(self):
        """
            Method to reduce population size using k-means
        """
        from sklearn.cluster import k_means
        populationLabels = k_means(self.population,self.numberClusters)[1]
        self.diffevol_redpop_ReinitializePopulation(populationLabels)

    def diffevol_redpop_ReinitializePopulation(self, populationLabels):
        """
            Method to reinitialize population with the best individuals and theirs fitness values
        """
        from collections import defaultdict
        import numpy as np
        population , fitnessValuesPopulation = defaultdict(int) , defaultdict(int)
        for labelCluster , individual , individualFitnessValue in zip(populationLabels,self.population,self.fitnessValuesPopulation):
            if population[labelCluster] == 0:
                population[labelCluster] = individual
                fitnessValuesPopulation[labelCluster] = individualFitnessValue
            else:
                if individualFitnessValue < fitnessValuesPopulation[labelCluster]:
                    population[labelCluster] = individual
                    fitnessValuesPopulation[labelCluster] = individualFitnessValue
        self.population = np.array(population)
        self.fitnessValuesPopulation = np.array(fitnessValuesPopulation)

    def diffevol_redpop_IncreasePopulation(self):
        """
            Method to generate new solutions based on best solutions
            at each cluster
        """
        import numpy as np
        createdIndividuals , createdIndividualsFitnessValues = [] , []
        for _ in range(self.populationSize-self.numberClusters):
            createdIndividual = self.diffevol_redpop_CreateNewIndividual()
            createdIndividualFitnessValue = self.objectiveFunction(createdIndividual)
            createdIndividuals.append(createdIndividual)
            createdIndividualsFitnessValues.append(createdIndividualFitnessValue)
        self.population = np.concatenate((self.population,createdIndividuals))
        self.fitnessValuesPopulation = np.concatenate((self.fitnessValuesPopulation,createdIndividualsFitnessValues))
        self.optimalIndividual , self.optimalValue = self.diffevol_BestOptimalFound()

    def diffevol_redpop_CreateNewIndividual(self):
        """
            Method to generate an individual solution based 
            on best solutions at each cluster
        """
        import numpy as np
        randomWeighVector = 20*np.random.rand(self.numberClusters,1) - 10
        return self.population * randomWeighVector