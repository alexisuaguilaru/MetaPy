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
        from math import sqrt , ceil
        from sklearn.cluster import k_means
        numClusters = ceil(sqrt(self.populationSize))
        populationLabels = k_means(self.population,numClusters)[1]
        self.diffevol_redpop_ReinitializePopulation(populationLabels)
        self.populationSize = numClusters 

    def diffevol_redpop_ReinitializePopulation(self,populationLabels):
        """
            Method to reinitialize population with the best individuals and theirs fitness values
        """
        from collections import defaultdict
        import numpy as np
        populationIndividuals = defaultdict(int)
        for labelCluster , individidual , individualFitnessValue in zip(populationLabels,self.population,self.fitnessValuesPopulation):
            if populationIndividuals[labelCluster] == 0:
                populationIndividuals[labelCluster] = (individidual,individualFitnessValue)
            else:
                if individualFitnessValue < populationIndividuals[labelCluster][1]:
                    populationIndividuals[labelCluster] = (individidual,individualFitnessValue)
        population , fitnessValuesPopulation = [] , []
        [population.append(individidual),fitnessValuesPopulation.append(fitnessValue)  for individual,fitnessValue in populationIndividuals.values()]
        self.population = np.array(population)
        self.fitnessValuesPopulation = np.array(fitnessValuesPopulation)

    def diffevol_redpop_IncreasePopulation(self):
        """
            Method to generate new solutions based on best solutions
            at each cluster
        """
        from math import sqrt , ceil
        from random import randint
        numClusters = ceil(sqrt(self.populationSize))
        numIncreasePopulation = randint(0,numClusters)*numClusters
        newIndividuals = []
        for _ in range(numIncreasePopulation):
            individualGenerated = self.diffevol_redpop_ConvexCombination()
            newIndividuals.append(individualGenerated)
        self.populationSize += numIncreasePopulation

    def diffevol_redpop_ConvexCombination(self):
        pass