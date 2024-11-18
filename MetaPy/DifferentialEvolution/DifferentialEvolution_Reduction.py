import numpy as np

from .DifferentialEvolution import DifferentialEvolution

class DifferentialEvolution_Reduction(DifferentialEvolution):
    def __init__(self, objectiveFunction, initializeIndividual, clusteringAlgorithm):
        """
        Base Class for Differential Evolution Metaheuristic with Population Reduction 
        based on Clustering Algorithms
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals
            -- clusteringAlgorithm : Algorithm to execute clustering from Scikit Learn

            Based on DE/rand/1/bin
        """
        self.baseClusteringAlgorithm = clusteringAlgorithm
        super().__init__(objectiveFunction, initializeIndividual)
    
    def __call__(self, iterations:int, populationSize:int, scalingFactor:float, crossoverRate:float , clusteringAlgorithm_kw:dict = {}):
        """
            Method to search optimal solution for objective function
            -- iterations : Amount of iterations
            -- populationSize : Parameter NP. Size of solutions population
            -- scalingFactor : Parameter F. Scaling factor for difference vector 
            -- crossoverRate : Parameter Cr. Crossover rate for crossover operation
            -- clusteringAlgorithm_kw : Keywords for clustering function

            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the population at each iteration
        """
        self.totalIterations = iterations
        self.clusteringAlgorithm_kw = clusteringAlgorithm_kw
        return super().__call__(iterations, populationSize, scalingFactor, crossoverRate)
    
    def diffevol_IterativeSearch(self, iteration: int):
        if self.diffevol_reduction_ApplyPopulationReduction(iteration):
            self.diffevol_reduction_InitializePopulationRepresentatives()
            self.optimalIndividual , self.optimalValue = self.diffevol_BestOptimalFound()
        super().diffevol_IterativeSearch(iteration)

    def diffevol_reduction_ApplyPopulationReduction(self,iteration:int) -> bool:
        """
            Method to determine whether it is necessary apply a 
            reduction of the population based on the criteria.

            -- iteration : Number of iteration  

            Return a boolean value based on the criteria.
        """
        pass

    def diffevol_reduction_InitializePopulationRepresentatives(self):
        """
            Method to reinitialize the population with the representatives of each cluster
        """
        clustersRepresentativeIndividuals = self.diffevol_reduction_ClustersRepresentatives()
        populationClusters , population , fitnessValuesPopulation = self.diffevol_reduction_UnpackClustersRepresentatives(clustersRepresentativeIndividuals)
        self.populationClusters = np.array(populationClusters) 
        self.population = np.array(population)
        self.fitnessValuesPopulation = np.array(fitnessValuesPopulation)
        self.populationSize = len(population)

    def diffevol_reduction_ClustersRepresentatives(self) -> dict:
        """
            Method to get clusters representatives of each cluster 

            Return a dict with cluster number and representatives 
        """
        pass

    def diffevol_reduction_UnpackClustersRepresentatives(self,clustersRepresentativeIndividuals) -> tuple:
        """
            Method to transform representative individuals into lits of cluster labels, population and fitness values
            -- clustersRepresentativeIndividuals : Dict with cluster labels and individuals with their fitness values
        """
        populationClusters , population , fitnessValuesPopulation = [] , [] , []
        for cluster , representatives in clustersRepresentativeIndividuals.items():
            for representative in representatives:
                individual , fitnessValue = representative
                populationClusters.append(cluster)
                population.append(individual)
                fitnessValuesPopulation.append(fitnessValue)
        return populationClusters , population , fitnessValuesPopulation