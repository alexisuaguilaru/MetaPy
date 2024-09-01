from DifferentialEvolution_Clustering import DifferentialEvolution_Clustering

class DifferentialEvolution_ClusteringOperation(DifferentialEvolution_Clustering):
    def __init__(self, objectiveFunction, initializeIndividual, clusteringAlgorithm):
        """
        Base Class for Differential Evolution Metaheuristic with Local Optimization
        based on Clustering and Another Operation on Solutions to get Representatives
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals
            -- clusteringAlgorithm : Algorithm to execute clustering from Scikit Learn

            Based on DE/rand/1/bin
        """
        super().__init__(objectiveFunction, initializeIndividual, clusteringAlgorithm)
    
    def __call__(self, iterations:int, populationSize:int, scalingFactor:float, crossoverRate:float, applyClusteringAt:int = 63 , clusteringAlgorithm_kw:dict = {}):
        """
            Method to search optimal solution for objective function
            -- iterations : Amount of iterations
            -- populationSize : Parameter NP. Size of solutions population
            -- scalingFactor : Parameter F. Scaling factor for difference vector 
            -- crossoverRate : Parameter Cr. Crossover rate for crossover operation
            -- applyClusteringAt : Parameter to control in which iteration apply clustering on solutions population
            -- clusteringAlgorithm_kw : Keywords for clustering function

            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the population at each iteration
        """
        self.applyClusteringAt = applyClusteringAt
        return super().__call__(iterations, populationSize, scalingFactor, crossoverRate, clusteringAlgorithm_kw)
    
    def diffevol_IterativeSearch(self, iteration: int):
        if iteration == self.applyClusteringAt:
            self.diffevol_clust_InitializePopulationRepresentatives()
            self.optimalIndividual , self.optimalValue = self.diffevol_BestOptimalFound()
        super().diffevol_IterativeSearch(iteration)

    def diffevol_clust_InitializePopulationRepresentatives(self):
        """
            Method to reinitialize the population with the representatives of each cluster
        """
        import numpy as np
        clustersRepresentativeIndividuals = self.diffevol_clust_ClustersRepresentatives()
        populationClusters , population , fitnessValuesPopulation = self.diffevol_clust_UnpackClustersRepresentatives(clustersRepresentativeIndividuals)
        self.populationClusters = np.array(populationClusters) 
        self.population = np.array(population)
        self.fitnessValuesPopulation = np.array(fitnessValuesPopulation)
        self.populationSize = len(population)

    def diffevol_clust_ClustersRepresentatives(self) -> dict:
        """
            Method to get clusters representatives on based of best optimal solutions of each cluster  
        """
        pass

    def diffevol_clust_UnpackClustersRepresentatives(self,clustersRepresentativeIndividuals) -> tuple:
        """
            Method to transform representative individuals into lits of cluster labels, population and fitness values
            -- clustersRepresentativeIndividuals : Dict with cluster labels and individuals with their fitness values
        """
        populationClusters , population , fitnessValuesPopulation = [] , [] , []
        for cluster , solution in clustersRepresentativeIndividuals.items():
            individual , fitnessValue = solution
            populationClusters.append(cluster)
            population.append(individual)
            fitnessValuesPopulation.append(fitnessValue)
        return populationClusters , population , fitnessValuesPopulation