from DifferentialEvolution_Clustering import DifferentialEvolution_Clustering

class DifferentialEvolution_ClusteringBest(DifferentialEvolution_Clustering):
    def __init__(self, objectiveFunction, initializeIndividual, clusteringAlgorithm):
        """
        Class for Differential Evolution Metaheuristic with Local Optimization
        based on Clustering and Best Solutions
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals
            -- clusteringAlgorithm : Algorithm to execute clustering from Scikit Learn

            Based on DE/rand/1/bin
        """
        super().__init__(objectiveFunction, initializeIndividual, clusteringAlgorithm)

    def __call__(self, iterations: int, populationSize: int, scalingFactor: float, crossoverRate: float, clusteringAlgorithm_kw: dict = ...):
        return super().__call__(iterations, populationSize, scalingFactor, crossoverRate, clusteringAlgorithm_kw)

    def diffevol_IterativeSearch(self, iteration: int) -> None:
        if iteration == 63:
            self.diffevol_clust_InitializePopulationRepresentatives()
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
        from collections import defaultdict
        clustersRepresentativeIndividuals = defaultdict(int)
        for clusterBelongs , individual , fitnessValue in zip(self.populationClusters,self.population,self.fitnessValuesPopulation):
            if clusterBelongs == -1:
                continue
            else:
                if clustersRepresentativeIndividuals[clusterBelongs] == 0:
                    clustersRepresentativeIndividuals[clusterBelongs] = (individual,fitnessValue)
                elif fitnessValue < clustersRepresentativeIndividuals[clusterBelongs][-1]:
                    clustersRepresentativeIndividuals[clusterBelongs] = (individual,fitnessValue)
        return clustersRepresentativeIndividuals

    def diffevol_clust_UnpackClustersRepresentatives(clustersRepresentativeIndividuals) -> tuple:
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