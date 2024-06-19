from DifferentialEvolution import DifferentialEvolution

class DifferentialEvolution_Clustering(DifferentialEvolution):
    def __init__(self, objectiveFunction, initializeIndividual, clusteringAlgorithm,clusteringAlgorithm_kw:dict={}):
        """
            Class for Differential Evolution Metaheuristic with Local Optimization 
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals
            -- clusteringAlgorithm : Algorithm to execute clustering from Scikit Learn
            -- clusteringAlgorithm_kw : Keywords for clustering function

            Based on DE/rand/1/bin
        """    
        super().__init__(objectiveFunction, initializeIndividual)
        self.clusteringAlgorithm = clusteringAlgorithm
        self.clusteringAlgorithm_kw = clusteringAlgorithm_kw
    
    def diffevol_InitializePopulation(self) -> None:
        """
            Method to initialize population, fitnessValuesPopulation and  attributes  
        """
        import numpy as np
        from copy import deepcopy
        super().diffevol_InitializePopulation()
        returnClustering = self.clusteringAlgorithm(deepcopy(self.population),**self.clusteringAlgorithm_kw)
        if type(returnClustering) == tuple:
            self.populationClusters = returnClustering[1]
        else:
            self.populationClusters = returnClustering

    def diffevol_SnapshotPopulation(self, iteration: int):
        """
            Method to save a snapshot of the population at iteration-st
            -- iteration : Number of iteration 
        """
        from copy import deepcopy
        self.SnapshotsSaved.append((iteration,deepcopy(self.population),deepcopy(self.populationClusters)))