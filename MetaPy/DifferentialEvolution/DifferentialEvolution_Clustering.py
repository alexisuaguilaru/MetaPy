from DifferentialEvolution import DifferentialEvolution

class DifferentialEvolution_Clustering(DifferentialEvolution):
    def __init__(self, objectiveFunction, initializeIndividual, clusteringAlgorithm):
        """
            Class for Differential Evolution Metaheuristic with Local Optimization 
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals
            -- clusteringAlgorithm : Algorithm to execute clustering from Scikit Learn

            Based on DE/rand/1/bin
        """
        self.clusteringAlgorithm = clusteringAlgorithm
        super().__init__(objectiveFunction, initializeIndividual)
    
    def diffevol_InitializePopulation(self) -> None:
        """
            Method to initialize population, fitnessValuesPopulation and  attributes  
        """
        import numpy as np
        super().diffevol_InitializePopulation()
        self.populationClusters = self.clusteringAlgorithm.fit(self.population)

    def diffevol_SnapshotPopulation(self, iteration: int):
        """
            Method to save a snapshot of the population at iteration-st
            -- iteration : Number of iteration 
        """
        from copy import deepcopy
        self.SnapshotsSaved.append((iteration,deepcopy(self.population),deepcopy(self.populationClusters)))