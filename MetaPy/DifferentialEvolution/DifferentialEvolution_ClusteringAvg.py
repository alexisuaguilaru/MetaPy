from collections import defaultdict
import numpy as np

from .DifferentialEvolution_ClusteringOperation import DifferentialEvolution_ClusteringOperation

class DifferentialEvolution_ClusteringAvg(DifferentialEvolution_ClusteringOperation):
    def __init__(self, objectiveFunction, initializeIndividual, clusteringAlgorithm):
        """
        Class for Differential Evolution Metaheuristic with Local Optimization
        based on Clustering and Average Solutions
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals
            -- clusteringAlgorithm : Algorithm to execute clustering from Scikit Learn

            Based on DE/rand/1/bin
        """
        super().__init__(objectiveFunction, initializeIndividual, clusteringAlgorithm)

    def diffevol_clust_ClustersRepresentatives(self) -> dict:
        """
            Method to get clusters representatives on based of average solution of each cluster  
        """
        clustersRepresentativeIndividuals = defaultdict(list)
        for clusterBelongs , individual in zip(self.populationClusters,self.population):
            if clusterBelongs == -1:
                continue
            else:
                clustersRepresentativeIndividuals[clusterBelongs].append(individual)
        for clusterBelongs , individuals in clustersRepresentativeIndividuals.items():
            averageIndividual = np.mean(individuals, axis=0)
            fitnessValueIndividual = self.objectiveFunction(averageIndividual)
            clustersRepresentativeIndividuals[clusterBelongs] = (averageIndividual,fitnessValueIndividual)
        return clustersRepresentativeIndividuals