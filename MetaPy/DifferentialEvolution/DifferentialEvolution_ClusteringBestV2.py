from collections import defaultdict

from .DifferentialEvolution_ClusteringOperation import DifferentialEvolution_ClusteringOperation

class DifferentialEvolution_ClusteringBestV2(DifferentialEvolution_ClusteringOperation):
    def __init__(self, objectiveFunction, initializeIndividual, clusteringAlgorithm):
        """
        Class for Differential Evolution Metaheuristic with Local Optimization
        based on Clustering and Best Solutions of all clusters
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals
            -- clusteringAlgorithm : Algorithm to execute clustering from Scikit Learn

            Based on DE/rand/1/bin
        """
        super().__init__(objectiveFunction, initializeIndividual, clusteringAlgorithm)

    def diffevol_clust_ClustersRepresentatives(self) -> dict:
        """
            Method to get clusters representatives on based of best optimal solutions of each cluster  
        """
        clustersRepresentativeIndividuals = defaultdict(int)
        for clusterBelongs , individual , fitnessValue in zip(self.populationClusters,self.population,self.fitnessValuesPopulation):
            if clustersRepresentativeIndividuals[clusterBelongs] == 0:
                clustersRepresentativeIndividuals[clusterBelongs] = (individual,fitnessValue)
            elif fitnessValue < clustersRepresentativeIndividuals[clusterBelongs][-1]:
                clustersRepresentativeIndividuals[clusterBelongs] = (individual,fitnessValue)
        return clustersRepresentativeIndividuals