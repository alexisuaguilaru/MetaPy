from sklearn import cluster
import math
import numpy as np
from collections import defaultdict
import random 

from .DifferentialEvolution_Reduction import DifferentialEvolution_Reduction

class DifferentialEvolution_Reduction_RandomSample(DifferentialEvolution_Reduction):
    def __init__(self, objectiveFunction, initializeIndividual):
        """
        Base Class for Differential Evolution Metaheuristic with Population Reduction 
        based on K Means Algorithm with Random Sample of Representatives
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals

            Based on DE/rand/1/bin
        """
        clusteringAlgorithm = cluster.k_means
        super().__init__(objectiveFunction, initializeIndividual, clusteringAlgorithm)

    def __call__(self, iterations, populationSize, scalingFactor, crossoverRate):
        """
            Method to search optimal solution for objective function
            -- iterations : Amount of iterations
            -- populationSize : Parameter NP. Size of solutions population
            -- scalingFactor : Parameter F. Scaling factor for difference vector 
            -- crossoverRate : Parameter Cr. Crossover rate for crossover operation

            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the population at each iteration
        """
        self.clusteringAlgorithm = self.baseClusteringAlgorithm
        clusteringAlgorithm_kw = {'n_clusters':math.floor(math.sqrt(populationSize))}
        return super().__call__(iterations, populationSize, scalingFactor, crossoverRate, clusteringAlgorithm_kw)

    def diffevol_reduction_ApplyPopulationReduction(self,iteration:int) -> bool:
        """
            Method to determine whether it is necessary apply a 
            reduction of the population based on if iteration 
            is half of the total of iterations.

            -- iteration : Number of iteration  

            Return a boolean value based on the criteria.
        """
        return iteration == self.totalIterations//2
    
    def diffevol_reduction_ClustersRepresentatives(self) -> dict:
        """
            Method to get clusters representatives of each cluster 

            Return a dict with cluster number and representatives 
        """
        __populationFitness = np.concat([self.population,np.reshape(self.fitnessValuesPopulation,shape=(self.populationSize,1))],axis=1)
        __poulationClusters = self.clusteringAlgorithm(__populationFitness,**self.clusteringAlgorithm_kw)[1]
        clustersRepresentativeIndividuals = defaultdict(list)
        for clusterBelongs , individual_fitness in zip(__poulationClusters,__populationFitness):
            individual , fitnessValue = individual_fitness[:-1] , individual_fitness[-1]
            clustersRepresentativeIndividuals[clusterBelongs].append((individual,fitnessValue))
        for clusterBelongs , individualsCluster in clustersRepresentativeIndividuals.items():
            if len(clustersRepresentativeIndividuals[clusterBelongs]) > 3:
                clustersRepresentativeIndividuals[clusterBelongs] = random.sample(individualsCluster,3)
        return clustersRepresentativeIndividuals