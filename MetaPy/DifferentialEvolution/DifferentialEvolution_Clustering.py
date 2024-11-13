from copy import deepcopy

from .DifferentialEvolution import DifferentialEvolution

class DifferentialEvolution_Clustering(DifferentialEvolution):
    def __init__(self, objectiveFunction, initializeIndividual, clusteringAlgorithm):
        """
            Class for Differential Evolution Metaheuristic with Population Clustering 
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals
            -- clusteringAlgorithm : Algorithm to execute clustering from Scikit Learn

            Based on DE/rand/1/bin
        """    
        super().__init__(objectiveFunction, initializeIndividual)
        self.clusteringAlgorithm = clusteringAlgorithm
    
    def __call__(self, iterations: int, populationSize: int, scalingFactor: float, crossoverRate: float,clusteringAlgorithm_kw: dict={}):
        """
            Method to search optimal solution for objective function
            -- iterations : Amount of iterations
            -- populationSize : Parameter NP. Size of solutions population
            -- scalingFactor : Parameter F. Scaling factor for difference vector 
            -- crossoverRate : Parameter Cr. Crossover rate for crossover operation
            -- clusteringAlgorithm_kw : Keywords for clustering function

            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the population at each iteration
        """
        self.clusteringAlgorithm_kw = clusteringAlgorithm_kw
        return super().__call__(iterations, populationSize, scalingFactor, crossoverRate)

    def diffevol_IterativeSearch(self, iteration: int) -> None:
        returnClustering = self.clusteringAlgorithm(self.population,**self.clusteringAlgorithm_kw)
        if type(returnClustering) == tuple:
            self.populationClusters = returnClustering[1]
        else:
            self.populationClusters = returnClustering 
        super().diffevol_IterativeSearch(iteration)

    def diffevol_SnapshotPopulation(self, iteration: int):
        """
            Method to save a snapshot of the population at iteration-st
            -- iteration : Number of iteration 
        """
        self.SnapshotsSaved.append((iteration,deepcopy(self.population),self.callingObjectiveFunction,deepcopy(self.populationClusters),self.optimalValue))