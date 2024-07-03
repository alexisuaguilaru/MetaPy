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
        self.halfIterations = iterations//2
        self.clusteringAlgorithm_kw = clusteringAlgorithm_kw
        return super().__call__(iterations, populationSize, scalingFactor, crossoverRate)

    def diffevol_IterativeSearch(self, iteration: int) -> None:
        returnClustering = self.clusteringAlgorithm(self.population,**self.clusteringAlgorithm_kw)
        if type(returnClustering) == tuple:
            self.populationClusters = returnClustering[1]
        else:
            self.populationClusters = returnClustering 
        if iteration == self.halfIterations:
            self.diffevol_clust_InitializePopulationRepresentatives()
        super().diffevol_IterativeSearch(iteration)

    def diffevol_SnapshotPopulation(self, iteration: int):
        """
            Method to save a snapshot of the population at iteration-st
            -- iteration : Number of iteration 
        """
        from copy import deepcopy
        self.SnapshotsSaved.append((iteration,deepcopy(self.population),deepcopy(self.populationClusters),self.optimalValue))

    def diffevol_clust_InitializePopulationRepresentatives(self):
        """
            Method to reinitialize the population with the representatives of each cluster
        """
        from collections import defaultdict
        import numpy as np
        tableClustersIndividuals = defaultdict(int)
        for clusterBelongs , individual , fitnessValue in zip(self.populationClusters,self.population,self.fitnessValuesPopulation):
            if clusterBelongs == -1:
                continue
            else:
                if tableClustersIndividuals[clusterBelongs] == 0:
                    tableClustersIndividuals[clusterBelongs] = (individual,fitnessValue)
                elif fitnessValue < tableClustersIndividuals[clusterBelongs][-1]:
                    tableClustersIndividuals[clusterBelongs] = (individual,fitnessValue)
        populationClusters , population , fitnessValuesPopulation = [] , [] , []
        for cluster , solution in tableClustersIndividuals.items():
            individual , fitnessValue = solution
            populationClusters.append(cluster)
            population.append(individual)
            fitnessValuesPopulation.append(fitnessValue)
        self.populationClusters = np.array(populationClusters) 
        self.population = np.array(population)
        self.fitnessValuesPopulation = np.array(fitnessValuesPopulation)
        self.populationSize = len(population)