from DifferentialEvolution import DifferentialEvolution

class DifferentialEvolution_ReductionPopulation(DifferentialEvolution):
    def __init__(self, objectiveFunction, initializeIndividual):
        """
            Class for Differential Evolution Metaheuristic Variant which use
            K-Means Algorithm for improving solutions
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals

            Based on DE/rand/1/bin
        """
        super().__init__(objectiveFunction, initializeIndividual)
    
    def __call__(self, populationSize:int , scalingFactor:float , crossoverRate:float) -> tuple:
        """
            Method to search optimal solution for objective function
            -- populationSize : Parameter NP. Size of solutions population
            -- scalingFactor : Parameter F. Scaling factor for difference vector 
            -- crossoverRate : Parameter Cr. Crossover rate for crossover operation
            No iteration parameter is required because ends when the population size is equal to 1
            
            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the population at each iteration
        """
        self.populationSize = populationSize
        self.scalingFactor = scalingFactor
        self.crossoverRate = crossoverRate
        self.SnapshotsSaved = []
        self.diffevol_InitializePopulation()
        self.optimalIndividual , self.optimalValue = self.diffevol_BestOptimalFound()
        iteration = 0
        while self.populationSize > 3:
            if iteration%10 == 0 and iteration != 0:
                self.diffevol_redpop_ReductionPopulation()
                self.diffevol_redpop_IncreasePopulation()
            self.diffevol_IterativeSearch(iteration)
            iteration += 1
        return self.optimalIndividual , self.SnapshotsSaved

    def diffevol_redpop_ReductionPopulation(self):
        """
            Method to reduce population size using k-means
        """
        from math import sqrt , ceil
        from sklearn.cluster import k_means
        numClusters = ceil(sqrt(self.populationSize))
        populationLabels = k_means(self.population,numClusters)[1]
        self.diffevol_redpop_ReinitializePopulation(populationLabels)
        self.populationSize = numClusters

    def diffevol_redpop_ReinitializePopulation(self,populationLabels):
        """
            Method to reinitialize population with the best individuals and theirs fitness values
        """
        from collections import defaultdict
        import numpy as np
        populationIndividuals = defaultdict(int)
        for labelCluster , individual , individualFitnessValue in zip(populationLabels,self.population,self.fitnessValuesPopulation):
            if populationIndividuals[labelCluster] == 0:
                populationIndividuals[labelCluster] = (individual,individualFitnessValue)
            else:
                if individualFitnessValue < populationIndividuals[labelCluster][1]:
                    populationIndividuals[labelCluster] = (individual,individualFitnessValue)
        population , fitnessValuesPopulation = [] , []
        [(population.append(individual),fitnessValuesPopulation.append(fitnessValue))  for individual,fitnessValue in populationIndividuals.values()]
        self.population = np.array(population)
        self.fitnessValuesPopulation = np.array(fitnessValuesPopulation)

    def diffevol_redpop_IncreasePopulation(self):
        """
            Method to generate new solutions based on best solutions
            at each cluster
        """
        from math import sqrt , ceil
        from random import randint
        import numpy as np
        numClusters = self.populationSize
        numIncreasePopulation = randint(0,numClusters)*numClusters
        newIndividuals , newIndividualsFitnessValues = [] , []
        for _ in range(numIncreasePopulation):
            individualGenerated = self.diffevol_redpop_CreateNewIndividual()
            newIndividuals.append(individualGenerated)
            newIndividualsFitnessValues.append(self.objectiveFunction(individualGenerated))
        self.populationSize += numIncreasePopulation
        print(self.population.shape,(len(newIndividuals),len(newIndividuals[0])))
        self.population = np.concatenate((self.population,newIndividuals))
        self.fitnessValuesPopulation = np.concatenate((self.fitnessValuesPopulation,newIndividualsFitnessValues))
        self.optimalIndividual , self.optimalValue = self.diffevol_BestOptimalFound()

    def diffevol_redpop_CreateNewIndividual(self):
        """
            Method to generate an individual solution based 
            on best solutions at each cluster
        """
        from random import randint
        mutatedIndividual = self.diffevol_MutationOperation()
        indexBaseIndividual = randint(0,self.populationSize-1)
        crossoverIndividual = self.diffevol_CrossoverOperation(indexBaseIndividual,mutatedIndividual)
        return crossoverIndividual