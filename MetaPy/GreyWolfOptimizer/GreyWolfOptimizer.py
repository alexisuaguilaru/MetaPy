class GreyWolfOptimizer:
    def __init__(self,objectiveFunction,initializeIndividual):
        """
            Class for Grey Wolf Optimizer Metaheuristic
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals

            Based on GWO
        """
        self.objectiveFunction = objectiveFunction
        self.initializeIndividual = initializeIndividual
    
    def __call__(self, iterations:int ,populationSize:int):
        import numpy as np
        self.populationSize = populationSize
        self.GWO_initializePopulation()
        vector_a = np.full(len(self.population[0]),2,dtype=np.float64)
        randNumGen = np.random.default_rng()
        for iteration in range(iteration):
            self.GWO_updateVectorA(randNumGen,vector_a)
            self.GWO_updateVectorC(randNumGen)
            solutionAlpha , solutionBetta , solutionDelta = self.GWO_solutionsAlphaBettaDelta()
            for indexSolution in range(self.populationSize):
                self.GWO_updateIndexSolution(indexSolution,solutionAlpha,solutionBetta,solutionDelta)
            self.GWO_decreaseVector_a(vector_a,iteration,iterations)
            
    def GWO_initializePopulation(self):
        """
            Method to initialize population and fitnessValuesPopulation attributes  
        """
        import numpy as np
        self.population = np.array([self.initializeIndividual() for _ in range(self.populationSize)])
        self.fitnessValuesPopulation = np.array([self.objectiveFunction(individual) for individual in self.population])

    def GWO_solutionsAlphaBettaDelta(self):
        """
            Method to initialize alpha (best), betta (second) and delta (third) solutions
        """
        import numpy as np
        indexAlpha , indexBetta , indexDelta = [0,0,0]
        for indexSolution , fitnessValueSolution in enumerate(self.fitnessValuesPopulation):
            if fitnessValueSolution >= self.fitnessValuesPopulation[indexAlpha]:
                indexAlpha , indexBetta , indexDelta = indexSolution , indexAlpha , indexBetta
            elif fitnessValueSolution >= self.fitnessValuesPopulation[indexBetta]:
                indexBetta , indexDelta = indexSolution , indexBetta
            elif fitnessValueSolution > self.fitnessValuesPopulation[indexDelta]:
                indexDelta = indexSolution
        return self.population[indexAlpha] , self.population[indexBetta] , self.population[indexDelta]
    
    def GWO_updateVectorA(self,randNumGen,vector_a):
        """
            Method to update vector A
        """
        randomVector = randNumGen.random((len(self.population[0])))
        self.vectorA = 2*vector_a*randomVector - vector_a

    def GWO_updateVectorC(self,randNumGen):
        """
            Method to update vector C
        """
        randomVector = randNumGen.random((len(self.population[0])))
        self.vectorC = 2*randomVector

    def GWO_updateIndexSolution(self,indexSolution,solutionAlpha,solutionBetta,solutionDelta):
        pass
        
    def GWO_decreaseVector_a(self,vector_a,iteration,iterations):
        """
            Method to linearly decrease the vector a's entries 
        """
        import numpy as np
        if iteration == iterations-2:
                vector_a = np.full(len(self.population[0]),0,dtype=np.float64)
        else:
            vector_a = vector_a - 2/(iterations-1)