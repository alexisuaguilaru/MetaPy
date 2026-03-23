import numpy as np

from ..Base import MetaheuristicOptimizer , MetaheuristicSimulations

from typing import Callable , Any

class EvolutiveProgrammingOptimizer(MetaheuristicOptimizer,MetaheuristicSimulations):
    def __init__(
            self,
            ObjectiveFunction: Callable[[np.ndarray],float],
            InitializePopulation: Callable[[int],tuple[np.ndarray,np.ndarray]],
        ):
        """
        Class for implementation of Evolutive Programming
        based on (mu+mu). Search the minimum value 
        of `ObjectiveFunction`.
        
        Parameters
        ----------
        ObjectiveFunction: Callable[[np.ndarray],float]
            Function to optimize. Takes a solution/individual of shape `(Dim,)` and returns its fitness value

        InitializePopulation: Callable[[int],np.ndarray]
            Function to create a population of solutions and their deviations (sigmas). Return two `np.ndarray` objects both of shape `(Size,Dim)`
        """

        self.ObjectiveFunction = ObjectiveFunction
        self.InitializePopulation = InitializePopulation

    def __call__(
            self,
            Iterations: int,
            PopulationSize: int,
            MinStd: float,
            MutationFactor: float,
        ) -> tuple[np.ndarray,list[float]]:
        """
        Method for searching optimal solution for a give objective 
        function. Return the best optimal solution and a list of 
        optimal values at each iteration.

        Parameters
        ----------
        Iterations: int
            Number of iterations/generations for the search

        PopulationSize: int 
            Size of population of solutions and deviations (number of solutions)

        MinStd: float
            Minimum allowed standard deviation

        MutationFactor: float
            Scaling factor to mutate the sigma values

        Returns
        -------
        OptimalIndividual: np.ndarray
            Best solution/individual that was founded

        Snapshots: list[float] 
            List of the optimal values at each iteration/generation
        """

        self.PopulationSize = PopulationSize
        self.MinStd = MinStd
        self.MutationFactor = MutationFactor

        self.InitializeOptimization()
        self.OptimalIndividual , self.OptimalValue = self.BestOptimalIndividual()

        self.Snapshots = []
        self.Snapshots.append(self.OptimalValue)

        self.FindOptimal(Iterations)

        return self.OptimalIndividual , self.Snapshots
    
    def FineTuningHyperparameters(
            self,
            Iterations: int,
            Hyperparameters: dict[str,tuple[str,tuple]] = {
                    'PopulationSize': ('int',(1,100)),
                    'MinStd': ('float',(1e-15,0.5)),
                    'MutationFactor': ('float',(1e-15,2)),
                },
            NumTrials: int = 10,
            NumJobs: int = 1,
        ) -> dict[str,Any]:

        return super().FineTuningHyperparameters(Iterations,Hyperparameters,NumTrials,NumJobs)
    
    def InitializeOptimization(
            self,
        ) -> None:
        """
        Method for initializing `PopulationIndividuals` and 
        `FitnessValuesPopulation` attributes.
        """

        self.PopulationIndividuals , self.PopulationDeviations = self.InitializePopulation(self.PopulationSize)
        self.FitnessValuesPopulation = np.apply_along_axis(self.ObjectiveFunction,1,self.PopulationIndividuals)

        self.PopulationIndexes = np.arange(self.PopulationSize)

    def BestOptimalIndividual(
            self
        ) -> tuple[np.ndarray,float]:
        """
        Method for finding the best optimal 
        individual at the current `PopulationIndividuals`.
            
        Returns
        -------
        BestOptimalSolution: np.ndarray
            Best optimal solution/individual in the `PopulationIndividuals`
        
        BestOptimalValue: float
            Best optimal (minimum) value in the `PopulationIndividuals`
        """

        IndexOptimalIndividual = np.argmin(self.FitnessValuesPopulation)
        return self.PopulationIndividuals[IndexOptimalIndividual] , self.FitnessValuesPopulation[IndexOptimalIndividual]
    
    def FindOptimal(
            self,
            Iterations: int,
        ) -> None:
        """
        Method for finding the optimal solution 
        for the `ObjectiveFunction`.

        Parameter
        ---------
        Iterations: int
            Number of iterations/generations for the search
        """

        for iteration in range(Iterations):
            self.MutationOperation()

            self.SelectionOperation()

            self.Snapshots.append(self.OptimalValue)

    def MutationOperation(
            self,
        ) -> None:
        """
        Method for applying Evolutive Programming 
        Mutation Operation to the `PopulationIndividuals`.
        """

        Mutations = 1 + self.MutationFactor*self.RandNormalVector()
        self.MutatedDeviations = np.clip(self.PopulationDeviations.copy()*Mutations[:,None],self.MinStd,None)
        
        self.MutatedIndividuals = self.PopulationIndividuals.copy() + self.MutatedDeviations*self.RandNormalVector()[:,None]

        self.FitnessValuesMutated = np.apply_along_axis(self.ObjectiveFunction,1,self.MutatedIndividuals)

    def SelectionOperation(
            self,
        ) -> None:
        """
        Method for applying Evolutive Programming 
        Selection Operation to the `PopulationIndividuals` 
        and `MutatedIndividuals` with (\mu+\mu) strategy
        """

        TotalPopulation = np.concat([self.PopulationIndividuals,self.MutatedIndividuals],axis=0)
        TotalDeviations = np.concat([self.PopulationDeviations,self.MutatedDeviations],axis=0)
        
        TotalFitnessValues = np.concat([self.FitnessValuesPopulation,self.FitnessValuesMutated],axis=0)
        BestIndividuals = TotalFitnessValues.argsort()[:self.PopulationSize]

        self.PopulationIndividuals = TotalPopulation[BestIndividuals]
        self.PopulationDeviations = TotalDeviations[BestIndividuals]

        self.OptimalIndividual , self.OptimalValue = self.BestOptimalIndividual()

    def RandNormalVector(
            self
        ) -> np.ndarray:
        """
        Method to generate a normal values 
        vector of size `PopulationSize`
        """

        return np.random.normal(0,1,self.PopulationSize)