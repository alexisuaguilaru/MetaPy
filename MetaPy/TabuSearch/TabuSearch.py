import numpy as np

from ..Base import MetaheuristicOptimizer , MetaheuristicSimulations

from typing import Callable , Any

class TabuSearchOptimizer(MetaheuristicOptimizer,MetaheuristicSimulations):
    def __init__(
            self,
            ObjectiveFunction: Callable[[np.ndarray],float],
            InitializeSolution: Callable[[],np.ndarray],
            GenerateNeighborhood: Callable[[np.ndarray,list],list[np.ndarray]],
            TabuRepresentation: Callable[[np.ndarray,np.ndarray],Any],
        ):
        """
        Class for implementation of Tabu Search based 
        on Fred Glover's version. Search the minimum 
        value of `ObjectiveFunction`. 
        
        Parameters
        ----------
        ObjectiveFunction: Callable[[np.ndarray],float]
            Function to optimize. Takes a individual of shape `(Dim,)` and returns its fitness value

        InitializeSolution: Callable[[],np.ndarray]
            Function to initialize a feasible solution/individual

        GenerateNeighborhood: Callable[[np.ndarray,list],list[np.ndarray]]
            Function to generate the neighborhood of a solution

        TabuRepresentation: Callable[[np.ndarray,np.ndarray],Any]
            Function to get the tabu representation of a solution
        """

        self.ObjectiveFunction = ObjectiveFunction
        self.InitializeSolution = InitializeSolution
        self.GenerateNeighborhood = GenerateNeighborhood
        self.TabuRepresentation = TabuRepresentation

    def __call__(
            self,
            Iterations: int,
            TabuTime: int,
        ) -> tuple[np.ndarray,list[float]]:
        """
        Method to search the closest optimal solution from a 
        given solution for the objective function.  Return 
        optimal solution and a list of optimal values at 
        each iteration.

        Parameters
        ----------
        Iterations: int
            Number of iterations/generations for the search

        TabuTime: int
            Number of iterations to mark a solution as tabu

        Returns
        -------
        OptimalIndividual: np.ndarray
            Best solution/individual that was founded

        Snapshots: list[float] 
            List of the optimal values at each iteration/generation
        """

        CurrentSolution = self.InitializeSolution()
        CurrentFitnessValue = self.ObjectiveFunction(CurrentSolution)

        OptimalIndividual = CurrentSolution.copy()
        OptimalFitnessValue = CurrentFitnessValue

        TabuList = []

        Snapshots = []
        Snapshots.append(OptimalFitnessValue)

        for iteration in range(Iterations):
            current_neighborhood = self.GenerateNeighborhood(CurrentSolution,TabuList)
            reduced_neighborhood = self.ReduceNeighborhoodOperation(TabuList,current_neighborhood)

            fitness_values = list(map(self.ObjectiveFunction,reduced_neighborhood))
            index_best_neighbor = np.argmin(fitness_values)

            PreviousSolution = CurrentSolution.copy()
            CurrentSolution = reduced_neighborhood[index_best_neighbor]
            CurrentFitnessValue = fitness_values[index_best_neighbor]

            if CurrentFitnessValue < OptimalFitnessValue:
                OptimalIndividual = CurrentSolution.copy()
                OptimalFitnessValue = CurrentFitnessValue

            Snapshots.append(OptimalFitnessValue)

            self.UpdateTabuList(TabuList,TabuTime,PreviousSolution,CurrentSolution)

        return OptimalIndividual , Snapshots

    def FineTuningHyperparameters(
            self,
            Iterations: int,
            Hyperparameters: dict[str,tuple[str,tuple]] = {
                    'TabuTime': ('int',(1,5))
                },
            NumTrials: int = 10,
            NumJobs: int = 1,
        ) -> dict[str,Any]:

        return super().FineTuningHyperparameters(Iterations,Hyperparameters,NumTrials,NumJobs)

    def ReduceNeighborhoodOperation(
            self,
            TabuList: list[np.ndarray],
            CurrentNeighborhood: list[np.ndarray],
        ) -> list[np.ndarray]:
        """
        Method to reduce the current neighborhood 
        based on the tabu list to remove invalid/tabu solutions.

        Parameters
        ----------
        TabuList: list[np.ndarray]
            A list with the tabu solutions and their representations and times

        CurrentNeighborhood: list[np.ndarray]
            A list with the neighborhood to reduce by the tabu list

        Return
        ------
        ReducedNeighborhood: list[np.ndarray]
            A list with the valid neighborhood based on the tabu list
        """

        reduced_neighborhood = []
        tabu_solutions = list(map(lambda tabu: tabu[1],TabuList))
        for neighborhood_solution in CurrentNeighborhood:
            if all((np.any(neighborhood_solution != tabu) for tabu in tabu_solutions)):
                reduced_neighborhood.append(neighborhood_solution)

        return reduced_neighborhood
    
    def UpdateTabuList(
            self,
            TabuList: list,
            TabuTime: int,
            PreviousSolution: np.ndarray,
            CurrentSolution: np.ndarray,
        ) -> None:
        """
        Method to update the times of each tabu solution and 
        to add a new tabu solution.

        Parameters
        ----------
        TabuList: list
            A list with the tabu solutions and their representations and times
        
        TabuTime: int
            Max time to mark a solution as tabu
        
        PreviousSolution: np.ndarray
            Solution already marked as tabu used to generate a new solution
        
        CurrentSolution: np.ndarray
            New solution to mark it as tabu and generated by the previous solution
        """

        for index , tabu in enumerate(TabuList):
            tabu[-1] -= 1
            if tabu[-1] == 0:
                del TabuList[index]
        TabuList.append([self.TabuRepresentation(PreviousSolution,CurrentSolution),CurrentSolution,TabuTime])