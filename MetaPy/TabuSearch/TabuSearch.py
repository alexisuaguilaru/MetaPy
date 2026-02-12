import numpy as np
from typing import Callable , Any

class TabuSearch:
    def __init__(
            self,
            ObjectiveFunction: Callable[[np.ndarray],float],
            InitializeSolution: Callable[[],np.ndarray],
            GenerateNeighborhood: Callable[[np.ndarray,list],list[np.ndarray]],
            TabuRepresentation: Callable[[np.ndarray,np.ndarray],Any],
        ):
        """
        """

        self.ObjectiveFunction = ObjectiveFunction
        self.InitializeSolution = InitializeSolution
        self.GenerateNeighborhood = GenerateNeighborhood
        self.TabuRepresentation = TabuRepresentation

    def __call__(
            self,
            Iterations: int,
            TabuTime: int,
        ):
        """
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

            self.UpdataTabuList(TabuList,TabuTime,PreviousSolution,CurrentSolution)

        return OptimalIndividual , Snapshots

    def ReduceNeighborhoodOperation(
            self,
            TabuList: list[np.ndarray],
            CurrentNeighborhood: list[np.ndarray],
        ) -> list[np.ndarray]:
        """
        """

        reduced_neighborhood = []
        tabu_solutions = list(map(lambda tabu: tabu[1],TabuList))
        for neighborhood_solution in CurrentNeighborhood:
            if all((np.any(neighborhood_solution != tabu) for tabu in tabu_solutions)):
                reduced_neighborhood.append(neighborhood_solution)

        return reduced_neighborhood
    
    def UpdataTabuList(
            self,
            TabuList: list,
            TabuTime: int,
            PreviousSolution: np.ndarray,
            CurrentSolution: np.ndarray,
        ) -> None:
        """
        """

        for tabu in TabuList:
            tabu[-1] -= 1
        TabuList.append([self.TabuRepresentation(PreviousSolution,CurrentSolution),CurrentSolution,TabuTime])