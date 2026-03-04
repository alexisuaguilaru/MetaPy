from random import choice
import numpy as np

from ..Base import MetaheuristicOptimizer , MetaheuristicSimulations

from typing import Callable , Any

class SimulatedAnnealingOptimizer(MetaheuristicSimulations,MetaheuristicOptimizer):
    def __init__(
            self,
            ObjectiveFunction: Callable[[np.ndarray],float],
            InitializeSolution: Callable[[],np.ndarray],
            GenerateNeighborhood: Callable[[np.ndarray],list[np.ndarray]],
            TemperatureSchedule: Callable[[float,float],list[float]],
        ):
        """
        Class for implementation of Simulated Annealing 
        based on Kirkpatrick, Gelatt and Vecchi's version. 
        Search the minimum value of `ObjectiveFunction`. 

        Parameters
        ----------
        ObjectiveFunction: Callable[[np.ndarray],float]
            Function to optimize. Takes a individual of shape `(Dim,)` and returns its fitness value

        InitializeSolution: Callable[[],np.ndarray]
            Function to initialize a feasible solution/individual

        GenerateNeighborhood: Callable[[np.ndarray],list[np.ndarray]]
            Function to generate the neighborhood of a solution

        TemperatureSchedule: Callable[[float,float],list[float]],
            Function to get the temperature in each iteration based on initial and final temperature hyperparameters
        """

        self.ObjectiveFunction = ObjectiveFunction
        self.InitializeSolution = InitializeSolution
        self.GenerateNeighborhood = GenerateNeighborhood
        self.TemperatureSchedule = TemperatureSchedule

    def __call__(
            self,
            Iteartions: int,
            InitialTemperature: float,
            FinalTemperature: float,
        ) -> tuple[np.ndarray,list[float]]:
        """
        Method to search the closest optimal solution from a 
        given solution for the objective function.  Return 
        optimal solution and a list of optimal values at 
        each iteration.

        Parameters
        ----------
        Iteartions: int
            A dummy variable to maintain consistency in the library. It is not used during the optimization

        InitialTemperature: float
            Initial temperature for the annealing

        FinalTemperature: float
            Final temperature for the annealing

        Returns
        -------
        OptimalIndividual: np.ndarray
            Best solution/individual that was founded

        Snapshots: list[float] 
            List of the optimal values at each iteration/generation
        """

        CurrentSolution = self.InitializeSolution()
        CurrentFitnessValue = self.ObjectiveFunction(CurrentSolution)

        OptimalSolution = CurrentSolution.copy()
        OptimalFitnessValue = CurrentFitnessValue

        Snapshots = []
        Snapshots.append(OptimalFitnessValue)

        for current_temperature in self.TemperatureSchedule(InitialTemperature,FinalTemperature):
            current_neighborhood = self.GenerateNeighborhood(CurrentSolution)
            random_neighbor = choice(current_neighborhood)

            fitness_neighbor = self.ObjectiveFunction(random_neighbor)
            if fitness_neighbor <= OptimalFitnessValue:
                OptimalSolution = random_neighbor
                OptimalFitnessValue = fitness_neighbor
            
            if (fitness_neighbor <= CurrentFitnessValue) or (np.random.rand() < np.e**(-(fitness_neighbor-CurrentFitnessValue)/current_temperature)):
                CurrentSolution = random_neighbor
                CurrentFitnessValue = fitness_neighbor
            
            Snapshots.append(OptimalFitnessValue)

        return OptimalSolution , Snapshots
    
    def FineTuningHyperparameters(
            self,
            Hyperparameters: dict[str,tuple[str,tuple]] = {
                    'InitialTemperature': ('float',(50,100)),
                    'FinalTemperature': ('float',(0.001,10)),
                },
            NumTrials: int = 10,
            NumJobs: int = 1,
        ) -> dict[str,Any]:

        return super().FineTuningHyperparameters(0,Hyperparameters,NumTrials,NumJobs)