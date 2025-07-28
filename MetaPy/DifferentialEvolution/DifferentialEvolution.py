from functools import partial

import numpy as np
import optuna

from typing import Callable , Any

suggest_float = optuna.trial.Trial.suggest_float
suggest_int = optuna.trial.Trial.suggest_int
suggest_categorical = optuna.trial.Trial.suggest_categorical

class DifferentialEvolutionOptimizer:
    def __init__(
            self,
            ObjectiveFunction: Callable[[np.ndarray],float],
            InitializePopulation: Callable[[int],np.ndarray],
        ):
        """
        Class for implementation of Differential Evolution 
        based on DE/rand/1/bin. Search the minimum value 
        of `ObjectiveFunction`.
        
        Parameters
        ----------
        ObjectiveFunction: Callable[[np.ndarray],float]
            Function to optimize. Takes a solution/individual of shape `(Dim,)` and returns its fitness value

        InitializePopulation: Callable[[int],np.ndarray]
            Function to create a population of solutions. Return a `np.ndarray` object of shape `(Size,Dim)`
        """

        self.ObjectiveFunction = ObjectiveFunction
        self.InitializePopulation = InitializePopulation

    def FineTuningHyperparameters(
            self,
            Iterations: int,
            Hyperparameters: dict[str,tuple[str,tuple]] = {
                    'PopulationSize': ('int',[1,100]),
                    'ScalingFactor': ('float',[1e-15,1]),
                    'CrossoverRate': ('float',[1e-15,1]),
                },
            NumTrials: int = 10,
            NumJobs: int = 1,
        ) -> dict[str,Any]:
        """
        Method for fine-tuning hyperparameters of 
        Differential Evolution based on find the 
        minimum of `self.ObjectiveFunction`.

        Parameters
        ----------
        Iterations: int
            `Iterations` parameter of `self.__call__`
        
        Hyperparameters: dict[str,tuple[str,tuple]]
            Range of values of each hyperparameter to 
            fine-tune

        NumTrials: int
            `n_trials` parameter for `optuna.create_study`

        NumJobs: int
            `n_jobs` parameter for `optuna.create_study`

        Return
        ------
        BestHyperparameters: dict[str,Any]
            A dict with the best hyperparameters for Differential Evolution
        """

        HyperparameterSuggestFunctions = self.GetHyperparameterSuggestFunctions(Hyperparameters)
        OptunaObjective = self.GetOptunaObjective(Iterations,HyperparameterSuggestFunctions)

        study = optuna.create_study(study_name='OptimizeHyperparameters')
        study.optimize(OptunaObjective,n_trials=NumTrials,n_jobs=NumJobs)

        return study.best_params

    def GetHyperparameterSuggestFunctions(
            self,
            Hyperparameters: dict[str,tuple[str,tuple]],
        ) -> dict[str,Callable]:
        """
        Method for defining/creating functions to 
        suggests values for each hyperparameter 
        using `suggest_*` methods of `optuna.trial.Trial`.

        Parameters
        ----------
        Hyperparameters: dict[str,tuple[str,tuple]]
            Range of values of each hyperparameter to 
            fine-tune

        Return
        ------
        HyperparameterSuggestFunctions: dict[str,Callable]
            A dict with the functions to suggest the hyperparameters for a trial
        """

        HyperparameterSuggestFunctions = dict()
        for hyperparam_name , (type_suggest , params_suggest) in Hyperparameters.items():
            if type_suggest == 'float':
                param_low , param_high = params_suggest
                suggest_function = partial(suggest_float,name=hyperparam_name,low=param_low,high=param_high)

            elif type_suggest == 'int':
                param_low , param_high = params_suggest
                suggest_function = partial(suggest_int,name=hyperparam_name,low=param_low,high=param_high)

            elif type_suggest == 'categorical':
                suggest_function = partial(suggest_categorical,name=hyperparam_name,choices=params_suggest)

            else:
                raise Exception(f'{type_suggest} Not Implemented')

            HyperparameterSuggestFunctions[hyperparam_name] = suggest_function

        return HyperparameterSuggestFunctions

    def GetOptunaObjective(
            self,
            Iterations: int,
            HyperparameterSuggestFunctions: dict[str,Callable],
        ) -> Callable:
        """
        Method for getting the `func` parameter 
        of `optuna.Study.optimize` which calls 
        `self.__call__` method with a certain 
        hyperparameters.

        Parameters
        ----------
        Iterations: int
            `Iterations` parameter of `self.__call__`

        HyperparameterSuggestFunctions: dict[str,Callable]
            Functions to suggest the hyperparameters for a trial
        
        Return
        ------
        OptunaObjective: Callable
            Objective function for `optuna.Study`
        """

        def OptunaObjective(
                Trial: optuna.trial.Trial,
            ) -> float:
            """
            Function used to evaluate Differential 
            Evolution with a certain hyperparameters

            Parameters
            ----------
            Trial: optuna.trial.Trial
                Trial from `optuna.study.Study` object

            Return
            ------
            OptimalVale: float
                Optimal (best) value of `self.__call__`
            """

            SuggestedHyperparameters = self.GetSuggestedHyperparameters(Trial,HyperparameterSuggestFunctions)
            Result = self.__call__(Iterations,**SuggestedHyperparameters)
            return Result[1][-1]

        return OptunaObjective

    def GetSuggestedHyperparameters(
            self,
            Trial: optuna.trial.Trial,
            HyperparameterSuggestFunctions: dict[str,Callable],
        ) -> dict[str,Any]:
        """
        Method for getting the hyperparameters for 
        a trial using the suggest functions.

        Parameters
        ----------
        Trial: optuna.trial.Trial
            Trial from `optuna.study.Study` object

        HyperparameterSuggestFunctions: dict[str,Callable]
            Functions to suggest the hyperparameters for a trial

        Return
        ------
        SuggestedHyperparameters: dict[str,Any]
            A dict with the suggested hyperparameters for a trial
        """

        SuggestedHyperparameters = {}
        for hyperparam_name , suggest_function in HyperparameterSuggestFunctions.items():
            SuggestedHyperparameters[hyperparam_name] = suggest_function(Trial)

        return SuggestedHyperparameters

    def __call__(
            self,
            Iterations: int,
            PopulationSize: int,
            ScalingFactor: float,
            CrossoverRate: float,
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
            Parameter NP. Size of population of solutions

        RangeScalingFactor: float
            Parameter F. Scaling factor for difference between vector.

        RangeCrossoverRate: float
            Parameter Cr. Crossover rate for crossover operation

        Returns
        -------
        OptimalIndividual: np.ndarray
            Best solution/individual that was founded

        Snapshots: list[float] 
            List of the optimal values at each iteration/generation
        """

        self.PopulationSize = PopulationSize
        self.ScalingFactor = ScalingFactor
        self.CrossoverRate = CrossoverRate
        
        self.InitializeOptimization()
        self.OptimalIndividual , self.OptimalValue = self.BestOptimalIndividual()
        self.ProblemDimension = self.OptimalIndividual.shape[0]

        self.Snapshots = []
        self.WriteSnapshot()

        self.FindOptimal(Iterations)
        
        return self.OptimalIndividual , self.Snapshots
    
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
            self.CrossoverOperation()

            self.SelectionOperation()
            
            self.WriteSnapshot()

    def MutationOperation(
            self,
        ) -> None:
        """
        Method for applying Differential Evolution 
        Mutation Operation to the `Population`.
        """

        self.MutatedPopulation = self.RandomSampleSolutions()
        self.MutatedPopulation += self.ScalingFactor*(self.RandomSampleSolutions()-self.RandomSampleSolutions())
    
    def CrossoverOperation(
            self,
        ) -> None:
        """
        Method for applying Differential Evolution 
        Crossover Operation to the `Population`.
        """

        CrossoverThreshold = np.random.random((self.PopulationSize,self.ProblemDimension)) <= self.CrossoverRate
        
        IndexesMutated = np.random.randint(self.ProblemDimension,size=self.PopulationSize)
        CrossoverThreshold[self.PopulationIndexes,IndexesMutated] = True
        
        self.CrossoverPopulation = self.Population.copy()
        self.CrossoverPopulation[CrossoverThreshold] = self.MutatedPopulation[CrossoverThreshold]

        self.FitnessCrossoverPopulation = np.apply_along_axis(self.ObjectiveFunction,1,self.CrossoverPopulation)

    def SelectionOperation(
            self,
        ) -> None:
        """
        Method for applying Differential Evolution 
        Selection Operation between the `Population` 
        and `CrossoverPopulation` (Offsprings solutions/individuals).
        """

        for index_individual in self.PopulationIndexes:
            fitness_crossovered = self.FitnessCrossoverPopulation[index_individual]
            fitness_population = self.FitnessValuesPopulation[index_individual]
            
            if fitness_crossovered <= fitness_population:
                crossovered_individual = self.CrossoverPopulation[index_individual]
                self.Population[index_individual] = crossovered_individual
                self.FitnessValuesPopulation[index_individual] = fitness_crossovered

                if fitness_crossovered < self.OptimalValue:
                    self.OptimalValue = fitness_crossovered
                    self.OptimalIndividual = crossovered_individual

    def InitializeOptimization(
            self,
        ) -> None:
        """
        Method for initializing `Population` and 
        `FitnessValuesPopulation` attributes.
        """

        self.Population = self.InitializePopulation(self.PopulationSize)
        self.FitnessValuesPopulation = np.apply_along_axis(self.ObjectiveFunction,1,self.Population)

        self.PopulationIndexes = np.arange(self.PopulationSize)

    def BestOptimalIndividual(
            self,
        ) -> tuple[np.ndarray,float]:
        """
        Method for finding the best optimal 
        individual at the current `Population`.
            
        Returns
        -------
        BestOptimalSolution: np.ndarray
            Best optimal solution/individual in the `Population`
        
        BestOptimalValue: float
            Best optimal (minimum) value in the `Population`
        """

        IndexOptimalIndividual = np.argmin(self.FitnessValuesPopulation)
        return self.Population[IndexOptimalIndividual] , self.FitnessValuesPopulation[IndexOptimalIndividual]
    
    def WriteSnapshot(
            self,
        ) -> None:
        """
        Method to save a snapshot of the optimal/best 
        value at current generation.
        """

        self.Snapshots.append(self.OptimalValue)

    def RandomSampleSolutions(
            self,
        ) -> np.ndarray:
        """
        Method for generating a random 
        sample of solutions/individuals 
        from the `Population`.

        Return
        ------
        RandomSample : np.ndarray
            Random sample of solutions/individuals from the `Population`
        """

        RandomIndexes =  np.random.randint(self.PopulationSize,size=self.PopulationSize)
        return self.Population[RandomIndexes]