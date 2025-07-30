from functools import partial

import optuna

from typing import Callable , Any

SuggestFloat = optuna.trial.Trial.suggest_float
SuggestInt = optuna.trial.Trial.suggest_int
SuggestCategorical = optuna.trial.Trial.suggest_categorical

class MetaheuristicOptimizer:
    """
    Base class for implementing metaheuristics/optimizers, 
    adding methods for fine-tuning of hyperparameters using 
    [Optuna](https://optuna.org/).

    It is required to implement `__call__` and (optionally) 
    `FineTuningHyperparameters` methods.
    """

    def FineTuningHyperparameters(
            self,
            Iterations: int,
            Hyperparameters: dict[str,tuple[str,tuple]],
            NumTrials: int = 10,
            NumJobs: int = 1,
        ) -> dict[str,Any]:
        """
        Method for fine-tuning hyperparameters of 
        a Metaheuristic based on find the 
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
            A dict with the best hyperparameters for the Metaheuristic
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
                suggest_function = partial(SuggestFloat,name=hyperparam_name,low=param_low,high=param_high)

            elif type_suggest == 'int':
                param_low , param_high = params_suggest
                suggest_function = partial(SuggestInt,name=hyperparam_name,low=param_low,high=param_high)

            elif type_suggest == 'categorical':
                suggest_function = partial(SuggestCategorical,name=hyperparam_name,choices=params_suggest)

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
        `self.__call__` method with certain 
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
            Function used to evaluate the Metaheuristic 
            with certain hyperparameters

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
        a trial using the `HyperparameterSuggestFunctions`.

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