from joblib import Parallel, delayed

from typing import Iterator , Callable

class MetaheuristicSimulations:
    """
    """

    def GenerateSimulations(
            self,
            Iterations: int,
            *Hyperparameters,
            Simulations: int = 10,
            NumJobs: int = 1,
            **KwHyperparameters,
        ) -> Iterator[list[float]]:
        """
        Method for simulating a Metaheuristic or 
        calling `self.__call__` method 
        several times with given hyperparameters.

        Parameters
        ----------
        Iterations: int
            `Iterations` parameter of `self.__call__`

        Hyperparameters: tuple[Any]
            Args parameters of `self.__call__`

        Simulations: int
            Number of simulations or calling for self

        NumJobs: int
            `n_jobs` parameter of [joblib](https://joblib.readthedocs.io/en/stable/)

        KwHyperparameters: dict[str,Any]
            Kwargs parameters of `self.__call__`

        Return
        ------
        ResultSimulations: Iterator[list[float]]
            Iterator with the snapshots of each simulation or calling of the Metaheuristic
        """

        WrappedCallMethod = self.WrapCallMethod(Iterations,*Hyperparameters,**KwHyperparameters)
        ResultSimulations = Parallel(n_jobs=NumJobs,return_as='generator')(delayed(WrappedCallMethod)() for _ in range(Simulations))

        return ResultSimulations

    def WrapCallMethod(
            self,
            Iterations: int,
            *Hyperparameters,
            **KwHyperparameters,
        ) -> Callable[[None],list[float]]: 
        """
        Method for wrapping the `self.__call__` method 
        for calling it with given parameters.

        Iterations: int
            `Iterations` parameter of `self.__call__`

        Hyperparameters: tuple[Any]
            Args parameters of `self.__call__`

        KwHyperparameters: dict[str,Any]
            Kwargs parameters of `self.__call__`

        Return
        ------
            A wrapped version of `self.__call__` method
        """

        def WrappedCallMethod() -> list[float]:
            """
            Function for calling `self.__call__` method 
            with given parameters and getting its snapshots.

            Return
            ------
            Snapshots: list[float]
                `Snapshots` return of `self.__call__`
            """

            return self.__call__(Iterations,*Hyperparameters,**KwHyperparameters,)[1]
        
        return WrappedCallMethod