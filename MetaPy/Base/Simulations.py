from joblib import Parallel, delayed
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.dataset import dataset , Dataset

from typing import Iterator , Callable

class MetaheuristicSimulations:
    """
    Base class for generating the results 
    of several simulations of a Metaheuristic 
    with given hyperparameters. It is not required 
    to implement any method. 
    """

    def GenerateSimulations(
            self,
            Iterations: int,
            *Hyperparameters,
            Simulations: int = 10,
            NumJobs: int = 1,
            FileName: str = 'Results',
            **KwHyperparameters,
        ) -> Dataset:
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

        FileName: str
            Name of the file where the snapshots of 
            each simulation/calling are saved. 
            `source` parameter of `pyarrow.dataset.dataset`

        KwHyperparameters: dict[str,Any]
            Kwargs parameters of `self.__call__`

        Return
        ------
        DatasetResults: pyarrow.dataset.Dataset
            PyArrow dataset with the snapshots of each simulation
        """

        WrappedCallMethod = self.WrapCallMethod(Iterations,*Hyperparameters,**KwHyperparameters)
        ResultSimulations = Parallel(n_jobs=NumJobs,return_as='generator')(delayed(WrappedCallMethod)() for _ in range(Simulations))

        self.SaveResults(ResultSimulations,Iterations,FileName)

        return dataset(f'{FileName}.parquet')

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
    
    def SaveResults(
            self,
            Results: Iterator[list[float]],
            Iterations: int,
            FileName: str,
        ) -> None:
        """
        Method for saving the results of the 
        simulations into a *.parquet file.

        Results: Iterator[list[float]]
            Iterator with the snapshots of each 
            simulation

        Iterations: int
            `Iterations` parameter of `self.__call__`

        FileName: str
            Name of the file where the snapshots 
            are saved. `where` parameter of 
            `pyarrow.parquet.ParquetWriter` 
        """

        SchemaFile = pa.schema((f'{iteration}',pa.float64()) for iteration in range(Iterations+1))
        with pq.ParquetWriter(f'{FileName}.parquet',SchemaFile) as WriterFile:
            for result_simulation in Results:
                arrays = [pa.array([optimal_value]) for optimal_value in result_simulation]
                batch = pa.RecordBatch.from_arrays(arrays,schema=SchemaFile)
                WriterFile.write_batch(batch)