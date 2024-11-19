from multiprocess import Pool
import numpy as np
import matplotlib.pyplot as plt

def StatisticsOptimizer(optimizer,argsOptimizer:tuple=(),kwargsOptimizer:dict={},numberRuns:int=100,process:int|None=None) -> tuple[float,float,np.ndarray]:
    """
        Function to run/execute the optimizer 
        with the given parameters to simulate 
        several callings.

        -- optimizer : Optimizer to simulate 
        -- argsOptimizer : Arguments passed 
        to the optimizer
        -- kwargsOptimizer : Keyword arguments passed 
        to the optimizer 
        -- numberRuns : Number of runs, executions 
        or simulations of the optimizer
        -- process : Number of simulations that are 
        executing at same time

        Return mean, standard deviation and simulation results
    """
    functionOptimizer = lambda parameters : optimizer(*parameters[0],**parameters[1])[-1][-1][-1]
    with Pool(process) as pool:
        optimalsFound = np.array(pool.map(functionOptimizer,[(argsOptimizer,kwargsOptimizer) for _ in range(numberRuns)]),dtype=float)
    return np.mean(optimalsFound) , np.std(optimalsFound) , optimalsFound

def PlotOptimalsRestuls(optimalsFound:np.ndarray,meanOptimals:float,titleOptimals:str,binsHisogram:int=50) -> None:
    """
        Function to plot a histogram based on 
        the optimal values found.

        -- optimalsFound : Optimal values of 
        the simulation results
        -- meanOptimals : Mean value of optimal 
        values found
        -- titleOptimals : Tittle of the histogram 
        -- binsHisogram : Number of bins used to  
        create the histogram
    """
    plt.hist(optimalsFound,bins=binsHisogram,alpha=0.8,color='gray')
    plt.axvline(meanOptimals,c='k',ls=':')
    plt.title(titleOptimals)
    plt.xlabel('Optimal Value')
    plt.ylabel('Number of Values')
