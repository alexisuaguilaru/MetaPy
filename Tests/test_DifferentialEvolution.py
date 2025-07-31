from MetaPy import DifferentialEvolutionOptimizer , RealValueIndividuals
from scipy.optimize import rosen

import os

# Defining instance for testing and auxiliar variables

Dim = 2
ObjFunc = rosen
PopFunc = RealValueIndividuals(-100,100,Dim)

DiffEvol = DifferentialEvolutionOptimizer(
        ObjFunc,
        PopFunc,
    )

iters = 50
params = {
        'PopulationSize': 100, 
        'ScalingFactor': 0.5, 
        'CrossoverRate': 0.5, 
    }

# Test cases

def test_Functionality():
    """
    Function for evaluate functionality of Differential Evolution and check its optimization
    """

    BestSolution , Snapshots = DiffEvol(iters,**params)
    assert Snapshots[0] >= Snapshots[-1]

def test_FineTuning():
    """
    Function for testing fine-tuning of Differential Evolution
    """

    best_params = DiffEvol.FineTuningHyperparameters(iters)

    BestSolution , Snapshots = DiffEvol(iters,**best_params)
    assert Snapshots[0] >= Snapshots[-1]

def test_Simulations():
    """
    Function for testing simulations of Differential Evolution
    """

    file_name = '__Test'
    try:
        DiffEvol.GenerateSimulations(iters,Simulations=2,FileName=file_name,**params)
    except Exception as excpt:
        print(excpt)
        assert False
    else:
        assert True
    finally:
        os.remove(f'{file_name}.parquet')