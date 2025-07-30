from MetaPy import DifferentialEvolutionOptimizer , RealValueIndividuals
from scipy.optimize import rosen

# Defining instance for testing

Dim = 2
ObjFunc = rosen
PopFunc = RealValueIndividuals(-100,100,Dim)

DiffEvol = DifferentialEvolutionOptimizer(
        ObjFunc,
        PopFunc,
    )

iters = 50

def test_Functionality():
    """
    Function for evaluate functionality of Differential Evolution and check its optimization
    """

    params = {
        'PopulationSize': 100, 
        'ScalingFactor': 0.5, 
        'CrossoverRate': 0.5, 
    }
    BestSolution , Snapshots = DiffEvol(iters,**params)
    assert Snapshots[0] >= Snapshots[-1]

def test_FineTuning():
    """
    Function for testing fine-tuning  of Differential Evolution
    """

    best_params = DiffEvol.FineTuningHyperparameters(iters)

    BestSolution , Snapshots = DiffEvol(iters,**best_params)
    assert Snapshots[0] >= Snapshots[-1]