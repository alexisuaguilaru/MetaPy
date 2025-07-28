from MetaPy import DifferentialEvolutionOptimizer , RealValueIndividuals
from scipy.optimize import rosen

def GetResults():
    """
    Function for simulate Differential Evolution with Rosenbrock function
    """

    Dim = 2
    ObjFunc = rosen
    PopFunc = RealValueIndividuals(-100,100,Dim)

    DiffEvol = DifferentialEvolutionOptimizer(
            ObjFunc,
            PopFunc,
        )
    
    iters = 100
    best_params = DiffEvol.FineTuningHyperparameters(iters)

    BestSolution , Snapshots = DiffEvol(iters,**best_params)

    return BestSolution , Snapshots

def test_Functionality():
    """
    Function for evaluate functionality of Differential Evolution and check its optimization
    """

    _ , Snapshots = GetResults()
    assert Snapshots[0] >= Snapshots[-1]