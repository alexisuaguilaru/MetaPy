from DifferentialEvolution import DifferentialEvolution

class DifferentialEvolution__Plotting(DifferentialEvolution):
    def __call__(self, iterations: int, populationSize: int, scalingFactor: float, crossoverRate: float, snapshots: int):
        """
            Method to search optimal solution for objective function
            -- iterations : Amount of iterations
            -- populationSize : Parameter NP. Size of solutions population
            -- scalingFactor : Parameter F. Scaling factor for difference vector 
            -- crossoverRate : Parameter Cr. Crossover rate for crossover operation
            -- snapshots : Number of plots of the population 

            Plot snapshots of the population of solutions
            Return the best optimal solution, because of implementation will be the minimum
        """
        self.MaxIterations = iterations
        self.MaxSnapshots = snapshots
        self.SnapshotsSaved = []
        BestOptimalSolution =  super().__call__(iterations, populationSize, scalingFactor, crossoverRate)
        self.diffevol_PlottingSnapshots()
        return BestOptimalSolution
    
    def diffevol_IterativeSearch(self, iteration: int) -> None:
        if iteration%(self.MaxIterations//self.MaxSnapshots) == 0:
            self.diffevol_SnapshotPopulation(iteration)
        super().diffevol_IterativeSearch(iteration)

    def diffevol_SnapshotPopulation(self,iteration: int) -> None:
        from copy import deepcopy
        self.SnapshotsSaved.append((iteration,deepcopy(self.population)))

    def diffevol_PlottingSnapshots(self) -> None:
        import matplotlib.pyplot as plt
        AmountPlots = len(self.SnapshotsSaved)
        RowsCanvas , ResidualPlots = divmod(AmountPlots,3)
        if ResidualPlots>0:
            RowsCanvas += 1
        Fig , Axs = plt.subplots(RowsCanvas,3)
        Axs = Axs.flat
        for AxPosition , Snapshot in zip(range(3*RowsCanvas),self.SnapshotsSaved):
            Iteration , Population = Snapshot
            Axs[AxPosition].scatter(Population)
            Axs[AxPosition].set_tittle(f'Iteration : {Iteration}') 