from DifferentialEvolution import DifferentialEvolution

class DifferentialEvolution__Plotting(DifferentialEvolution):
    def __init__(self, objectiveFunction, initializeIndividual,SubPlot_kw:dict={},Fig_kw:dict={}):
        """
            Class for Differential Evolution Metaheuristic with plotting of snapshots of the population in two dimensions
            -- objectiveFunction : Function being optimized 
            -- initializeIndividual : Function to create individuals
            -- SubPlot_kw : Dict with keywords to config each subplot
            -- Fig_kw : Dict with keywords to config the figure of plotting

            Based on DE/rand/1/bin
        """
        self.SubPlot_kw = SubPlot_kw
        self.Fig_kw = Fig_kw
        super().__init__(objectiveFunction, initializeIndividual)

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
        self.diffevol_SnapshotPopulation(iterations)
        self.diffevol_PlottingSnapshots()
        return BestOptimalSolution
    
    def diffevol_IterativeSearch(self, iteration: int) -> None:
        """
            Method to search optimal solutions iteratively 
            -- iteration : Number of iteration

            Save snapshots of the population based on iteration
        """
        if iteration%(self.MaxIterations//self.MaxSnapshots) == 0:
            self.diffevol_SnapshotPopulation(iteration)
        super().diffevol_IterativeSearch(iteration)

    def diffevol_SnapshotPopulation(self,iteration: int) -> None:
        """
            Method to save a snapshot of the population at iteration-st
        """
        from copy import deepcopy
        self.SnapshotsSaved.append((iteration,deepcopy(self.population)))

    def diffevol_PlottingSnapshots(self) -> None:
        """
            Method to plot the saved snapshots
        """
        import matplotlib.pyplot as plt
        AmountPlots = len(self.SnapshotsSaved)
        RowsCanvas , ResidualPlots = divmod(AmountPlots,3)
        if ResidualPlots>0:
            RowsCanvas += 1
        Fig , Axs = plt.subplots(RowsCanvas,3,subplot_kw=self.SubPlot_kw,figsize=(12,4*RowsCanvas),layout='constrained',**self.Fig_kw)
        Axs = Axs.flat
        for AxPosition , Snapshot in zip(range(3*RowsCanvas),self.SnapshotsSaved):
            Iteration , Population = Snapshot
            X , Y = [ind[0] for ind in Population] , [ind[1] for ind in Population]
            Axs[AxPosition].scatter(X,Y)
            Axs[AxPosition].set_title(f'Iteration : {Iteration}') 