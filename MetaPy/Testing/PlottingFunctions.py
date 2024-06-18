def PlottingSnapshots(snapshots,intervals:int) -> None:
    """
        Function to scatter plots of snapshots of the population
        -- snapshots : List of snapshots
        -- intervals : Every how many iterations being plotting
    """
    import matplotlib.pyplot as plt
    AmountPlots = len(snapshots)//intervals
    RowsCanvas , ResidualPlots = divmod(AmountPlots,3) 
    if ResidualPlots>0:
        RowsCanvas += 1
    Fig , Axs = plt.subplots(RowsCanvas,3,figsize=(12,4*RowsCanvas),layout='constrained',subplot_kw={'autoscale_on':False})
    Axs = Axs.flat
    if len(snapshots[0])==2:
        for AxPosition , Snapshot in zip(range(3*RowsCanvas),snapshots):
            Iteration , Population = Snapshot
            X , Y = Population[:,0] , Population[:,1]
            Axs[AxPosition].scatter(X,Y)
            Axs[AxPosition].set_title(f'Iteration : {Iteration}') 
    else:
        for AxPosition , Snapshot in zip(range(3*RowsCanvas),snapshots):
            Iteration , Population , Clusters = Snapshot
            X , Y = Population[:,0] , Population[:,1]
            Axs[AxPosition].scatter(X,Y,c=Clusters)
            Axs[AxPosition].set_title(f'Iteration : {Iteration}') 