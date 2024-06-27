def PlottingSnapshots(snapshots,intervals: int,SubPlot_kw: dict={'autoscale_on':False},Fig_kw: dict={'layout':'constrained'}, Scatter_kw: dict={}) -> None:
    """
        Function to scatter plots of snapshots of the population
        -- snapshots : List of snapshots
        -- intervals : Every how many iterations being plotting
        -- SubPlot_kw : Dict with keywords to config each subplot
        -- Fig_kw : Dict with keywords to config the figure of plotting
        -- Scatter_kw : Dict with keywords to config the scatter plotting
    """
    import matplotlib.pyplot as plt
    AmountPlots = len(snapshots)//intervals+1
    RowsCanvas , ResidualPlots = divmod(AmountPlots,3)

    if ResidualPlots>0:
        RowsCanvas += 1
    Fig , Axs = plt.subplots(RowsCanvas,3,figsize=(12,4*RowsCanvas),subplot_kw=SubPlot_kw,**Fig_kw)
    Axs = Axs.flat
    if len(snapshots[0])==2:
        for AxPosition , Snapshot in zip(range(3*RowsCanvas),snapshots[::intervals]):
            Iteration , Population , _ = Snapshot
            X , Y = Population[:,0] , Population[:,1]
            Axs[AxPosition].scatter(X,Y,**Scatter_kw)
            Axs[AxPosition].set_title(f'Iteration : {Iteration}') 
    
    else:
        for AxPosition , Snapshot in zip(range(3*RowsCanvas),snapshots[::intervals]):
            Iteration , Population , Clusters , _ = Snapshot
            X , Y = Population[:,0] , Population[:,1]
            Axs[AxPosition].scatter(X,Y,c=Clusters,**Scatter_kw)
            Axs[AxPosition].set_title(f'Iteration : {Iteration}') 