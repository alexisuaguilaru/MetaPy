import matplotlib.pyplot as plt
import numpy as np

def PlottingSnapshots(snapshots,intervals: int,SubPlot_kw: dict={'autoscale_on':False},Fig_kw: dict={'layout':'constrained'}, Scatter_kw: dict={}) -> None:
    """
        Function to scatter plots of snapshots of the population
        -- snapshots : List of snapshots
        -- intervals : Every how many iterations being plotting
        -- SubPlot_kw : Dict with keywords to config each subplot
        -- Fig_kw : Dict with keywords to config the figure of plotting
        -- Scatter_kw : Dict with keywords to config the scatter plotting
    """
    AmountPlots = len(snapshots)//intervals+1
    RowsCanvas , ResidualPlots = divmod(AmountPlots,3)

    if ResidualPlots>0:
        RowsCanvas += 1
    Fig , Axs = plt.subplots(RowsCanvas,3,figsize=(12,4*RowsCanvas),subplot_kw=SubPlot_kw,**Fig_kw)
    Axs = Axs.flat
    if len(snapshots[0]) == 4:
        for AxPosition , Snapshot in zip(range(3*RowsCanvas),snapshots[::intervals]):
            Iteration , Population , *_ = Snapshot
            X , Y = Population[:,0] , Population[:,1]
            Axs[AxPosition].scatter(X,Y,**Scatter_kw)
            Axs[AxPosition].set_title(f'Iteration : {Iteration}') 

    else:
        for AxPosition , Snapshot in zip(range(3*RowsCanvas),snapshots[::intervals]):
            Iteration , Population , _ , Clusters , *_ = Snapshot
            X , Y = Population[:,0] , Population[:,1]
            Axs[AxPosition].scatter(X,Y,c=Clusters,**Scatter_kw)
            Axs[AxPosition].set_title(f'Iteration : {Iteration}')

def PlottingOptimalsFound_Iterations(snapshots,fmt: str='.:b',YScale='linear',Plot_kw: dict={}):
    """
        Function to plot optimal values at each iteration / convergence of solutions
        -- snapshots : List of snapshots
        -- fmt : Formatting of marker, linestyle and color 
        -- YScale : Type of scale of Y axis 
        -- Plot_kw : Dict with keywords to config the plot
    """
    Fig , Ax = plt.subplots(figsize=(10,6))
    X_Iterations , Y_OptimalValues = [Snapshot[0] for Snapshot in snapshots] , [Snapshot[-1] for Snapshot in snapshots]
    Ax.set_yscale(YScale)
    Ax.plot(X_Iterations,Y_OptimalValues,fmt,**Plot_kw)

def MultiPlottingOptimalsFound_Iterations(parameterList: list,parameterName: str,optimizer,kwargsOptimizer: dict,ftmList: list[str],subplot_kw: dict={}):
    """
        Function to optimal values at each iteration with variation of 
        a parameter
        -- parameterList : List of possible values taking for the parameter
        -- parameterName : Name of the parameter
        -- optimizer : Metaheuristic being studied
        -- kwargsOptimizer : Fixed parameters of the optimizer 
        -- ftmList : List of line styles for the plot
        -- subplot_kw : Keywords of the plot
    """
    Fig , Ax = plt.subplots(subplot_kw=subplot_kw,figsize=(10,6))
    for parameterValue,fmt in zip(parameterList,ftmList):
        kwargsOptimizer[parameterName] = parameterValue
        _ , snapshots = optimizer(**kwargsOptimizer)
        X_Iterations , Y_OptimalValues = [Snapshot[0] for Snapshot in snapshots] , [Snapshot[-1] for Snapshot in snapshots]
        Ax.plot(X_Iterations,Y_OptimalValues,fmt,label=f'{parameterName}:{parameterValue}')
    Ax.legend()

def PlottingObjectiveFunction(objectiveFunction,PlotSurface: bool=True,domainX: tuple[float,float]=(-100,100),domainY: tuple[float,float]=(-100,100),stepSize: float=1):
    """
        Function to plot objective function
        -- objectiveFunction : Objective function being plotting
        -- PlotSurface : If it will surface or contour plot 
        -- domainX : Range of X's values
        -- domainY : Range of Y's values
        -- stepSize : Step size between values in the domain

    """
    axisX , axisY = np.arange(*domainX,stepSize) , np.arange(*domainY,stepSize)
    axisZ = np.array(np.meshgrid(axisX,axisY))
    axisZ = np.apply_along_axis(objectiveFunction,0,axisZ)
    if PlotSurface:
        Fig , Ax = plt.subplots(figsize=(10,7),subplot_kw={'projection':'3d'})
        plot = Ax.plot_surface(axisX,axisY,axisZ,cmap='cool',edgecolor='black',lw=0.02)
        Ax.set(xlabel='X', ylabel='Y', zlabel='Z')
    else:
        Fig , Ax = plt.subplots(figsize=(10,7))
        plot = Ax.contour(axisX,axisY,axisZ,cmap='cool')
        Ax.set(xlabel='X', ylabel='Y')
    Fig.colorbar(plot,shrink=0.5, aspect=5, pad=0.15, label='')
    plt.show()

