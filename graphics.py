import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd


# create a correlogram as a heatmap for a correlation matrix
def correlogramCorr(matrix=None, dec=2, title='Correlogram', valmin=-1, valmax=1, Xlabel=None, Ylabel=None):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(matrix, dec), vmin=valmin, vmax=valmax, xticklabels=Xlabel, yticklabels=Ylabel, cmap='bwr', annot=True)

# correlograms
def correlogram(matrix=None, dec=2, title='Correlogram', valmin=-1, valmax=1):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(matrix, dec), vmin=valmin, vmax=valmax, cmap='bwr', annot=True)

def correlCircle(matrix=None, V1=0, V2=1, dec=1,
                 XLabel=None, YLabel=None, minVal=-1, maxVal=1, title='Correlation Circle'):
    plt.figure(title, figsize=(8, 8))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    T = [t for t in np.arange(0, np.pi*2, 0.01)]
    X = [np.cos(t) for t in T]
    Y = [np.sin(t) for t in T]
    plt.plot(X, Y)
    plt.axhline(y=0, color='g')
    plt.axvline(x=0, color='g')
    if XLabel==None or YLabel==None:
        if isinstance(matrix, pd.DataFrame):
            plt.xlabel(matrix.columns[V1], fontsize=14, color='k', verticalalignment='top')
            plt.ylabel(matrix.columns[V2], fontsize=14, color='k', verticalalignment='bottom')
        else:
            plt.xlabel('Var '+str(V1+1), fontsize=14, color='k', verticalalignment='top')
            plt.ylabel('Var '+str(V2+1), fontsize=14, color='k', verticalalignment='bottom')
    else:
        plt.xlabel(XLabel, fontsize=14, color='k', verticalalignment='top')
        plt.ylabel(YLabel, fontsize=14, color='k', verticalalignment='bottom')

    if isinstance(matrix, np.ndarray):
        plt.scatter(x=matrix[:, V1], y=matrix[:, V2], c='r', vmin=minVal, vmax=maxVal)
        for i in range(matrix.shape[0]):
            plt.text(x=matrix[i, V1], y=matrix[i, V2], s='(' +
                    str(np.round(matrix[i, V1], dec))
                     + ', ' + str(np.round(matrix[i, V2], dec)) + ')')

    if isinstance(matrix, pd.DataFrame):
        # plt.text(x=0.5, y=0.5, s='we have a pandas.DatFrame')
        plt.scatter(x=matrix.iloc[:, V1], y=matrix.iloc[:, V2], c='b', vmin=minVal, vmax=maxVal)
        for i in range(matrix.values.shape[0]):
            plt.text(x=matrix.iloc[i, V1], y=matrix.iloc[i, V2], s='(' +
                    str(np.round(matrix.iloc[i, V1], dec))
                     + ', ' + str(np.round(matrix.iloc[i, V2], dec)) + ')')


def principalComponents(eigenValues=None, XLabel='Principal components', YLabel='Eigenvalues (variance)',
                        title='Explained variance by the principal components'):
    plt.figure(title, figsize=(13, 8))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    plt.xlabel(XLabel, fontsize=14, color='k', verticalalignment='top')
    plt.ylabel(YLabel, fontsize=14, color='k', verticalalignment='bottom')
    # f(x) = y
    # create labels for the X axis: C1, C2, C3, ...
    components = ['C'+str(j+1) for j in range(eigenValues.shape[0])]
    plt.plot(components, eigenValues, 'bo-')
    plt.axhline(y=1, color='r')

def scatterPlot(matrix=None, title='Scatter plot', Xlabel='Variables', Ylabel='Observations'):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    plt.xlabel(xlabel=Xlabel, fontsize=14, color='k', verticalalignment='top')
    plt.ylabel(ylabel=Ylabel, fontsize=14, color='k', verticalalignment='bottom')

    plt.scatter(x=matrix.iloc[:, 0].values, y=matrix.index[:])

def show():
    plt.show()