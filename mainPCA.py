import pandas as pd
import utils as utl
import PCA.PCA as pca
import graphics as g

table = pd.read_csv('./dataIN/Offences.csv',index_col=0, na_values=':')
print(table)

# list of observation labels
observations = table.index.values
# list of useful variables
variables = table.columns.values
datasetValues = table.values

# no. of variables
m = variables.shape[0]
print('Variables: ', m)
# no. of observations
n = len(observations)
print('Observations: ', n)

# replace the missing values and put it into a dataFrame
X = utl.replaceNAN(datasetValues)
X_df = pd.DataFrame(data=X, index=observations, columns=variables)
X_df.to_csv('./dataOUT/DatasetX.csv')

X = table[variables].values

# ******************** PCA ********************
# apply PCA on the initial data (standardisation included)
modelPCA = pca.PCA(X)

# Xstd - X standardised
Xstd = modelPCA.getXstd()
# save Xstd in a CSV file
Xstd_df = pd.DataFrame(data=Xstd, index=observations, columns=variables)
Xstd_df.to_csv('./dataOUT/Xstd.csv')

# correlation matrix of causal variables
corr = modelPCA.getCorr()
# save correlation matrix in a CSV file
corr_df = pd.DataFrame(data=corr, index=variables, columns=variables)
corr_df.to_csv('./dataOUT/Corr.csv')
g.correlogramCorr(matrix=corr, Xlabel=variables, Ylabel=variables, title='Correlogram of correlation')

# principal components
principalComps = modelPCA.getPrincipalComps()
components = ['C'+str(j+1) for j in range(X.shape[1])]
# save the principal components into a CSV file
principalComps_df = pd.DataFrame(data=principalComps, index=observations, columns=components)
principalComps_df.to_csv('./dataOUT/PrincipalComponents.csv')\

# eigenvalues
eigenvalues = modelPCA.getEigenValues()
g.principalComponents(eigenValues=eigenvalues, title='The eigenvalues')

# correlation factors (factor loadings)
factorLoadings = modelPCA.getRxc() # correlation between the initial variables and the principal components
# save the factor loadings into a CSV file
factorLoadings_df = pd.DataFrame(data=factorLoadings, index=variables, columns=components)
factorLoadings_df.to_csv('./dataOUT/FactorLoadings.csv')
g.correlogram(matrix=factorLoadings_df, title='Correlogram of factor loadings')

# scores (standardized principal components)
scores = modelPCA.getScores()
# save the scores matrix into s CSV file
scores_df = pd.DataFrame(data=scores, index=observations, columns=components)
scores_df.to_csv('./dataOUT/Scores.csv')
g.correlogram(matrix=scores_df, title='Correlogram of scores')

# quality of observations representation
qualityObs = modelPCA.getQualityObs()
# save the quality of observations representation into a CSV file
qualityObs_df = pd.DataFrame(data=qualityObs, index=observations, columns=components)
qualityObs_df.to_csv('./dataOUT/QualityObs.csv')
g.correlogram(matrix=qualityObs_df, title='Correlogram of quality of observations')

# observation contributions to the axes variance
contribObs = modelPCA.getContribObs()
# save the obs contrib into a CSV file
contribObs_df = pd.DataFrame(data=contribObs, index=observations, columns=components)
contribObs_df.to_csv('./dataOUT/ContribObs.csv')
g.correlogram(matrix=contribObs_df, title='Observation contributions to the axes variance')

# communalities
commun = modelPCA.getCommun()
# save the communalities into a CSV file
commun_df = pd.DataFrame(data=commun, index=variables, columns=components)
commun_df.to_csv('./dataOUT/Communalities.csv')
g.correlogram(matrix=commun_df, title='Correlogram of communalities')

# display all graphs
g.show()