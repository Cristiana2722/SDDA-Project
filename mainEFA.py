import numpy as np
import pandas as pd
import utils as utl
import EFA.EFA as efa
import factor_analyzer as fa
import graphics as g

table = pd.read_csv('./dataIN/Offences.csv', index_col=0, na_values=':')
print(table)

# list of observation labels
observations = table.index.values
# list of useful variables
variables = table.columns.values
datasetValues = table.values

# replace the missing values
X = utl.replaceNAN(datasetValues)
X_df = pd.DataFrame(data=X, index=observations, columns=variables)

# ******************** EFA ********************
# compute Bartlett sphericity test
sphericityBartlett = fa.calculate_bartlett_sphericity(X_df)
print(sphericityBartlett, type(sphericityBartlett))
if sphericityBartlett[0] > sphericityBartlett[1]:
    print('There is at least one common factor!')
else:
    print('There are no common factors!')
    exit(-1)

# compute Kaiser-Meyer-Olkin (KMO) indices to check the factorability of the initial variables
KMO = fa.calculate_kmo(X_df)
print('KMO: ',KMO)
if KMO[1] > 0.5:
    print('Initial variables can be expressed by at least one common factor!')
else:
    print('The initial variables have no common factors!')
    exit(-2)

vector = KMO[0]
print(vector)
matrix = vector[:, np.newaxis]
print(matrix)

# save KMO indices into a CSV file
KMO_df = pd.DataFrame(data=matrix, index=variables, columns=['KMO indices'])
KMO_df.to_csv('./dataOUT/KMOindices.csv')
# create a correlogram of KMO indices
g.correlogram(matrix=KMO_df, title='Correlogram of KMO indices')

# extract the significant factors
noOfSignificantFactors = 1
chi2TabMin = 1
for k in range(1, len(variables)):
    modelFA = fa.FactorAnalyzer(n_factors=k)
    modelFA.fit(X_df)
    factorLoadingsFA = modelFA.loadings_ # gives us the matrix of common factors
    specificFactors = modelFA.get_uniquenesses() # gives us the specific factors
    print(factorLoadingsFA)
    print(specificFactors)
    modelEFA = efa.EFA(X)
    chi2Calc, chi2Tab = modelEFA.calcBartlettTest(factorLoadingsFA, specificFactors)
    print(chi2Calc, chi2Tab)

    if np.isnan(chi2Calc) or np.isnan(chi2Tab):
        break
    if chi2Tab < chi2TabMin:
        chi2TabMin = chi2Tab
        noOfSignificantFactors = k

print('The number of significant factors: ', noOfSignificantFactors)

# create a model with the number of significant factors determined
fitModelFA = fa.FactorAnalyzer(n_factors=noOfSignificantFactors)
fitModelFA.fit(X_df)
factorLoadingsEFA = fitModelFA.loadings_
eigenvalues = fitModelFA.get_eigenvalues()
print('The eigenvalues: ', eigenvalues)

# create the graphic of eigenvalues for initial model
g.principalComponents(eigenValues=eigenvalues[0], title='The eigenvalues of the initial model')
# create the graphic of eigenvalues for the model built from factors
g.principalComponents(eigenValues=eigenvalues[1], title='The eigenvalues of the model based on the extracted factors')

# create the correlogram of factor loadings
factors = ['F'+str(j+1) for j in range(noOfSignificantFactors)]
factorLoadingsEFA_df = pd.DataFrame(data=factorLoadingsEFA, index=variables, columns=factors)
g.correlogram(matrix=factorLoadingsEFA_df, title='Correlogram of factor loadings')
# save the factor loadings
factorLoadingsEFA_df.to_csv('./dataOUT/FactorLoadingsEFA.csv')

EFAmodel = efa.EFA(X)
# quality of observations representation
qualityObs = EFAmodel.getQualityObs()
# save the quality of observations representation into a CSV file
qualityObsEFA_df = pd.DataFrame(data=qualityObs, index=observations, columns=['F'+str(j+1) for j in range(X.shape[1])])
qualityObsEFA_df.to_csv('./dataOUT/QualityObsEFA.csv')
g.correlogram(matrix=qualityObsEFA_df, title='Correlogram of quality of observations representation')

# observation contributions to the axes variance
contribObs = EFAmodel.getContribObs()
# save the obs contrib into a CSV file
contribObsEFA_df = pd.DataFrame(data=contribObs, index=observations, columns=['F'+str(j+1) for j in range(X.shape[1])])
contribObsEFA_df.to_csv('./dataOUT/ContribObsEFA.csv')
g.correlogram(matrix=contribObsEFA_df, title='Observation contributions to the axes variance')

# display all graphs
g.show()
