import numpy as np
import PCA.PCA as pca
import scipy.stats as sts


class EFA:

    def __init__(self, matrice):  # numpy.ndarray
        self.X = matrice

        pcaModel = pca.PCA(self.X)
        self.Xstd = pcaModel.getXstd()
        self.Corr = pcaModel.getCorr()
        self.EigenValues = pcaModel.getEigenValues()
        self.Scores = pcaModel.getScores()
        self.QualityObs = pcaModel.getQualityObs()
        self.ContribObs = pcaModel.getContribObs()

    def getXstd(self):
        return self.Xstd

    def getValProp(self):
        return self.EigenValues

    def getScores(self):
        return self.Scores

    def getQualityObs(self):
        return self.QualityObs

    def getContribObs(self):
        return self.ContribObs

    def calcBartlettTest(self, loadings, epsilon):
        n = self.X.shape[0]
        m, q = np.shape(loadings)
        print(n, m, q)
        V = self.Corr
        # diagonal matrix of specific factors
        psi = np.diag(epsilon)
        Vestim = loadings @ np.transpose(loadings) + psi
        Iestim = np.linalg.inv(Vestim) @ V
        detIestim = np.linalg.det(Iestim)
        if detIestim > 0:
            traceIestim = np.trace(Iestim)
            chi2Calc = (n - 1 - (2*m - 4*q - 5) / 6) * \
                       (traceIestim - np.log(detIestim) - m)
            numarGradeLibertate = ((m - q)**2 - m - q) / 2
            chi2Tab = 1 - sts.chi2.cdf(chi2Calc, numarGradeLibertate)
        else:
            chi2Calc, chi2Tab = np.nan, np.nan

        return chi2Calc, chi2Tab


