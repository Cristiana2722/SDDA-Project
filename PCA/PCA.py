import numpy as np

class PCA:

    def __init__(self, matrice):
        self.X = matrice

        # correlation matrix
        self.R = np.corrcoef(self.X, rowvar=False)

        # standardisation
        mean = np.mean(self.X, axis=0)
        std = np.std(self.X, axis=0)
        self.Xstd = (self.X - mean) / std

        # variance-covariance matrix for standardised X
        self.Cov = np.cov(self.Xstd, rowvar=False)

        # eigenvalues and eigenvectors for variance-covariance matrix
        eigenValues, eigenVectors = np.linalg.eigh(self.Cov)
        print(eigenValues)

        # eigenvalues and eigenvectors in descending order
        k_des = [k for k in reversed(np.argsort(eigenValues))]
        print(k_des)
        self.alpha = eigenValues[k_des]
        self.a = eigenVectors[:, k_des]
        print(self.alpha)

        # regularize eigenvectors
        for col in range(self.a.shape[1]):
            minim = np.min(self.a[:, col])
            maxim = np.max(self.a[:, col])
            if np.abs(minim) > np.abs(maxim):
                self.a[:, col] *= -1

        # principal components
        self.C = self.Xstd @ self.a

        # factor loadings matrix
        # represents the correlation between the initial variables and the principal components
        self.Rxc = self.a * np.sqrt(self.alpha)

        # scores (standardized principal components)
        self.scores = self.C / np.sqrt(self.alpha)

        # quality of observations
        C2 = self.C * self.C
        C2sum = np.sum(C2, axis=1)
        self.QualityObs = np.transpose(np.transpose(C2) / C2sum)

        # observation contributions to the axes variance
        self.betha = C2 / (self.alpha * self.X.shape[0])

        # communalities (finding the principal components in the initial variables)
        Rxc2 = self.Rxc * self.Rxc
        self.Commun = np.cumsum(Rxc2, axis=1)


    def getCorr(self):
        return self.R

    def getXstd(self):
        return self.Xstd

    def getEigenValues(self):
        return self.alpha

    def getPrincipalComps(self):
        return self.C

    def getRxc(self):
        return self.Rxc

    def getScores(self):
        return self.scores

    def getQualityObs(self):
        return self.QualityObs

    def getContribObs(self):
        return self.betha

    def getCommun(self):
        return self.Commun