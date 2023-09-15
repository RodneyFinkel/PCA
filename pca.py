import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets


class PCA:
    def fit_transform(self, X, n_components =2):
        pass
    
    def standardize_data(self, X):
        # subtract mean and divide by standard deviation columnwise
        numerator = X - np.mean(X, axis=0)
        denominator = np.std(X, axis=0)
        return numerator/denominator
        
    
    def get_covariance_matrix(self):
        # calculate co-variance matrix with standardized matrix A
        C = np.dot(self.A.T, self.A)/(self.n_samples-ddof)
        return C
        
        
    
    def get_eigienvectors(self, C):
        pass
    
    def project_matrix(self, eigenvectors):
        pass
    
    
# Testing
if __name__ == "__main__":
    #load iris data set
    iris_dataset = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    # instantiate and fit_transform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X, n_components=2)
    
    #plot results
    fig, ex = plt.subplots(1, 1, figsize=(10, 6))
    
    sns.scatterplot(
        x = X_pca[:,0],
        y = X_pca[:,1],
        hue = y
    )
    
    ax.set_title('Iris Dataset')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    
    sns.despine()
    plt.show
    