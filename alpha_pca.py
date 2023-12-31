import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Download Data
tech_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG']
bank_tickers = ['JPM', 'BAC', 'GS', 'MS']
finance_tickers = ['MA', 'AXP']

tech_data = yf.download(tech_tickers, start='2015-01-01', end='2023-03-01', threads=True)['Adj Close']
finance_data = yf.download(finance_tickers, start='2015-01-01', end='2023-03-01')['Adj Close']
bank_data = yf.download(bank_tickers, start='2015-01-01', end='2023-03-01')['Adj Close']

# Combine data into a single dataframe
data = pd.concat([tech_data, finance_data, bank_data], axis=1)
data.columns = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'JPM', 'BAC', 'GS', 'MS', 'MA', 'AXP']

# Calculate daily returns
returns = data.pct_change().dropna()
# Fit PCA model
pca = PCA(n_components=3)
pca.fit(returns)

# Plot principal components
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
# enumerate creates a tuple of index and pairwise ellements from the principal components that have been zipped into a preceeding tuple
# i identifies the stock with it's new relative position in the new principal axii/principal components
# the scatter plot is a pairwise plot of each stock's variance along a principal component ie x, y are the stock's coordinates in the PCA space
# using pca.components_ consider investigating the explained variance/EV associated with each principal component
for i, (x, y) in enumerate(zip(pca.components_[0], pca.components_[1])): 
    ax.scatter(x, y, marker='o', s=100)
    ax.annotate(returns.columns[i], xy=(x, y), xytext=(x+0.04, y+0.06),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=6, headlength=8, connectionstyle='arc3,rad=0.1'))

ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('Principal Components')
plt.show()

# Calculate alpha factors. This is a projection of each stock's daily returns onto the PCA space by each eigenvector
alpha_factors = pd.DataFrame()
for i in range(3):
    alpha_factors[f'alpha_{i+1}'] = returns.dot(pca.components_[i])
print('alpha_factors:', alpha_factors)
# Add alpha factors to returns data
alpha_returns = pd.concat([alpha_factors, returns], axis=1)
print(alpha_returns)

# Plot scatter matrix of alpha factors: which represents the pairwise relationship between each alpha factor 
# and also a univariate plot along the diagonal of the scatteroplot matrix
sns.set(style='ticks', palette='viridis')
g = sns.pairplot(alpha_factors)
g.fig.set_size_inches(10,10)
plt.show()

# calculate alpha of each stock by regressing (OLSr) it's returns against each alpha factor
# alpha factors are the projection of returns against the principal components/eigenvectors of the covariance matrix
alpha = pd.DataFrame()
for i in range(10):
    model = sm.OLS(alpha_returns.iloc[:, 3+i], alpha_returns.iloc[:, :3])
    results = model.fit()
    alpha[f'alpha_{i+1}'] = results.params # storing the estimated co-efficients in alpha dataframe 
alpha.columns = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'JPM', 'BAC', 'GS', 'MS', 'MA', 'AXP']
print(alpha)                
alpha_mean = alpha.mean(axis=1) # mean of alpha co-efficients across all stocks
alpha_std = alpha.std(axis=1) # higher std of each alpha across stocks indicates greater variablity in alpha's impact, lower implies more consistent behaviour of alpha across the stocks
print(f'Mean alpha: \n{alpha_mean}')
print('='*50)
print(f'Std alpha: \n{alpha_std}')

# Cumulative Returns
alpha_cum_returns = (1 + alpha_factors).cumprod() - 1

# Plot returns
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
ax.plot(alpha_cum_returns)  # alpha_factors has a date column which is used as 0-axis/independent variable
ax.legend(alpha_cum_returns.columns)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Returns')
ax.set_title('Cumulative Returns of Alpha Factors')
plt.show()
    
# Calculate pairwise correlation between all collumns of alpha_returns dataframe. 
# corr() uses Pearson's Correlation Coefficient which is the Covariance between two variables divided by eeach of their standard deviations
# COULD USE STANDARDIZED AND MEAN CENTRED DATA FOR NORMAL COVARTIANCE INSTEAD OF PCC(p) to yield similar results
corr_matrix = alpha_returns.corr()
alpha_corr = corr_matrix.iloc[:3, 3:13] # selecting first 3 rows of corr_matrix and then last 10 coullumns
    
# Plot Heatmap     
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
sns.heatmap(alpha_corr, cmap='coolwarm', annot=True, fmt='.2f', ax=ax)
plt.title('Correlation Matrix between Alpha Factors and Stock Returns')
plt.show() 

# Calculate alpha factor's Sharpe Ratios 
risk_free_rate = 0.2
sharpe_ratio = alpha.mean()/(alpha.std() *np.sqrt(252))
print(f'Sharpe ratio: \n{sharpe_ratio}')
  
# Sharpe Ratios of daily returns
sharpe_ratio2 = (returns.mean()-risk_free_rate)/returns.std()
print(f'Sharpe_ratio2: \n{sharpe_ratio2}')                                                      