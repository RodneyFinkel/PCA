import yfinance as yf
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Download Data
tech_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG']
bank_tickers = ['JPM', 'BAC', 'GS', 'MS']
finance_tickers = ['MA', 'AXP']

tech_data = yf.download(tech_tickers, start='2015-01-01', end='2023-08-01')['Adj Close']
finance_data = yf.download(bank_tickers, start='2015-01-01', end='2-23-08-01')['Adj Close']
bank_data = yf.download(bank_tickers, start='2015-01-01', end='2023-08-01')['Adj Close']

# Combine data into a single dataframe
data = pd.concat([tech_data, finance_data, bank_data], axis=1)
data.columns = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'JPM', 'BAC', 'GS', 'MS', 'MA', 'AXP']
data

# Calculate daily returns
returns = data.pct_change().dropna()
# Fit PCA model
pca = PCA(n_components=3)
pca.fit(returns)

# Plot principal components
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
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

# Calculate alpha factors
alpha_factors = pd.DataFrame()
for i in range(3):
    alpha_factors[f'alpha_{i+1}'] = returns.dot(pca.components_[i])

# Add alpha factors to returns data
alpha_returns = pd.concat([alpha_factors, returns], axis=1)

# Plot scatter matrix of alpha factors
sns.set(style='ticks', palette='viridis')
g = sns.pairplot(alpha_factors)
g.fig.set_size_inches(10,10)
plt.show()