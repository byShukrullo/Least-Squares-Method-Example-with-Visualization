# Least-Squares-Method-Example-with-Visualization
The Least Squares Method finds the line  of best fit by minimizing the sum of  squared differences between observed  and predicted values. It calculates  optimal slope and intercept parameters that reduce overall error, making it fundamental for regression analysis and data modeling in statistics.
# Least Squares Method

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate example data with noise
np.random.seed(42)
x = np.linspace(0, 10, 20)
y_true = 2 * x + 5
noise = np.random.normal(0, 2, size=len(x))
y = y_true + noise

# Manual calculation of least squares
def least_squares_manual(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = sum((x_i - x_mean) * (y_i - y_mean) for x_i, y_i in zip(x, y))
    denominator = sum((x_i - x_mean)**2 for x_i in x)
    slope = numerator / denominator
    
    intercept = y_mean - slope * x_mean
    
    y_pred = slope * x + intercept
    ss_total = sum((y_i - y_mean)**2 for y_i in y)
    ss_residual = sum((y_i - y_pred_i)**2 for y_i, y_pred_i in zip(y, y_pred))
    r_squared = 1 - (ss_residual / ss_total)
    
    return slope, intercept, r_squared

# Calculate using our manual method
slope_manual, intercept_manual, r_squared_manual = least_squares_manual(x, y)

# Calculate residuals
y_pred = slope_manual * x + intercept_manual
residuals = y - y_pred

# Calculate sum of squared residuals
ssr = np.sum(residuals**2)

# Visualization
plt.figure(figsize=(12, 10))

# Plot 1: Original data and fitted line
plt.subplot(2, 2, 1)
plt.scatter(x, y, color='blue', label='Observed data')
plt.plot(x, y_true, 'g--', label='True relationship: y = 2x + 5')
plt.plot(x, y_pred, 'r-', label=f'Fitted line: y = {slope_manual:.4f}x + {intercept_manual:.4f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Least Squares Linear Regression')
plt.legend()
plt.grid(True)

# Plot 2: Residuals
plt.subplot(2, 2, 2)
plt.scatter(x, residuals, color='red')
plt.axhline(y=0, color='black', linestyle='-')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.grid(True)

# Plot 3: Residuals histogram
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=10, color='green', alpha=0.7)
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.grid(True)

# Plot 4: Visualization of squared residuals
plt.subplot(2, 2, 4)
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'r-', alpha=0.5)
    
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, 'k-', label='Regression line')

for i in range(len(x)):
    x_coord = x[i]
    y_coord = y[i]
    y_pred_i = y_pred[i]
    plt.plot([x_coord, x_coord, x_coord+abs(residuals[i]), x_coord+abs(residuals[i]), x_coord], 
             [y_coord, y_pred_i, y_pred_i, y_coord, y_coord], 
             'g-', alpha=0.3)

plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Visualization of Squared Residuals (SSR = {ssr:.2f})')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('least_squares_visualization.png')
plt.show()
```
