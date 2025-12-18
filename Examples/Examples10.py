import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# Create a new figure, fig
fig = plt.figure()

# Plot the first layer: y = x

sns.axline(xy1 = (0,0), slope = 1, linewidth = 2, color = "green")

# Add scatter plot with linear regression trend line (Regression line is critical)

sns.regplot(x = "returns_2018", y = "returns_2019", data = sb500_yearly_returns,ci=None)

# Set the axes so that the distances along the x and y axes look the same

plt.axis("equal")

plt.show()

# Models the returns

mdl_returns = ols("returns_2019  ~ returns_2018", data = sb500_yearly_returns).fit()

# Create a DataFrame with return_2018 at -1, 0, and 1 

explanatory_data = pd.DataFrame({"returns_2018" : [-1,0,1]})

# Use mdl_returns to predict with explanatory_data

print(mdl_returns.predict(explanatory_data))