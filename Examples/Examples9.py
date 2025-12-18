import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# Create explanatory_data

explanatory_data = pd.DataFrame({'n_convenience' : np.arange(0,11)})

# Print it 

print(explanatory_data)

# Use mdl_price_vs_conv to predict with explanatory_data, call it price_twd_msq

price_twd_msq = mdl_price_vs_conv.predict(explanatory_data)

print(price_twd_msq)

# Create prediction data 

prediction_data = explanatory_data.assign(
    price_twd_msq = mdl_price_vs_msq.predict(explanatory_data)
)

print(prediction_data)

# Create a new figure, fig

fig = plt.figure()

sns.regplot(x = "n_convenience",
            y = "price_twd_msq",
            ci = None,
            data = taiwan_real_estate)

# Add a scatter plot layer to the regplot

sns.scatterplot(x="n_convenience",
                y = "price_twd_msq",
                data = prediction_data,
                color="red",
                marker="s")

plt.show()
