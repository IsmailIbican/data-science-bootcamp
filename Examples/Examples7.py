import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from statsmodels.formula.api import ols

# Create the model object : 

mdl_price_vs_conv = ols('price_twd_msq ~ n_convenience',
                        data = taiwan_real_estate)

# Fit the model : 

mdl_price_vs_conv = mdl_price_vs_conv.fit()

# Print the parameters of fitted model 

print(mdl_price_vs_conv.params)
