import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# Histograms of price_twd_msq with 10 bins, split by the age of each house

sns.displot(x = "price_twd_msq", 
            col='house_age_years',
            data=taiwan_real_estate,
            bin = 10)

plt.show()

# Calculate the mean of price_twd_msq, grouped by house age

mean_price_by_age = taiwan_real_estate.groupby('house_age_years')['price_twd_msq'].mean()

print(mean_price_by_age)

# Create the model, fit it

mdl_price_vs_age = ols('price_twd_msq ~ house_age_years',
                       data=taiwan_real_estate).fit()

print(mdl_price_vs_age.params)
print(mdl_price_vs_age.summary())

# There is negative values out there fix them 

mdl_price_vs_age0 = ols("price_twd_msq ~ house_age_years + 0 ",
                        data = taiwan_real_estate).fit()

print(mdl_price_vs_age0.params)
print(mdl_price_vs_age0.summary())
