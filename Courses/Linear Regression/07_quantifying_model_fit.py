import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# Coefficient determination : 

# Bazen r-squared ya da R-squared olarak da belirtilir 

# The proportion of the variance in the responce variable that is predictable from the explanatory variable

# 1 -> perfect fit

# 0 the worst possible fit 

fishes = pd.read_csv("fish_data.csv")

bream = fishes[fishes["species"] == "Bream"]

mdl_bream = ols("mass_g ~ length_cm",
                data = bream).fit()

print(mdl_bream.summary())

# Modelin r-squared değerini bulmak için sadece şunu kullanabiliriz : 

print(mdl_bream.rsquared) # Bu aslında coeff_determinationdır 

# Aslında sadece correlation değerinin karesi alınmış halidir r squared 

coeff_determination = bream["length_cm"].corr(bream["mass_g"]) ** 2
print(coeff_determination)

# Residual standart error (RSE)

# Difference between a predicted value and an observed value

# It has the same unit as the response variable

# MSE  = RSE ** 2

# Summary method RSE değerini içermiyor ancak biz direkt olarak bunu çağırabiliriz : 

mse = mdl_bream.mse_resid

print(mse) # Bu mean squared error değerini bulmak için kullandığımız method

# RSE için : 

rse = np.sqrt(mse)

print(rse)

# RSE değerini kendi başımıza hesaplamak biraz karışık ancak şöyle yapılıyor :

residual_sqrt = mdl_bream.resid ** 2

print("residual sq : \n", residual_sqrt)

resid_sum_of_sq = sum(residual_sqrt)

deg_freedom = len(bream.index) - 2 

print("deg freedom: ", deg_freedom)

# Degrees of freedom equals the number of obesrvations - numer of model coeffs 

rse1 = np.sqrt(resid_sum_of_sq/deg_freedom)

print(rse1)

# Şu anlama gelmektedir tahmin edilen bream ağırlığı ile gözlemlenen bream ağırlığı arasındaki fark 74 gram

# Diğer error type ise RMSE(root mean squar error) : 

residuals_sq_1 = mdl_bream.resid ** 2

resid_sum_of_sq_1 = sum(residuals_sq_1)

n_obs = len(bream.index)

rmse = np.sqrt(resid_sum_of_sq_1/n_obs)

print("rmse : ", rmse)





