import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Burada istatistiksel olarak verilerin birbirleri arasındaki ilişkiler ile ilgileneceğiz

swedish_motor_insurance = pd.read_csv('swedish_motor_insurance_data.csv')

print(swedish_motor_insurance.mean())

print(swedish_motor_insurance['n_claims'].corr(swedish_motor_insurance['total_payment_sek']))

# Correlation şunu diyor burda n_claims artarsa büyük ihtimalle total_payment değeri de artar

# Peki ya regression nedir ? 

# Açıklayıcı ve yanıtlayıcı değişkenleri istatistiksel olarak model ile araştıran bir konudur

# Verilen explanatory variable ile biz predict ederek response variable değerlerini bulabiliriz regression buna yarar

# Jargon : 

# Response  variable : The variable that you want to predict 

# Explanatory variable : The variables that explain how the response variable will change

# 3 tür regression türü vardır : 

# Linear regression : verilerin tahminini numeric olarak yapar

# Logical regression : verilerin tahminini logical olarak yapar(True ya da False değerleri alır)

# Simple linear/logistic regression : burda sadece bir tane explanatory variable var elimizde 

# Regression'a geçmeden önce data setimizi visualize etmek önemli 

sns.scatterplot(x = "n_claims", y ="total_payment_sek", data= swedish_motor_insurance)
plt.show()

# Adding linear line :  price_twd_msq n_convenience

sns.regplot(x = "n_claims", y="total_payment_sek", data=swedish_motor_insurance, ci = None)
plt.show()

# Burada trendline çizgiini linear regression kullanıp hesaplayarak çizdik

# Regression ünitesinde kullanılacak bazı libraryler ve packagelar : 

# statsmodels 

# scikit-learn

# scatter_kws={'alpha': 0.5} -> Bu parametre bizim noktalarımızı transparent yapar