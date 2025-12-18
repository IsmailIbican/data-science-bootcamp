import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols # ols -> Ordinary least squares

# Burada bir kaç tane önemli tanım var  : 

""" 
Intercept : The y value at the point when x is zero

Slope : The amount the y value increases if you increase x by one 

Equation : y = intercept + slope * x -> 0,0 noktasından geçen doğru bu linear doğru denklemi

"""

# Linear regression model'i çalıştırmak için  : 

""" 
from statsmodels.formula.api import ols
"""

# Burada şimdi biz modelimiz oluştururken fonksiyonumuz da şunlara dikkat etmemiz lazım : 

# Soldaki verimiz her zaman dependent variable sağdaki iste independent variable olmak zorunda 

swedish_motor_insurance = pd.read_csv("swedish_motor_insurance_data.csv")

mdl_payment_vs_claims = ols("total_payment_sek ~ n_claims",
                            data = swedish_motor_insurance) # osl bir matematiksel yöntemdir bizim daha önceden gördüğümüz trendline çizgisinin kat sayılarını ve sabit değerlerini hesaplamya yaran bir matematiksel ifadedir sadece 

# Son olarak da modeli fitlemek için şunu yaparız : 

mdl_payment_vs_claims = mdl_payment_vs_claims.fit() # Asıl regression kısmı buradadır

# Burada printlerken biz paramterlerini printlememiz gerekiyor çünkü denklemdeki intercept ve slope değerlerini verecek bize o yüzdem ya results.summary() methodu kullanılacak ya da burada variable.params() kullanılacak

print(mdl_payment_vs_claims.params) # Bu sadece slope ve intercept değerlerini gösterecek bize 

print(mdl_payment_vs_claims.summary()) # Bu da bize tüm detayları ile summary'sini verecek 
 
# Burada r^2 değeri bizim için değerli çünkü bizim modelimizin tutarlılık derecesini gösteriyor bir nevi yani x ile bağlantısının ne kadar olup olmadıgını gösteriyor eğer ki 1 e yakınsa çok iyi 0 a yakınsa bağlantı yok negatif ise o kadar kötü ki artık

