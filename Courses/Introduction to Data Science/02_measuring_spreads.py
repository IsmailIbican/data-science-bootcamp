import numpy as np 
import pandas as pd 
import statistics 
from scipy.stats import iqr

# Calculating Variance

# 1- Subtract mean from each data point 

msleep = pd.read_csv("mammal_sleep_data.csv")

dists = msleep['awake'] - np.mean(msleep['awake']) # msleep veri tabanındaki awake sütunundaki değerlerin medyanını her bir awake değerinden çıkar demek (dist değişkeni birden fazla değer tutmakta burada sütun görevi görmekte)

print(dists)

# 2- Square each distance

sqr_dists = dists ** 2

print(sqr_dists)

# 3- Sum squared distances 

sum_sqr_dists = np.sum(sqr_dists)

print(sum_sqr_dists)

# 4- Divide by number of data points - 1

variance = sum_sqr_dists / (83 - 1) # buradaki 83 değişebilir kaç tane değer var ise eğer o değerlerinin sayısından 1 çıkarıyoruz

print(variance)

# IF variance is high data is more spreaded out

# OR

# We can calculate variance like this way

np.var(msleep['awake'], ddof= 1) # Eğer ki ddof=1 olarak almazsak formül değişir ve biraz daha farklı bir sonuç elde ederiz(Population variance hesaplanır sample variance değil)

# Calculationg Standart Deviation 

np.sqrt(np.var(msleep['awake'], ddof= 1)) # Daha uzun aralıklar

# OR

np.std(msleep['awake'], ddof=1)

# Calculating Mean Absolute Deviation

dists = msleep['awake']- np.mean(msleep['awake']) # Daha kısa aralıklar

np.mean(np.abs(dists))

 
# Finding Quantiles

# Quantiles demek aslında bir sütunu veriyi belli başlı parçalara aralıklara bölmek demektir

# Aslında yüzdelik dilimlere bölmek gibi verileri

np.quantile(msleep['awake'], 0.5) # Bu aslında şu anlama gelir awake columunu al ve ikiye böl bunları eşit olacak şekilde yarısı x saatten az uyanık kalırken diger yarısı da x saatten cok uyanık kalır gibi gibi

# 0.5 quantile = median

# Finding Quartiles

np.quantile(msleep['awake'], [0, 0.25, 0.5, 0.75, 1]) # Burada verileri %25lik dilimlere böldük ve iki aralık arasında değerlendiriyoruz

# Another way to make quantiles 

# np.linspace(start, stop ,number of quantile)

np.linspace(msleep['awake'], [0, 1, 5])

# Interquartile range (IQR)

# IQR represents boxplots 

# Eğer ki lower outlier bulmak istiyorsak 0.25 upper bulmak istiyorsak 0.75

# Height of the box in a boxplot

# There is 2 way to represent IQR

np.quantile(msleep['awake'], 0.75) - np.quantile(msleep['awake'], 0.25)

# Or calling iqr from scipy.stats

iqr(msleep['awake'])

# How to find outliers 

iqr = iqr(msleep['awake'])

lower_threshold = np.quantile(msleep['awake'], 0.25)- 1.5 * iqr
higher_threshold =  np.quantile(msleep['awake'], 0.75)+ 1.5 * iqr 

msleep[(msleep['awake']) < lower_threshold | (msleep['awake']) > higher_threshold] # Bu tabloda awake değerleri bu eşitsizlikleri sağlayan değerleri göseter

# All in one 

msleep['awake'].describe()