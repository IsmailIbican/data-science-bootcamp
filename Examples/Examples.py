import pandas as pd
import numpy as np
from scipy.stats import iqr


# Example of variance std hist and .agg

# Print variance and sd of co2_emission for each food category 
print(food_consumption.groupby('food_category')['co2_emission']).agg([np.var, np.std], ddof= 1) # groupby methodu bir sütundaki aynı isimli string ya da charları aynı liste altında toplamaya yarayan bir methoddur bir fonksiyon değil (Burada gruplandırma yaparken parantez kullanırız)

# Create histogram of co2_emission for food_category 'beef'

food_consumption[food_consumption['food_category'] == 'beef'['co2_emission']].hist()
plt.show()
plt.figure()

# Create histogram of co2_emission for food_category 'eggs'

food_consumption[food_consumption['food_category'] == 'eggs'['co2_emission']].hist() # sütun belirtirken biz zorunlu olarak burada bir daha tabloyu çağırıp köşeli parantez ile yapmalıyız

plt.show()
plt.figure()


# Example of quartiles and so on 

# Calculate the quartiles of co2_emission (4 eşit parçaya böl demek istiyor burada)

np.quantile(food_consumption['co2_emission'], [0,1, 5]) # Bu kısa yolu uzun yol MeasuringSpreads.py  dosyasında var

# Calculate the quintiles of co2_emission

np.quantile(food_consumption['co2_emission'], [0, 1, 6])

# Calculate the deciles of co2_emission

np.quantile(food_consumption['co2_emission', [0,1, 11]])

# Examples of IQR

# Calculate total co2_emission per country: emissions_by_country 

# Bizde burdan total co2_emission istiyor yani .sum() operation kullanıcaz ve sum operation her zaman en ssonda kullanılır

emissions_by_country = food_consumption.groupby('country')['co2_emission'].sum()
print(emissions_by_country)

# Compute the first and third quartiles and IQR of emissions_by_country
q1 = np.quantile(emissions_by_country, 0.25) # 2. çeyrek
q3 = np.quantile(emissions_by_country, 0.75) # 3. çeyrek
iqr = q3 - q1 # height of plotbox

# Calculate the lower and upper cutoffs for outliers

# Burada sınırları bulacagız

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr # np.quantile ile de yapılabilir

# Subset emissions_by_country to find outliers

outliers =  emissions_by_country[(emissions_by_country < lower) | (emissions_by_country > upper)] # Burada parantezler karışıklığı önlemek için önemli boolean işlem yaptıgımız için

print(outliers)

