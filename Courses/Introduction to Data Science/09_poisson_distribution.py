import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Olayın outcome değeri yani olasılığı belirliymiş gibi görünür ancak hiç bir zaman belirli değildir tamamen randomdur

""" 
Poisson distribution : 

Probability of some # of events occurring over a fixed period time

We can use it in : 

Probability o >= 5 animal adopted from animal shelter per week 

"""

# Lambda ile ifade edilir (belirli bir zaman aralığında gerçekleşen ortalama olay sayısı)

# Lambda bi yandan da expected value olur o distribution da 

# Lambda değiştikçe grafik de değişir

# Şimdi bir tane değerin poission olasılığını bulalım 

# Sorumuz şu hafta ortalama 8 tane hayvan adopt ediliyor ise bir hafta için de 5 tane adopt edilme olasılığı nedir : 

poisson.pmf(5,8) # İlk değer -> istenen sayı İkinci değer -> expected value yani ortalama değer

# Eğer ki 5 ya da daha az hayvan adopt edilme ihtimali sorulursa : 

poisson.cdf(5, 8)

# 5 veya daha fazla ise

1 - poisson.cdf(5,8)

# Sampling from a Poisson distribution : 

print(poisson.rvs(8, size = 10))