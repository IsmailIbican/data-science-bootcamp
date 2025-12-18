import pandas as pd
import numpy as np
from scipy.stats import iqr
import statistics
import matplotlib.pyplot as plt
from scipy.stats import uniform

#Burada discrete distribution gibi barlar şeklinde değilde bizim grafiklerimiz düz bir çizgi şeklinde olacak

# Biz buna continous uniform distribution diyoruz

# Probability hala bizim altta kalan alanımızdır

# Uniform distribution in python :

# Şimdiki senaryomuz şu olsun bir otobüs durağından 12 dakikada bir otobüs geçiyor ve geçtiği saatleri bilmiyoruz 7 ya da daha az süre beklersek otobüse binme olasılığımız nedir onu bulacağız

uniform.cdf(7, 0, 12) # İlk baştaki değer bizim bekleme süremiz diğer iki parametre ise alt sınır ile üst sınır 0 dakika da bekleyebiliriz 12 dakika da 

# Eğer ki 7 dakikadan fazla bekleme olasılığımıza bakacak olursak :

1 - uniform.cdf(7,0,12) # 1 den çıkarma sebebimiz ikisinin toplamı bize 1 yani kesin olasılığı verecek

# Peki ya 4 dakika ile 7 dakika arasında otobüsün gelme olasılığını bulalım : 

# Büyük sınırdan küçük sınırı çıkartıyoruz çünkü 7 dakika bekleyip binme olasılığımız daha fazla

uniform.cdf(7,0,12) - uniform.cdf(4,0,12)



# Generating random numbers according to uniform distribution

uniform.rvs(0,5, size = 10) # min değer, max değer, size = kaç tane sayı oluşturmak istiyoruz

# Continous variables'da bazı değerler daha fazla ihtimale sahip olabilirler ve grafikleri daha şekilli olabilir