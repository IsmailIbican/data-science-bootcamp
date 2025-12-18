import pandas as pd 
import numpy as np
from scipy.stats import expon

# Exponential distribution : 

""" 
Probability of time between Poisson events

We can use them on this way : 

Probability of > 1 day between adoption (yani adoptionlar arasında 1 günden fazla olma olasılığı)

Probability of < 10 minutes between restaurant arrivals (Restorana gelen her iki müşteri arasındaki farkın 10dan küçük olması)

"""

# Exponential distribution'da poisson distribution gibi lambda değerini kullanır

# Zamanı gösterir

# Expected value değeri ise 1/ lambda'dır exponential distribution'ın 


# Şimdi olayımız şu olsun bir tane customer request şirketi customerlara bilet kesiyor her 2 dakika içinde 

# Sorumuz da şu olsun bir customer'ın 1 dakikadan daha az bir süre beklemeyle ticket alma olasılığı nedir

expon.cdf(1, scale= 2) # İlk değer -> istatistiğini bulmak istediğimiz sayı,değer İkinci değer -> scale yani bizim büyüme oranımız yani ortalama bekleme ya da yapma süremiz

# Scale şöyle hesaplanır 1 / lambda bu soruda ki lambda değerimiz 0.5 imiş çünkü elimizde ki scale değerimiz 2'dir

# Peki ya 4 dakikadan fazla bekleme ihtimalimiz nedir : 

1 - expon.cdf(4, scale = 2)

# Ve interval yani bir aralık belirtir ise sorunumuz : (Soru da şu 1 ile 4 dakika arasında ki bekleme olasılığımız)

expon.cdf(4, scale = 2) - expon.cdf(1, scale = 2)

# t- distribution ya da (Student's t-distribution) : 

# t-distribution grafiğinin ya da distribution'ının 'Degrees of Freedom' diye bir parametresi vardır ve normal distribution ile arasındaki farkı oluşturur.

# Bu fark t-distribution grafiğinin peak noktasının daha aşağıda ve kuyruklarının daha kalın yani daha üstte olmasını sağlar 

# Biz 'Degrees of Freedom' paramteresini df ile gösteririz

# Lower df -> Thicker tails, higher standart deviation

# Higher df = closer to normal distribution 


# Log-normal distribution : 

""" 
Variable whose logarithm is normally distributed

Log-normal distribution grafikleri skew şeklindedir 

Examples of Log-normal distribution : 

Length of chess games
Adult blood pressure
Number of hospitalization in the 2003 SARS outbreak
"""



