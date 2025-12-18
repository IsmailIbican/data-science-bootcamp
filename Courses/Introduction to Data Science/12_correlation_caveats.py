import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation kullanacağımız zaman bodoslama kullanamayız

# İlk başta datamızı visualize etmeliyiz

# Non-linear relationship'te correlation değerleri küçük olur ve linear bir şekilde ilerlemez bizim scatterplotımız

msleep= pd.read_csv("mammal_sleep_data.csv")

sns.scatterplot(x = "bodywt", y="awake", data = msleep)

plt.show()

msleep['bodywt'].hist()

plt.show()

# Örneğe baktığımızda bizim bodyweight tablosunun distribution'ı highly skewed'dur 

# Eğer ki baktık bizim değerimiz sadece belli bir tarafa baskın o zaman biz log transformation kullanacağız

# İlk adım (log_bodywt column'u oluştur) : 

msleep['log_bodywt'] = np.log(msleep['bodywt'])

# Biz eğer ki bunun scatterplot değerini alırsak daha çok linear gözükecektir ve bunun üstünden bir işlem yapabiliriz

sns.lmplot(x= "log_bodywt", y="awake", data=msleep, ci=None)

plt.show()

# Ve biz burdan correlationlarını bulmak istersek eğer daha accurate bir değer alırız 

print(msleep["log_bodywt"].corr(msleep['awake']))

""" 
There's more way to transform our highly skewed data to make relationship more linear

Ways : 

Log transformation (log(x))
Square root transformation(sqrt(x))
Reciprocal transformation(1/x)

Each way can be used in different situations
"""

""" 
Peki ya neden transformation kullanıyoruz : 

Genel olarak istatistikte bizim değişkenlerimiz linear bir relationshipe sahiptir

Mesela bunlardan bazılarına aslında şunları örnek verebiliriz : 

Correlation coefficient
Linear Regression -> Bu tarz konular bizim değişkenlerimizin ya da verilerimizi linear olarak istemekte o yüzden transformation kullanırız
"""


# Correalation bir neden sonuç ilişkisi oluşturmaz verilerimiz arasında sadece bağlantı olup olmadığını söyler 

# Mesela x ile y correlated ise bu bunların arasında bir neden sonuç ilişkisi var anlamına gelmez

# Confounding : 

# Aslında confounding bir nevi gizli bir elemandır 

# Mesela kahve içme ile akciğer kanserinin bir bağlantısı olsun diyelim 

# Buradaki gizli cofounder aslında sigara içmektir sigara içmekte kahve içmeyle alakalı olsun ve akciğer kanserine sebebiyet versin

# Bu tarz correlationlara biz spurious correlation deriz


