import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.api import qqplot

""" 
Residual properties of a good fit
"""

# Residuals are normally distributed

# The mean of residuals is zero always

fishes = pd.read_csv("fish_data.csv")

bream = fishes[fishes["species"] == "Bream"]

perch = fishes[fishes["species"] == "Perch"]

mdl_bream = ols("mass_g ~ length_cm",
                data = bream).fit()

mdl_perch = ols("mass_g ~ length_cm",
                data = perch).fit()

explanatory_data = pd.DataFrame({"length_cm": np.arange(20,41)})

prediction_data = explanatory_data.assign(
    mass_g = mdl_bream.predict(explanatory_data)
)


sns.regplot(x = "length_cm",
            y= "mass_g",
            data = bream,
            ci = None)

plt.show() # İyi model

sns.regplot(x = "length_cm",
            y="mass_g",
            data = perch,
            ci = None)

plt.show() # Kötü Model


# Datamızı gözlemlemek için bir kaç tane plot çeşidi bulunmaktadır : 

# 1 - Diagnostic Plot : 

# Buradaki mavi çizgi bizim LOWESS trendline.

# Prediction yapmak için iyi değildir 

# Burada residuals ve fitted values bizim x ve y parametrelerimiz 

# Residuals değeri 0'a yakın olan ve 0'ı takip ediyorsa eğer iyi etmiyorsa kötüdür bu model 

# Bu modeli şöyle oluşturuyoruz : 

sns.residplot(x = "length_cm",
              y = "mass_g",
              data = bream,
              lowess= True)

plt.xlabel("Fitted Values")
plt.ylabel("Residuals")

plt.show()


# 2 - QQ Plot : 
 
# QQ plotta da bu çekilen çizgiyi düz bir şekilde takip ediyorsa bu iyi bir modeldir 

# Highest ve lowest residuals değerlerini bulmak için ayrık olan en küçük ve en büyük değere bakmak lazım 

# QQ Plot yapmak için : (Önce from statsmodels.api import qqplot)

qqplot(data = mdl_bream.resid, fit = True, line="45") # Burda resid değeri önemli olduğu için bize .resid fonksiyonu ile modelimiz residual değerleri lazım

plt.show()

# Fit argumanı bize normal bi distribution sağlayacak

# 3 - Scale-Location Plot : 

# Standardized square root residuals ile fitted values değerlerini alır x ve y axise 

# Burdaki büyümeye bakıyoruz eğer ki fitted value artarsa ve çok az da olsa scale location da artarsa bu iyi bir modeldir

# Ancak tam tersi ise kötüdür

# Bu da böyle yapılmaktadır : 

model_norm_residuals_bream = mdl_bream.get_influence().resid_studentized_internal

model_norm_residuals_abs_sqrt_bream = np.sqrt(abs(model_norm_residuals_bream))

sns.regplot(x =mdl_bream.fittedvalues,
            y = model_norm_residuals_abs_sqrt_bream,
            ci = None,
            lowess = True)

plt.xlabel("Fitted Values")
plt.ylabel("Sqer of abs val of stdized residuals")

plt.show()

# Bu model de ise normalized residuals değerlerini çekmemiz gerekiyor o yüzden get_influence() methodunu kullanıyoruz
 
# Sonrasında da resid_studentized_internala erişmek için de bu attirbute'u çağırıyoruz