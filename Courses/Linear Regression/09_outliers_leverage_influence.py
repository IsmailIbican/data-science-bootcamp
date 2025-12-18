import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.api import qqplot 


fish = pd.read_csv("fish_data.csv")

roach = fish[fish["species"] == "Roach"]

print(roach.head())

# Hangi noktalar bizim outlier'ımız peki ? 

sns.regplot(x = "length_cm",
            y = "mass_g",
            ci= None,
            data = roach)

plt.show()

# İlk outlier tipimiz -> Extreme explanatory values : 

roach["extreme_l"] = ((roach["length_cm"] < 15) | (roach["length_cm"] > 26))

fig = plt.figure()

sns.regplot(x = "length_cm",
            y = "mass_g",
            data = roach,
            ci = None)

sns.scatterplot(x = "length_cm",
                y="mass_g",
                hue = "extreme_l",
                data = roach)

plt.show()

# Response değerlerin regresyon çizgisine olan uzaklıkları

roach["extreme_m"] = roach["mass_g"] <  1

fig = plt.figure()

sns.regplot(x = "length_cm",
            y = "mass_g",
            data = roach,
            ci = None)

sns.scatterplot(x = "length_cm",
                y = "mass_g",
                hue = "extreme_l",
                style = "extreme_m",
                data = roach)

plt.show()

# Leverage and Influence : 

# Leverage -> Explanatory değerlerinin ne kadar ekstrem olduğunu ölçmeye yarar (Kısacası ekstrem değerin regresyon çizgisini etkileme büyüklüğünü ölçer)

# Influence -> Eğer modelleme yaparken gözlemlemeyi datasetin dışında bırakırsak modelin ne kadar değişeceğini ölçer

# Influence bir gözlemin hem leverage hem de residual bilgilerini kullanarak model üzerindeki gerçek etkisin ölcer



# .get_influence() and .summary_frame()

mdl_roach = ols("mass_g ~ length_cm",
                data = roach).fit()

summary_roach = mdl_roach.get_influence().summary_frame()

roach["leverage"] = summary_roach["hat_diag"]

# Bizim leverage değerlerimiz summar_frame dataFrameinde hat diag column da saklanmaktadır o yüzden biz bu leverage değerlerine ayrı bir column oluşturduk 

# Cook's distance : 

# Cook's distance influence'u ölçmek için kullanılır

roach['cooks_dist'] = summary_roach["cooks_d"]

print(roach.head())

# Most influential roaches : 

print(roach.sort_values("cooks_dist", ascending=False))

# Removing the most influential roach : 

roach_not_short = roach[roach["length_cm"] != 4.6]

fig = plt.figure()

sns.regplot(x = "length_cm",
            y = "mass_g",
            data = roach,
            ci = None,
            line_kws= {"color" : "green"})


sns.regplot(x = "length_cm",
             y = "mass_g",
             data = roach_not_short,
             ci = None,
             line_kws= {"color" : "red"})

plt.show()