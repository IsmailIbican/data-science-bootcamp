import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from statsmodels.formula.api import logit

# The regplot() predictions : 

churn = pd.read_csv("bank_churn_data.csv")

sns.regplot(x = "time_since_last_purchase",
            y = "has_churned",
            data = churn,
            ci = None,
            logistic=True)

plt.show()

# Logistic model ile predict yapmak istersek eğer : 

mdl_recency = logit("has_churned ~ time_since_last_purchase",
                    data = churn).fit()

explanatory_data = pd.DataFrame({"time_since_last_purchase" : np.arange(-1,6.25,0.25)})

prediction_data = explanatory_data.assign(
    has_churned = mdl_recency.predict(explanatory_data) 
) # Burda da tahmin yöntemi kullanarak hesaplanan bir response column ekliyoruz 

prediction_data["most_likely_outcome"] = np.round(prediction_data["has_churned"])

# Bunu göstermek için de şunu kullanıyoruz : 

sns.regplot(x = "time_since_last_purchase",
            y = "has_churned",
            data = churn,
            ci = None,
            logistic=True)

sns.scatterplot(x = "time_since_last_purchase",
                y = "has_churned",
                data = prediction_data,
                color = "red") # scatterplot oluştururken logistic kullanmamıza gerek yoktur

plt.show()

sns.regplot(x = "time_since_last_purchase",
            y = "has_churned",
            data = churn,
            ci = None,
            logistic=True)

sns.scatterplot(x = "time_since_last_purchase",
                y = "most_likely_outcome",
                data = prediction_data,
                color = "red")

plt.show()

# Hesaplama yapmak yerine daha basit bir tahmin, bir yanıtın olasılıklarını hesaplamak, en olası yanıtı hesaplamaktır

# Mesela bir müsteriyi kaybetme olasılığı 0.5 den az ise en olası sonuç müşteri kaybetmemektir

# Mesela olasılık 0.5 in üstündeyse eğer kaybetme ihtimalimiz daha fazla 

# Bunun için şunu ekliyoruz (yukarıda prediction datanın altında hemen)

# Odds ratio : 

""" 
odds_ratio = probability / 1 - probability
"""

# Calculating odds ratio : 

prediction_data["odds_ratio"] = prediction_data["has_churned"] / (1 - prediction_data["has_churned"])

# Visualizing odds ratio : 

sns.lineplot(x = "time_since_last_purchase",
             y = "odds_ratio",
             data = prediction_data)

plt.axhline(y = 1,
            linestyle = "dotted") # Burada çizgili bir çizgi çektik 1 noktasına burada kaybetme ile kaybetmeme olasılığı eşittir

plt.show() # Burada grafiğimiz eğik bir şekildedir 

# Visualizing log odds ratio : 

sns.lineplot(x = "time_since_last_purchase",
             y = "odds_ratio",
             data = prediction_data)

plt.axhline(y = 1,
            linestyle = "dotted") # Burada odds_ratio valuelarının logaritalarını aldığımız için grafik doğrusal bir görünüm kazanır

plt.yscale("log") # Logaritmik bir y ölçeği ekler

plt.show()

# Calculating log odds ratio : 

prediction_data["log_odds_ratio"] = np.log(prediction_data["odds_ratio"])

# Bu işlem aslında bizim kullandığımız logit fonksiyonudur 


