import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.formula.api import ols
from statsmodels.formula.api import logit

bank = pd.read_csv("bank_churn_data.csv")

# has_churned -> response variable 

# time_since_first_purchase -> length of relationship

# time_since_last_purchase ->  recency of activity (Explanatory variable)


# Churn vs. recency : a linear model 

mdl_churn_vs_recency_lm = ols("has_churned ~ time_since_last_purchase",
                              data = bank).fit()

intercept, slope = mdl_churn_vs_recency_lm.params

print(mdl_churn_vs_recency_lm.params)

# Visualizing data : 

sns.scatterplot(x="time_since_last_purchase",
                y = "has_churned",
                data = bank)

# Biz datamızı limitli bırakmamak için burada regplot yerine ax line kullanıcaz

plt.axline(xy1 = (0, intercept),
           slope = slope)

# Gördüğümüz üzere burda sıkıntılar var linear regression ile biz bunu daha iyi değerlerndirmek için veri setimizi yakınlaştırıcaz : 

# Görüyoruz ki linear regression bize - probability veriyor yabi hata alıyorz biz burada şu an 

plt.xlim(-10,10)
plt.ylim(-0.2,-0.2)

plt.show()

# Peki ya neden logistic regression : 

# Linear modelin generalized halidir 

# Response variable logical olduğunda kullanılır

# Response variable s shaped şeklindedir grafikte


# Logistic regression için bize şu gerekir : 

# from statsmodels.formula.api import logit

mdl_churn_vs_recency_logit = logit("has_churned ~ time_since_last_purchase",
                                   data = bank).fit()

print(mdl_churn_vs_recency_logit.params) # Gene 2 coefficient alıcaz sonuç olarak 

# Visualizing the logistic model : 

# Burada parametre olarak ufak bir değişikliğe gideceğiz : 

sns.regplot(x = "time_since_last_purchase",
            y = "has_churned",
            data = bank,
            ci = None,
            logistic = True)

# Eğerki logistic argumanını biz True yaparsak logistic trend line ekleyecektir

# Şimdi bir de axline çizelim : 

plt.axline(xy1=(0,intercept),
           slope = slope,
           color = "black")

plt.show()

# Burda gördüğümüz mavi çizgi hafif bir şekilde curve almış logistic regression ile linear regression arasındaki fark budur aslında

# Peki ya biz bunu zoomlarsak ne olur ? : 

sns.regplot(x = "time_since_last_purchase",
            y = "has_churned",
            data = bank,
            ci = None,
            logistic = True)

plt.axline(xy1 = (0,intercept),
           slope = slope,
           color = "black")

plt.xlim(-20,20)
plt.ylim(0,1)

plt.show()


