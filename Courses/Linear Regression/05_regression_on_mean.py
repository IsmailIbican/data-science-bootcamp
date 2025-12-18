import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# Response variable = fitted value + residual 

# Response variable = expected_data * slope + intercept

# Extreme cases are often due to randomness

# Regression to mean means extreme casess don't persist over time

height = pd.read_csv("pearson_father_son_data.csv")

# Bi scatterplot üstünden gösterelim

fig = plt.figure()

sns.scatterplot(x = "father_height_cm",
                y = "son_height_cm",
                data = height,
                )

# Ve biz bu scatter plot'ta baba ve oğullarının boyunun aynı olduğu değerlerden geçen bir çizgi çekmek istersek şunu yapıyoruz

plt.axline(xy1=(150,150), # burda biz x 150 ve y 150 noktasını referans olarak alıyoruz ve eğimi 1 veriyoruz yani x = y doğrusu çiziyoruz bu çizgi üstünde baba ve oğulların boyları tamamıyla aynı
           slope = 1,
           linewidth = 2,
           color="green")

# Şimdi bir de regression line ekleyelim : 

sns.regplot(x = "father_height_cm",
            y = "son_height_cm",
            ci = None,
            data = height,
            line_kws={"color" : "black"})

plt.axis("equal")
plt.show()

# Running a regression 

mdl_father_vs_son = ols("son_height_cm ~ father_height_cm",
                        data = height).fit()

print(mdl_father_vs_son.params)

# Making predictions : 

really_tall_father = pd.DataFrame({"father_height_cm": [190]})

print(mdl_father_vs_son.predict(really_tall_father))

son_height_cm = really_tall_father.assign(
    son_height_cm = mdl_father_vs_son.predict(really_tall_father)
)

print(son_height_cm)

print(mdl_father_vs_son.fittedvalues)

really_short_father = pd.DataFrame({"father_height_cm": [150]})

print(mdl_father_vs_son.predict(really_short_father))