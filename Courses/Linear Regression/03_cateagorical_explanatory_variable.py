import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.formula.api import ols

fishes = pd.read_csv('fish_data.csv')

# Bu dataFrame'de mass bizim response variableimiz

# Balıklar bir tür olduğu için kategorik olarak incelememiz gerekiyor bu yüzden de scatterplot hiç bir işe yaramaz burada

# Visualizing 1 numeric and 1 categorical variable

sns.displot(x = "mass_g", col = "species", data= fishes, col_wrap=2, bins = 9) # col_wrap parametresi şu işe yarar -> Mesela 2 tane kategoriyi ele aldıktan sonra alt satıra geç sadece bu işe yarar

plt.show()

# Summary statistics for species : 

summary_stats = fishes.groupby('species')['mass_g'].mean()
print(summary_stats)


# Fitting the model 

summary_stats_models = ols("mass_g ~ species",
                           data=fishes)

summary_stats_models = summary_stats_models.fit()

print(summary_stats_models.summary())
print(summary_stats_models.params) # Burada 4 tane coefficient bulduk çünkü bunlar kategori ve her birinin ayrı grafiği var ve biz her ayrı grafik için farklı linear line'lar çektik

# Sonuçta gördüğümüz gibi bizim beam species ortalıkta yok ama mean değerine bakıp intercept ile karşılaştırırsak aslında bizim beam species değerimiz bizim interceptimiz

# Burada aslında bir kafa karışıklığı da var coefficient negatif bizim balığımızın ağırlığı negatif olamaz bu yüzden biz bir değişikliğe gidiyoruz formulumuz de

mdl_mass_vs_species = ols("mass_g ~ species + 0", # Ols -> 1. dereceden bir bilinmeyenli denklem
                          data=fishes).fit() # Burda yazmış olduğumuz + 0 bizim bir limit sınır noktamız bulacağımız coefficient değerlerimizin 0'dan büyük olması gerektiğini söylüyor

# Burdaki işlem de bir yandan şunu temsil ediyor biz değerlerimizi intercept değerinden çıkartıyoruz ve bizim intercept değerimiz burada kayboluyor ancak ve ancak elimize pozitif değerler geçiyor

print(mdl_mass_vs_species.params)
