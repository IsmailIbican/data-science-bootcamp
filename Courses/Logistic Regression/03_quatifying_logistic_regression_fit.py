import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from statsmodels.formula.api import logit
from statsmodels.graphics.mosaicplot import mosaic  

# Diagnostic plotslar logistic bir alanda işlevsizdirler

# Logical response variable'da 4 tane possible outcome vardır

# Her bir outcome değerinin sayılmasına bir confusion matrix adını veriyoruz

# Counts of model outcomes için bize biraz data manipulation gerek

churn = pd.read_csv("bank_churn_data.csv")

churn.columns = churn.columns.str.strip()

print(churn.columns) # Sütunların kontrolü

mdl_recency = logit("has_churned ~ time_since_last_purchase",
                    data = churn).fit()

explanatory_data = pd.DataFrame({"time_since_last_purchase" : np.arange(-1,6.25,0.25)})

prediction_data = explanatory_data.assign(
    has_churned = mdl_recency.predict(explanatory_data)
)

actual_response = churn["has_churned"]

predicted_response = np.round(mdl_recency.predict()) # bu predicted values aslında probabilitylerdir olma ya da olmama probabilityssi biz bunları 0 ya da 1 e yuvarlıyoruz 0 ise True 1 ise False(most_likely_outcome)

# Sonrasında biz bunları combinelıyoruz : 

outcomes = pd.DataFrame({"actual_response" : actual_response,
                         "predicted_response " : predicted_response})

# Bunları countlamak için de .value_counts methodunu kullanıyoruz 

print(outcomes.value_counts(sort = False))

# Biz buna confusion matrix diyoruz


# Visualizing the confusion matrix : 

conf_matrix = mdl_recency.pred_table()

print(conf_matrix)

# There is a way to plot confusion matrix to show it on a plot : 

""" 
from statsmodels.graphics.mosaicplot import mosaic
"""

mosaic(conf_matrix)

plt.show()

# Accuracy : 

# Accuracy is the proportion of correct predictions

""" 
Accuracy : TN + TP / TN + FN + FP + TP
"""

TN = conf_matrix[0,0]
FP = conf_matrix[0,1]
FN = conf_matrix[1,0]
TP = conf_matrix[1,1]

accuracy = TN + TP / TN + FN + TP + FP

print(accuracy)

# Sensitivitiy is the proportion of true positive : 

""" 
sensitivity = TP / FN + TP
"""

# Specificity : 

# Specificity is the proportion of true negative : 

""" 
specificity  = TN / TN + FP
"""