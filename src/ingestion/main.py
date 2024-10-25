import pandas as pd
from ucimlrepo import fetch_ucirepo 
import os
# downloading repo
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features
y =  breast_cancer_wisconsin_diagnostic.data.targets
data = pd.concat([X,y],axis=1)
print(X.shape,data.shape)

#saving repo
os.chdir('data')
print(os.getcwd())
data.to_csv('data.csv')