import pandas as pd
import os
os.chdir('data')
df = pd.read_csv('data.csv')
# encoding the labels to 0 and 1
df['Diagnosis'] = df['Diagnosis'].map({'M':1,'B':0})
correlation_with_diagnosis = df.corr()['Diagnosis']
# Select features with a high degree of absolute correlation (greater than 0.6)
features_for_logistic_regression = correlation_with_diagnosis[abs(correlation_with_diagnosis) > 0.6].index.tolist()
features_for_logistic_regression.remove('Diagnosis')

# Create a new DataFrame with selected features and 'Diagnosis'
features_df = df[features_for_logistic_regression + ['Diagnosis']]

# Display the new DataFrame
print('Columns to be used for modeling:',features_df.columns)
print('Total number of features:',len(features_df.columns))

# Exporting final dataset
features_df.to_csv('cleaned_data.csv', index=False)

