import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('cars_dataset.csv')
df.rename(columns = {'car' : 'Condition'}, inplace=True)
df.rename(columns = {'buying' : 'Price'}, inplace=True)
df.rename(columns = {'doors' : 'Doors'}, inplace=True)
df.rename(columns = {'maint' : 'Maintenance'}, inplace=True)
df.rename(columns = {'lug_boot' : 'Luggage'}, inplace=True)
df.rename(columns = {'safety' : 'Safety'}, inplace=True)

label = LabelEncoder()
df['Price'] = label.fit_transform(df['Price'])
df['Doors'] = label.fit_transform(df['Doors'])
df['Maintenance'] = label.fit_transform(df['Maintenance'])
df['persons'] = label.fit_transform(df['persons'])
df['Luggage'] = label.fit_transform(df['Luggage'])
df['Safety'] = label.fit_transform(df['Safety'])
df['Condition'] = df['Condition'].replace({'unacc':'Unaccurate','acc':'Accurate','vgood':'Very Good','good':'Good'})

X = df.drop(['Condition'],axis='columns')
y = df['Condition']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2)

model = RandomForestClassifier(n_estimators=142)
print(model.fit(X_train,y_train))

y_pred = model.predict(X_test)
print(y_pred)

print(model.score(X_test,y_test)*100)

with open('CarClassifier.pkl', 'wb') as f:
    pickle.dump(model,f)