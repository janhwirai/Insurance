#Predict if a person would buy life insurance based on his age using logistic regression
#Binary logistic regression problem as there are only two outcomes he/she does or does'nt buy insurance
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\AI\insurance_data.csv')

#split train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[['age','affordibility']],df.bought_insurance,test_size=0.2, random_state=25)

X_train_scaled=X_train.copy()
X_train_scaled['age']=X_train_scaled['age']/100
X_test_scaled=X_test.copy()
X_test_scaled['age']=X_test_scaled['age']/100

#Model Building:Build a model and see what weights and bias values it comes up with
model=keras.Sequential([
    keras.layers.Dense(1,input_shape=(2,),
                       activation='sigmoid',
                       kernel_initializer='ones',
                       bias_initializer='zeros')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_scaled,y_train,epochs=100)

#Evaluate the model on test set
model.evaluate(X_test_scaled,y_test)
model.predict(X_test_scaled)

#Now get the value of weights and bias from the model
coef,intercept=model.get_weights()

print(model.predict(X_test_scaled))#Predict using tensorflow model
