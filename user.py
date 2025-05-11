import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

loaded_model = load_model('mi_model.h5')

storage = np.genfromtxt('namesfile.dat',delimiter = " ",dtype=None)
indexstorage = np.genfromtxt('numfile.dat',delimiter = " ")

year = float(input('Year -> '))
city = input('City -> ')
team1 = input('Home team name -> ')
team2 = input('Away team name -> ')
attendance = float(input('Attendance -> '))
referee = input('Referee -> ')

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


#toconvert
tc = [city,team1,team2,referee]

I = -1
for tc_ in tc:
    I+=1
    i = -1
    d = -1
    nameindex = -1
    for s in storage:
        i+=1
        d_ = similar(tc_,s)
        if d_>d:
            d = d_
            nameindex = indexstorage[i]
    print (tc[I],'corresponds to ',nameindex)
    tc[I] = nameindex
   


user_input_ = [year,tc[0],tc[1],tc[2],attendance,tc[3]]

#user_input= pd.read_csv('user.csv',sep = '\t',skiprows=1,on_bad_lines='skip')
user_input = [user_input_,user_input_]
print (np.array(user_input_))
print (user_input_)

user_prediction = loaded_model.predict(user_input)

print("Model Prediction:", user_prediction)

def whowon(up):
    if up > 0.55:
        print (tc[1])
        return 'done'
    if up < 0.45:
        print (tc[2])
        return 'done'
    print ('even sleven')
    return 'done'

print ('the winner is')
print (whowon(user_prediction[0][0]))
    
