import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

data = pd.read_csv('FIFA_World_Cup_1558_23.csv')

#dropping things that are either usless(roundid,...), harmful (half time home goals,...) or bothering me

data = data.drop("Half.time.Home.Goals", axis='columns')
data = data.drop("Half.time.Away.Goals", axis='columns')
data = data.drop("Win.conditions", axis='columns')
data = data.drop("RoundID", axis='columns')
data = data.drop("MatchID", axis='columns')
data = data.drop("Home.Team.Initials", axis='columns')
data = data.drop("Away.Team.Initials", axis='columns')
data = data.drop("Unnamed: 0", axis='columns')
data = data.drop("Assistant.1", axis='columns')#maybe relevant?
data = data.drop("Assistant.2", axis='columns')#maybe relevant?
data = data.drop("Stadium", axis='columns')#maybe relevant?
data = data.drop("Stage", axis='columns')#maybe relevant?

import re
remove_lower = lambda text: re.sub('[a-z]', '', text)

#I will remove the lower casing in the referees names to have a better looking file for now, this can be commented
i = -1
for d in data["Referee"]:
    i+=1
    data["Referee"].iloc[i] = remove_lower(d)


#def whowon(htg,atg):
#    if htg > atg:
#        return "1"
#    if htg == atg:
#        return "0"
#    return "-1"

def whowon(htg,atg):#home team, away team
    if htg > atg:
        return 1.0
    if htg == atg:
        return 0.5
    return 0

#adding a winner column
winner = []
i = -1
for h in data["Home.Team.Goals"]:
    i+=1
    a = data["Away.Team.Goals"].iloc[i]
    ww = whowon(h,a)
    winner.append(ww)
data["winner"]=winner

#removing the minutes of the date time because it bothers me and the file looks neater
i = -1
for dt in data["Datetime"]:
    i +=1
    dtt = dt[:-1][:-1][:-1][:-1]
    data["Datetime"].iloc[i] = dtt

#of course these things have to be removed
data = data.drop("Home.Team.Goals", axis='columns')
data = data.drop("Away.Team.Goals", axis='columns')
#I will remove it completly, there is no way to make sense of this when I'm casting to int
data = data.drop("Datetime", axis='columns')#more problems then anything



#I need to convert everything to an int and keep track of this
#things to be converted to ints (TTBC)
ttbc = ["City","Home.Team.Name","Away.Team.Name","Referee"]# other things must be added here If I don't drop them anymore

storage = []
indexstorage = []
for name in ttbc:
    dt = data[name]
    i = 1
    I = -1
    for d in dt:
        I+=1
#        if i == 0:
#            storage.append(d)
#            indexstorage.append(i)
#            i+=1
#            continue
        if d in storage:
            continue
        storage.append(d)
        indexstorage.append(i)
        i+=1

for name in ttbc:
    dt = data[name]
    i= -1
    for d in dt:
        i+=1
        j=-1
        for s in storage:
            j+=1
            if d == s:
#                data[name].iloc[i] = str(indexstorage[j]) + data[name].iloc[i]
                data[name].iloc[i] = indexstorage[j]

def removespaces(string):
    return string.replace(" ", "")
i = -1
for s in storage:
    i+=1
    storage[i] = removespaces(s)
 
# Driver Program
np.savetxt("namesfile.dat", storage, delimiter=" ", newline = "\n", fmt="%s")
np.savetxt("numfile.dat", indexstorage, delimiter=" ", newline = "\n", fmt="%s")

#casting to floats, just because I can
#nstrings = ["Year","Attendance"]
nstrings = data.columns.values.tolist()
print ('nstrings',nstrings)

for nstring in nstrings:
    i = -1
    for h in data[nstring]:
        i+=1
        data[nstring].iloc[i] = np.asarray(h).astype(np.float32)


winnerdata = data[['winner']].copy()
data = data.drop("winner", axis='columns')

data.to_csv('fifaclean.csv',sep='\t')
data = pd.read_csv('fifaclean.csv',sep = '\t',on_bad_lines='skip',skiprows=1)
winnerdata.to_csv('fifawinner.csv',sep='\t')
winnerdata = pd.read_csv('fifawinner.csv',sep = '\t',on_bad_lines='skip',skiprows=1)

# Split the data into features (X) and target (y)
X = data#data.drop('winner', axis="columns")
y = winnerdata#['winner']

X=np.asarray(X).astype('float64')#.reshape((-1,1))
y=np.asarray(y).astype('float64')#.reshape((-1,1))
y = np.transpose(y)
y = y[1]
y = np.transpose(y)
X = np.transpose(X)
X = X[1:]
X = np.transpose(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(units=1164, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1164, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1164, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1164, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1164, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1164, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1164, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1164, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1164, activation='relu', input_dim=X_train.shape[1]))
#model.add(Dense(units=1, activation='sigmoid'))
model.add(Dense(units=1, activation='softmax'))

from keras.optimizers import Adam  # Import the Adam optimizer
from keras.callbacks import LearningRateScheduler
def custom_learning_rate_schedule(epoch):
    return 0.97**epoch * 0.05

lr = .01
custom_optimizer = Adam(learning_rate=lr)
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer = custom_optimizer, metrics=['accuracy'])
# Define a LearningRateScheduler callback
lr_schedule = LearningRateScheduler(custom_learning_rate_schedule)
model.fit(X_train, y_train, epochs=200, batch_size=32, callbacks=[lr_schedule])


model.save('mi_model.h5')
loaded_model = load_model('mi_model.h5')


user_input = [0,2023,3,1,2,1000,4]
user_input= pd.read_csv('user.csv',sep = '\t',skiprows=1,on_bad_lines='skip')
print (')))))))')
print (np.array(user_input))
print (user_input)

user_prediction = loaded_model.predict(user_input)

print("Model Prediction:", user_prediction)

