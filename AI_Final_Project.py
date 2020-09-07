# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:27:45 2020

@author: shiva
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
import itertools
import datetime
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# load dataset
dataframe=pd.read_csv('heart.csv', sep=',', header=0)
#performing exploratory data analyis by displaying the correlation and the distribution of data
#(dataframe.head())
#(dataframe.isnull().sum())
#dataframe.describe()
#sns.heatmap(dataframe.corr())   
#sns.distplot(dataframe['age'], color = 'blue')
#plt.title('Distribution of Age', fontsize = 30)
#sns.countplot(x="target", data=dataframe)#, palette="bwr")
#plt.show()



#dataframe.corr()['target'].sort_values()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



#print(dataframe)
x = dataframe.iloc[:,0:13]#the features
y = dataframe.iloc[:,13]#target that we want to classify

encoder = LabelEncoder()
encoder.fit(y)

encodedy = encoder.transform(y)
y = encodedy

xscaler = StandardScaler()
x = xscaler.fit_transform(x)#scaling the data as we don't want to skew the predictions

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#splitting the data into train,test 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25)

ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start_time = datetime.datetime.now()
#creating the model
def create_model():
    model = Sequential()
    model.add(layers.Dense(25, activation='relu', kernel_initializer='random_normal', input_shape = (xtrain.shape[1],)))
    model.add(layers.Dense(25, activation='relu'))#, kernel_initializer='random_normal'))
    #model.add(layers.Dense(25, activation='relu'))#, kernel_initializer='random_normal'))
    #model.add(layers.Dense(25, activation='relu'))
    #model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))#, kernel_initializer='random_normal'))#used to be number of columns in the dataset
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Validate Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

k = 5#for cross validation
num_epochs = 300
samples = len(ytrain) // k 
all_acc_histories = []
all_loss_histories = []
for i in range(k):
    print('Currently processing fold number:', i)
    valdata = xtrain[i * samples: (i+1) * samples]
    valtargets = ytrain[i * samples: (i+1) * samples]
    partial_train_data = np.concatenate([xtrain[:i * samples], 
                                         xtrain[(i + 1) * samples:]], 
                                        axis=0)
    partial_train_targets = np.concatenate([ytrain[:i * samples], 
                                         ytrain[(i + 1) * samples:]], 
                                        axis=0)
    model = create_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(valdata, valtargets), 
                        epochs = num_epochs, 
                        batch_size =80, verbose = 1)
    acc_history = history.history['val_acc']
    all_acc_histories.append(acc_history)
    loss_history = history.history['val_loss']
    all_loss_histories.append(loss_history)

stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)

average_acc_history = [np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)]
average_loss_history = [np.mean([x[i] for x in all_loss_histories]) for i in range(num_epochs)] 

print('average val accuracy:',np.mean(average_acc_history)) 
print('average val loss:',np.mean(average_loss_history )) 
 

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
#plotting the data
plt.plot(epochs, loss,color='red', label='Training loss')
plt.plot(epochs, val_loss, color='blue', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc,color='red' , label='Training acc')
plt.plot(epochs, val_acc,color='blue' , label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Final Model Section
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
#final model
final_model = Sequential()
final_model.add(layers.Dense(50, activation='relu', input_shape = (xtrain.shape[1],)))
final_model.add(layers.Dense(50, activation='relu'))
final_model.add(layers.Dense(2, activation='softmax'))#used to be 7
final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start_time = datetime.datetime.now()
final_model.fit(xtrain,ytrain, epochs = 300, batch_size = 80, verbose = 1)
final_model.summary()
end_time = datetime.datetime.now()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Final Model Output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
test_loss, test_accuracy = model.evaluate(xtest, ytest)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Time required for model:",end_time - start_time)

#to plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset 
pred_label = model.predict(xtest)
# Convert predictions classes to one hot vectors 
pred_label_classes = np.argmax(pred_label, axis = 1) 
# Convert validation observations to one hot vectors
label_true = np.argmax(ytest,axis = 1) 
#print('Class with highest probability: ', label_true)
# compute the confusion matrix
confusion_mtx = confusion_matrix(label_true, pred_label_classes) 
print(confusion_mtx)
# Plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2))


