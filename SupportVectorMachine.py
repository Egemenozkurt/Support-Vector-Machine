#import required libraries
#To import libraries, libraries should be installed to your computer
#Given commands are for windows, if your operating system is not windows, google for your version of command
#python -m pip install pandas
import pandas as pd
#pip install seaborn
import seaborn as sns
#pip install matplotlib
import matplotlib.pyplot as plt
#pip install -U scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv('diabetes.csv')
#if your ide could not load the dataset 
#try
#df = pd.read_csv(r'C:\path\diabetes.csv')


#Declare Target Variable
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

#split data into seperate trainnig and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#import required libraries
from sklearn import svm

#Create a svm Classifier
model = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
model.fit(X_train, y_train)

#Predict the outcome
y_pred = model.predict(X_test)


#Compare the predicted outcome to actual outcome
df1=pd.DataFrame({"Actual Outcome":y_test,"Predicted Outcome":y_pred})
print(df1)

#Classification report
print("Classification report\n", classification_report(y_test, y_pred))

#Confusion matrix
print("Confusion matrix\n",confusion_matrix(y_test,y_pred))

#Accuracy score
print("Accuracy score: ",accuracy_score(y_test,y_pred))