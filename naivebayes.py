import pandas as pd
import numpy as np
crop= pd.read_csv("/content/Crop_recommendation.csv")
crop.head()
crop.isnull().sum()
crop.duplicated().sum()
crop.describe()
import seaborn as sns
sns.heatmap(crop.corr(),annot=True)
crop['label'].value_counts()
crop_dict = {
    'rice':1,     
    'maize':2,        
    'jute':3,         
    'cotton':4,         
    'coconut':5,
    'papaya':6,         
    'orange':7,         
    'apple':8,          
    'muskmelon':9,      
    'watermelon':10,     
    'grapes':11,         
    'mango':12,        
    'banana': 13,         
    'pomegranate':14,    
    'lentil':15,      
    'blackgram':16,      
    'mungbean': 17,       
    'mothbeans': 18,      
    'pigeonpeas': 19,     
    'kidneybeans': 20,    
    'chickpea': 21,       
    'coffee': 22         
}
crop['crop_num']=crop['label'].map(crop_dict)
crop['crop_num'].value_counts()
x = crop.drop('crop_num',axis=1)
y = crop['crop_num']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state =42)
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
sc.fit(x_train)
x_train=sc.transform(x_train)
x_test=sc.transform(x_test)
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics
NaiveBayes = GaussianNB()
NaiveBayes.fit(x_train,y_train)
predicted_values = NaiveBayes.predict(x_test)
x = metrics.accuracy_score(y_test, predicted_values)
print("Naive Bayes's Accuracy is: ", x)
def recommendation(N,P,K,temperature,humidity,ph,rainfall):
  features = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
  prediction = NaiveBayes.predict(features).reshape(1,-1)
  return prediction[0]
N= 20
P = 30
K= 40
temperature = 40.0
humidity = 20
ph = 30
rainfall = 50
predict = recommendation(N,P,K,temperature,humidity,ph,rainfall)
crop_dict = {1:'Rice',2: 'Maize',3:'Jute',4:'Cotton',5:'Coconut',6:'Papaya',7:'Orange',8:'Apple',
             9:'Muskmelon',10:'Watermelon',11:'Grapes',12:'Mango',13:'Banana',
           14:'Pomegranate',15:'Lentil',16:'Blackgram',17:'Mungbean',18:"Mothbeans",
             19:'Pigeonpeas',20:'Kidneybeans',21:'Chickpea',22:'Coffee'}
if predict[0] in crop_dict:
  crop=crop_dict[predict[0]]
  print("{}is a best crop to be cultivated".format(crop))
else:
  print("Sorry not able to recommend a proper crop for this environment")
