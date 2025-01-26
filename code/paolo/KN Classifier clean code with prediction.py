import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import os

#load cleaned dataset
data = r'C:\Users\paolof\Documents\Personal\AI Master\Applied Machine learning\DS_salaries_regression_classification_ONLY_numerical.csv'
df = pd.read_csv(data)
print (df.columns)

#Separating independant variable and dependent variable("salary_group") and removing the salary_in_usd from the dataset
X = df.drop(['salary_group','salary_in_usd'], axis=1)
y = df['salary_group']

# Split the dataset into training/validation and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the dataset into training and validation subsets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

print("Total number of datapoints: " + str(X.shape[0]))
print("Number of datapoints in the training set: " + str(X_train.shape[0]))
print("Number of datapoints in the validation set: " + str(X_val.shape[0]))
print("Number of datapoints in the test set: " + str(X_test.shape[0]))

#fit the standardscaler on the training data
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("target feature ditricution " + str(y.value_counts()))

"""
# accuracy vs k value distribution with Euclidean distance metric
num_neighbours = np.arange(1,361)
performance = []
for k in num_neighbours:
    clf = KNeighborsClassifier (n_neighbors=k, metric='euclidean')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_val)
    acc=accuracy_score(y_val, y_pred)
    performance.append(acc)

# plot the distribution
plt.plot(num_neighbours, performance)
plt.grid(True)
plt.show()
best_index = np.argmax(performance)
"""
# calculate performance metrics for the best value of k = 91
clf = KNeighborsClassifier (n_neighbors=91, metric='euclidean')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_val)
acc=accuracy_score(y_val, y_pred)
#print ("The performance of the KNN with K=91 is: %.2f"%(acc))

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred, labels=['L', 'M', 'H'])

#precision = precision_score(y_test, y_pred)
classification_metrics = classification_report(y_val, y_pred, target_names=['L', 'M', 'H'])

# Display performance results
#print("Confusion Matrix with K=91:")
#print(conf_matrix)

#print("\nClassification Report with k=91:")
#print(classification_metrics)


# calculate performance metrics for the best value of k = 91 using the test set

y_predt = clf.predict(X_test)
acct=accuracy_score(y_test, y_predt)
print ("The performance of the KNN with K=91 is using the test set: %.2f"%(acct))

# Compute the confusion matrix
conf_matrixt = confusion_matrix(y_test, y_predt, labels=['L', 'M', 'H'])

#precision = precision_score(y_test, y_pred)
classification_metrics_t = classification_report(y_test, y_predt, target_names=['L', 'M', 'H'])

# Display performance results
print("Confusion Matrix with K=91 with the test set:")
print(conf_matrixt)

print("\nClassification Report with k=91 with the test set:")
print(classification_metrics_t)

# map entry to be predicted
    
ymapping =1 # we can set the year as 2022, which has 1 as value, because it is the closest to today
exlemapping=[['EN',0],['MI',0.19],['SE',0.56],['EX',1.0]]
emtymapping=[['CT',1.0],['FL',0.10],['FT',0.52],['PT',0]]
jtmapping=[['3D Computer Vision Researcher',0],['AI Scientist',0.152],['Analytics Engineer',0.424],\
           ['Applied Data Scientist',0.426],['Applied Machine Learning Scientist',0.341],['BI Data Analyst',0.173],\
           ['Big Data Architect',0.235],['Big Data Engineer',0.116],['Business Data Analyst',0.178],\
           ['Cloud Data Engineer',0.298],['Computer Vision Engineer',0.097],['Computer Vision Software Engineer',0.250],\
           ['Data Analyst',0.211],['Data Analytics Engineer',0.148],['Data Analytics Lead',1.000],\
           ['Data Analytics Manager',0.304],['Data Architect',0.431],['Data Engineer',0.261],['Data Engineering Manager',0.294],\
           ['Data Science Consultant',0.160],['Data Science Engineer',0.176],['Data Science Manager',0.382],\
           ['Data Scientist',0.245],['Data Specialist',0.399],['Director of Data Engineering',0.378],\
           ['Director of Data Science',0.474],['ETL Developer',0.123],['Finance Data Analyst',0.141],\
           ['Financial Data Analyst',0.674],['Head of Data',0.387],['Head of Data Science',0.353],['Head of Machine Learning',0.184],\
           ['Lead Data Analyst',0.217],['Lead Data Engineer',0.336],['Lead Data Scientist',0.274],['Lead Machine Learning Engineer',0.206],\
           ['ML Engineer',0.280],['Machine Learning Developer',0.201],['Machine Learning Engineer',0.240],\
           ['Machine Learning Infrastructure Engineer',0.239],['Machine Learning Manager',0.279],['Machine Learning Scientist',0.383],\
           ['Marketing Data Analyst',0.208],['NLP Engineer',0.079],['Principal Data Analyst',0.293],['Principal Data Engineer',0.808],\
           ['Principal Data Scientist',0.525],['Product Data Analyst',0.019],['Research Scientist',0.259],['Staff Data Scientist',0.249]]
ermapping=[['AE',0.4896],['AR',0.2857],['AT',0.3711],['AU',0.5308],['BE',0.4168],['BG',0.3877],['BO',0.3622],['BR',0.2583],['CA',0.4754],\
           ['CH',0.6038],['CL',0.1838],['CN',0.2006],['CO',0.0910],['CZ',0.3367],['DE',0.4149],['DK',0.1696],['DZ',0.4896],\
           ['EE',0.1478],['ES',0.2734],['FR',0.2851],['GB',0.3952],['GR',0.2675],['HK',0.3164],['HN',0.0816],['HR',0.2123],\
           ['HU',0.1632],['IE',0.3441],['IN',0.1700],['IQ',0.4897],['IR',0],['IT',0.2938],['JE',0.4898],['JP',0.5078],['KE',0.0268],\
           ['LU',0.2811],['MD',0.0714],['MT',0.1243],['MX',0.0723],['MY',1.0],['NG',0.1326],['NL',0.2905],['NZ',0.6173],['PH',0.2130],\
           ['PK',0.1197],['PL',0.2662],['PR',0.7959],['PT',0.1982],['RO',0.2419],['RS',0.1098],['RU',0.5191],['SG',0.5111],\
           ['SI',0.3052],['TN',0.1422],['TR',0.0821],['UA',0.0479],['US',0.7453],['VN',0.1367]]
rrmapping=[['0',0.62],['50',0.00],['100',1.00]]
clmapping=[['AE',0.6254],['AS',0.0915],['AT',0.4489],['AU',0.6778],['BE',0.5322],['BR',0.0951],['CA',0.6262],['CH',0.3916],['CL',0.2347],\
           ['CN',0.4408],['CO',0.1162],['CZ',0.3057],['DE',0.5052],['DK',0.3282],['DZ',0.6255],['EE',0.1887],['ES',0.3196],['FR',0.3906],\
           ['GB',0.5058],['GR',0.3128],['HN',0.1042],['HR',0.2711],['HU',0.2067],['IE',0.4393],['IL',0.7495],['IN',0.1601],['IQ',0.6256],\
           ['IR',0.00],['IT',0.2108],['JP',0.7174],['KE',0.0343],['LU',0.2602],['MD',0.0912],['MT',0.1587],['MX',0.1832],['MY',0.2345],\
           ['NG',0.1693],['NL',0.3318],['NZ',0.7882],['PK',0.0608],['PL',0.4044],['PT',0.2853],['RO',0.3648],['RU',1.00],['SG',0.5556],\
           ['SI',0.3897],['TR',0.1048],['UA',0.0612],['US',0.9139],['VN',0.01]]
csmapping=[['L',1.00],['M',0.91],['S',0.00]]



# Function to retrieve value based on label
def get_value_by_label(label, couples_list):
    for couple in couples_list:
        if couple[0] == label:
            return couple[1]
    return None  # Return None if label is not found

# example of input and mapping
xinput=[ymapping,'EN','PT','Data Engineer','MT','100','MT','S']
#print (xinput)
x_to_predict=[ymapping,get_value_by_label(xinput[1], exlemapping),get_value_by_label(xinput[2], emtymapping),get_value_by_label(xinput[3], jtmapping),\
              get_value_by_label(xinput[4], ermapping),get_value_by_label(xinput[5], rrmapping),get_value_by_label(xinput[6], clmapping),\
              get_value_by_label(xinput[7], csmapping)]

#apply scaler to the input data point
new_data_point = scaler.transform([x_to_predict])


# predict salary class for the xinput

predict_salary_class = clf.predict(new_data_point)

if predict_salary_class == 'L':
    print("\n The salary group prediction for the input data point is LOW, meaning that the salary is lesser than 74,015.6 USD")
elif predict_salary_class == 'M':
    print("\n The salary group prediction for the input data point is MEDIUM, meaning that the salary is greater than or equal to 74,015.6 USD and lesser than 128,875.0 USD")
else:
    print("\n The salary group prediction for the input data point is HIGH, meaning that the salary is greater than or equal to 128,875.0 USD")
#print("\n The salary group prediction for the input vector " + str(x_to_predict) + " is = " + str(predict_salary_class))

