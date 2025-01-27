# install libs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
# SVM scaling is much more sensible than other methods
# therefore the StandardScalar is used to scale the features.
# Datenskalierung: SVMs sind empfindlich gegenüber der Skalierung der Daten.
# Daher nutze ich einen StandardScaler, um die Features zu skalieren.
from sklearn.preprocessing import StandardScaler
#Feature selection: sometimes some irrelevant or weak features can alter
# the model creation in a negative way, so we try to use as a feature selection 
# method something like SelectKBest
# Feature Selection: Manchmal können irrelevante oder schwach korrelierte Features
# die Modellleistung beeinträchtigen. Versuche, Feature-Selection-Methoden wie
# SelectKBest zu verwenden.
from sklearn.feature_selection import SelectKBest, f_classif
# Use cross validation to proof stabilitz and robustness of the modell
# Ich nutze Cross-Validation, um die Stabilität und Robustheit des Modells zu prüfen.
from sklearn.model_selection import cross_val_score

# Preprocess strings to numbers
def preprocess_strings(df):
    # Init the Label encoders
    # Initialisieren des LabelEncoders
    le = LabelEncoder()

    # iterate through every colomn of the DataFrame
    # Iteriere durch jede Spalte im DataFrame
    for column in df.columns:
        if df[column].dtype == 'object':
            # change stinges inte numerical values for better training
            # Strings in der Spalte in numerische Werte umwandeln
            df[column] = le.fit_transform(df[column])

    return df


# Read the CSV-Data
data = pd.read_csv('/content/drive/MyDrive/Coding/Colab Notebooks/data/ds_salaries.csv')
data.drop(data[['id','salary','salary_currency']],axis=1, inplace=True)

# Lets have a look at the data
### data.head()

# Preparation of the strings
data = preprocess_strings(data)

# Show the prepared data frames
### data.head()

# Define the Goal variable (salary_in_usd) and the features
# Definieren der Zielvariable (salary_in_usd) und der Features
X = data.drop(columns=['salary_in_usd'])
y = data['salary_in_usd']

# Split the dataset into training/validation and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the dataset into training and validation subsets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# copied from Paolo
print("Total number of datapoints: " + str(X.shape[0]))
print("Number of datapoints in the training set: " + str(X_train.shape[0]))
print("Number of datapoints in the validation set: " + str(X_val.shape[0]))
print("Number of datapoints in the test set: " + str(X_test.shape[0]))

# Remove non correlation features
# Entfernung von nicht korrilierenden Features
selector = SelectKBest(score_func=f_classif, k=8)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# For scaling i use the StandardScalar to exclude features.
# Zur Datenskalierung nutze ich einen StandardScaler, um die Features zu skalieren.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# create a SVM model
# SVM Modell erstellen
# Hyperparameter-Tuning: Standardparameter des SVM-Modells sind möglicherweise nicht ideal für deine Daten.
# Versuche, den C-Parameter (Regularisierung), den kernel-Parameter (linear, poly, rbf, etc.) oder andere Hyperparameter zu optimieren.
model = SVC(C=1.0, kernel='rbf', gamma='scale')

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_text = model.predict(X_test)

# Make predictions on the validation data
y_pred_val = model.predict(X_test)

# calculate the accuracy
# Genauigkeit des Modells berechnen
accuracy = accuracy_score(y_test, y_pred)

print(f'The accuracy of the modell is: {accuracy:.2f}')
scores = cross_val_score(model, X_train, y_train, cv=3)
print(f'Cross-Validation Scores: {scores}')
print(f'Mean Cross-Validation Score: {scores.mean()}')

# >> Total number of datapoints: 607
# >> Number of datapoints in the training set: 388
# >> Number of datapoints in the validation set: 97
# >> Number of datapoints in the test set: 122
# >> The accuracy of the modell is: 0.02
# >> Cross-Validation Scores: [0.02307692 0.01550388 0.02325581]
# >> Mean Cross-Validation Score: 0.020612204333134567


# exemple for new data to predict ( here we have to feed new data, but i use a excel ist to upload everything)
# Beispiel für neue Daten zur Vorhersage (hier musst du deine neuen Daten eingeben)
prediction_data = pd.read_csv('/content/drive/MyDrive/Coding/Colab Notebooks/data/ds_salary_predictions.csv')
prediction_data.drop(prediction_data[['id','salary','salary_currency']],axis=1, inplace=True)

# Of course we have to drop the variable that we are looking fore
# Natürlich müssen wir noch die Variable droppen, die wir suchen
prediction_data.drop(prediction_data[['salary_in_usd']],axis=1, inplace=True)
# Save this file for later export
prediction_data_with_strings = prediction_data.copy()

# preparation of the strings again, if nessessary
# Vorverarbeiten der neuen Daten (falls nötig)
prediction_data = preprocess_strings(prediction_data)

# Show the preprocessed dataframes
# Anzeigen des vorverarbeiteten DataFrames
### prediction_data.head()


# Predictions for new data
# Vorhersagen für die neuen Daten
new_pred = model.predict(prediction_data)

print("Vorhersagen für neue Daten:")
for i, pred in enumerate(new_pred):
    print(f'Vorhersage {i+1}: {pred}')

# test the lengths
# Prüfen der Längen
print(f'Length of new_pred: {len(new_pred)}')
print(f'Length of prediction_data: {len(prediction_data)}')

# adding the predictions to the original data CSV file
# Hinzufügen der Vorhersagen zur Original-CSV-Datei
data_with_predictions = prediction_data_with_strings.copy()
data_with_predictions['salary_in_usd'] = new_pred

# save the new updated file to this location
# Speichern der aktualisierten Daten in einer neuen CSV-Datei
data_with_predictions.to_csv('/content/drive/MyDrive/Coding/Colab Notebooks/data/salary_vorhersagen.csv', index=False)

print("Predictions were added to the CSV file and saved.")

### data_with_predictions.head()



