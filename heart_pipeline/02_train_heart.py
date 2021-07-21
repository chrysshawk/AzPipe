
# Import libraries
from azureml.core import Run, Model
import argparse
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--training-data',
                    type=str,
                    dest='training_data',
                    help='training_data')
args = parser.parse_args()
training_data = args.training_data

# Get the experiment run context
run = Run.get_context()

# Load the data
print('Loading Data...')
file_path = os.path.join(training_data, 'heartdata_prepped.csv')
heart = pd.read_csv(file_path)

# Separate features from labels
X, y = heart.iloc[:,0:-1].values, heart.iloc[:,-1]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                    random_state=0)

# Train model
print('Training model...')
model = DecisionTreeClassifier().fit(X_train, y_train)

# Calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# Calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test, y_scores[:,1])
print('AUC:', auc)
run.log('AUC', np.float(auc))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
fig = plt.figure(figsize=(6,4))
plt.plot([0,1], [0,1], 'k--') # diagonal 50% line
plt.plot(fpr, tpr, 'r--')
plt.xlabel('FP rate')
plt.ylabel('TP rate')
plt.title('ROC Curve')
run.log_image(name='ROC', plot=fig)
plt.show()

# Save trained model 
print('Saving model...')
os.makedirs('./outputs', exist_ok=True)
model_file = os.path.join('./outputs', 'heart_model.pkl')
joblib.dump(value=model, filename=model_file)

# Register model
print('Registering model...')
Model.register(workspace = run.experiment.workspace,
               model_path = model_file,
               model_name = 'heart_model',
               tags = {'Context' : 'Pipeline',
                       'Purpose' : 'DP100'},
               properties = {'AUC' : np.float(auc),
                             'Accuracy' : np.float(acc)}
               
run.complete()
