# Import libraries
import os
import argparse
import pandas as pd
from azureml.core import Run
from sklearn.preprocessing import MinMaxScaler

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type=str, dest='raw_dataset_id',
                    help='raw dataset')
parser.add_argument('--prepped-data', type=str, dest='prepped_data',
                    default='prepped_data',
                    help='Folder for results')
args = parser.parse_args()
save_folder = args.prepped_data

# Get experiment run context
run = Run.get_context()

# Load data
print('Loading Data...')
heart_df = run.input_datasets['raw_data'].to_pandas_dataframe()

# Log raw dataset details
run.log('Raw rows:', heart_df.shape[0])
run.log('Raw columns:', heart_df.shape[1])

# Drop NAs
heart_df = heart_df.dropna()

# Modifying prediction label
heart_df.rename(columns={'output' : 'heart_attack'}, inplace=True)

# Change sex categorical feature to dummies
sex_type = pd.get_dummies(data=heart_df['sex'])
sex_type.columns = ['Male', 'Female']
chest_pain = pd.get_dummies(data=heart_df['cp'])
chest_pain.columns = ['Chest Pain 1', 'Chest Pain 2', 'Chest Pain 3',
                      'Chest Pain 4']
ex_angina = pd.get_dummies(data=heart_df['exng'])
ex_angina.columns = ['Exercise Angina: No', 'Exercise Angina: Yes']
slp_type = pd.get_dummies(data=heart_df['slp'])
slp_type.columns = ['Slope 1', 'Slope 2', 'Slope 3']
caa_type = pd.get_dummies(data=heart_df['caa'])
caa_type.columns = ['CAA 1', 'CAA 2', 'CAA 3', 'CAA 4', 'CAA 5']
thall_type = pd.get_dummies(data=heart_df['thall'])
thall_type.columns = ['Thall 1', 'Thall 2', 'Thall 3', 'Thall 4']


# Joining and removing modified columns
heart_df = pd.concat([sex_type, chest_pain, ex_angina, slp_type,
                      caa_type, thall_type, heart_df],
                     axis=1)
heart_df.drop(labels=['sex', 'cp', 'exng', 'slp', 'caa', 'thall'], 
              axis=1, inplace=True)

# Normalize numeric columns
scaler = MinMaxScaler()
norm_cols = ['age', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
                'oldpeak']
heart_df[norm_cols] = scaler.fit_transform(heart_df[norm_cols])

# Log prepped dataset details
run.log('Prepped rows:', heart_df.shape[0])
run.log('Prepped columns:', heart_df.shape[1])

# Save the prepped data
print('Saving Data...')
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, 'heartdata_norm.csv')
heart_df.to_csv(save_path, index=False, header=True)

# End the run
run.complete()
