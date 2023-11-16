import pandas as pd

# Load the dataset
file_path = 'train.csv'
data = pd.read_csv(file_path)

# Separate the last 20 rows
unseen_data = data.tail(20)

# Optionally, you can drop the label column if you want to mimic an actual unseen dataset scenario
#unseen_data.drop('label', axis=1, inplace=True)

# Save the new file
unseen_data.to_csv('unseendata.csv', index=False)
