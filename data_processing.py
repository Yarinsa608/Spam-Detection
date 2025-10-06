import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib # Needed to save the scaler object

# --- Setup and directory creation (Safety Check) ---
output_dirs = ['./data', './visualizations', './models']
for dir_name in output_dirs:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
print("Project directories checked/created.")


#----------------Enron spam detection (text analysis)------------------------
print("\nStarting Enron data processing...")

#Load Enron data
try:
    df_enron = pd.read_csv('./data/enron_spam_data.csv')
except FileNotFoundError:
    print("FATAL ERROR: enron_spam_data.csv not found. Place it in the './data/' folder and try again.")
    exit()

#Drop the 'Date' column
df_enron = df_enron.drop(columns=['Date'])

#Combine Subject and Message into one text column 
#Fill any missing messages/subjects with an empty string first
df_enron['text'] = df_enron['Subject'].fillna('') + ' ' + df_enron['Message'].fillna('')

#Rename the label column and convert 'spam' to 1, 'ham' to 0
df_enron.rename(columns={'Spam/Ham': 'label'}, inplace=True)
df_enron['label'] = df_enron['label'].map({'ham': 0, 'spam': 1})

#Keep only the final columns need
df_enron = df_enron[['text', 'label']]

print("Enron Data Shape:", df_enron.shape)
print("Enron Label Distribution:\n", df_enron['label'].value_counts())

#Function to clean text
def clean_text(text):
    text = str(text).lower()
    # 1. Remove email headers/footers often seen in forwarded emails
    text = re.sub(r'-----original message-----|from:.*|sent:.*|to:.*|subject:.*', '', text, flags=re.MULTILINE)
    # 2. Remove all non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    # 3. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#Apply cleaning
df_enron['cleaned_text'] = df_enron['text'].apply(clean_text)

#Feature engineering for visualization (Count features)
df_enron['text_length'] = df_enron['cleaned_text'].apply(len)
df_enron['punc_count'] = df_enron['text'].apply(lambda x: len([c for c in str(x) if c in '!?$']))

print("Enron data processing complete.")
# Save this intermediate file to the data folder
df_enron.to_csv('./data/processed_enron.csv', index=False)


#----------------Phishing detection (numerical analysis)------------------------
print("\nStarting Phishing data processing...")

#Load Phishing data (Assuming Result is the label)
try:
    df_phish = pd.read_csv('./data/dataset_full.csv')
except FileNotFoundError:
    print("FATAL ERROR: dataset_full.csv not found. Place it in the './data/' folder and try again.")
    exit()

# Assuming the label column is the last one
label_col = df_phish.columns[-1]

X_phish = df_phish.drop(columns=[label_col]) # Features are all the qty_* columns
y_phish = df_phish[label_col]                # The label (target)

#Split the data first
X_train_ph, X_test_ph, y_train_ph, y_test_ph = train_test_split(X_phish, y_phish, test_size=0.2, random_state=42)

#Instantiate the Scaler (need for numerical ML models)
scaler = StandardScaler()

#Fit the scaler only on the training data and transform both sets
X_train_scaled = scaler.fit_transform(X_train_ph)
X_test_scaled = scaler.transform(X_test_ph)

print("Phishing data scaling complete.")

#Save the scaler object (required for new predictions in the future)
joblib.dump(scaler, './models/phish_scaler.joblib')

print("\nData processing successfully completed.")