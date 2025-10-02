import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#----------------Enron spam detection (text analysis)------------------------
#Load Enron data
df_enron = pd.read_csv('./data/enron_spam_data.csv')

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
#Load Phishing data (Assuming Result is the label)
df_phish = pd.read_csv('./data/dataset_full.csv')

# Assuming the label column is the last one
# Assuming the column is named 'class' or 'label' for now.
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

#Don't save the scaled data, but the scaler object will be saved later.