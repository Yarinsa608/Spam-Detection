import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# --- Setup and directory creation (Safety Check) ---
output_dirs = ['./data', './visualizations', './models']
for dir_name in output_dirs:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
print("Project directories checked/created.")


#---------------- spam detection (text analysis)------------------------
print("\nStarting  data processing...")

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
extra_stopwords = {
    "would", "could", "should", "will", "get", "know", "use", "say", "see", "make",
    "go", "like", "also", "thank", "regards", "please", "dear", "let", "us", "ok", "me","you"
}
stop_words |= extra_stopwords

# ----------------------------
# Cleaning function
# ----------------------------
def clean_dataset_basic(text):
    '''
    Clean and normalise Email text  BY:
    1. Convert ot lowercase
    2.replace mail/urls with blank
    3.remove punctuation,digits and extra whitespace
    4. remove stopwords


'''

    text=text.lower()
    #Potenially we could extarct with urls and sender stuff
    #Easiest way would be to detect based on sender recieved and patterns but would eventually be invalid
    text=re.sub(r"\S+@\s+","EMAIL",text)
    text = re.sub(r"http\S+|www\S+|mailto:\S+", "", text)#URLS purger
    #Remove punctuation,digits and special chars
    #older
    # text = re.sub(r"[^a-z\s]", " ", text)      #Remove any punct SPECIAL CHARS OR NUMS
    text=re.sub(r"\s", " ",text).strip()#Remove extra spaces
    #split and remove Stop words
    words=text.split()
    words= [x for x in words if x not in stop_words and len(x)>1]
    #Rejoin 
    text=' '.join(words)
  
    return text.strip()



# ----------------------------
# Preprocess function
# ----------------------------
def preprocess_all():
    try:
        dataset_selection = int(input(
            "Please select a dataset:\n"
            " 1: Enron\n"
            " 2: Figshare\n"
            " 3: NaserPhishingDataset\n"
            " 4: Cyber Cop\n"
        ))
        match dataset_selection:
            case 1:
                data_folder = os.path.join('data', 'raw', 'enron_spam_data-master')
                dataset = 'enron_spam_data.csv'
            case 2:
                data_folder = os.path.join('data', 'raw', 'Seven_Phishing_Email_Datasets')
                dataset = 'Seven_Phishing_Email_Datasets.csv'
            case 3:
                data_folder = os.path.join('data', 'raw', 'Naser-Phishing-Dataset')
                dataset = 'Naser.csv'
            case 4:
                data_folder = os.path.join('data', 'raw')
                dataset = 'Phishing_Email.csv'
            case _:
                raise ValueError("Invalid dataset selection.")
    except FileNotFoundError:
        print("FATAL ERROR: dataset not found. Place it in the './data/raw/' folder and try again.")
        exit()

    # Determine files to process
    dataset_path = os.path.join(data_folder, dataset)
    if os.path.isfile(dataset_path):
        files = [dataset_path]
    elif os.path.isdir(data_folder):
        files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    else:
        raise ValueError(f"Path does not exist: {dataset_path}")

    # Create output folder
    preprocessed_folder = os.path.join('data', 'preprocessed')
    os.makedirs(preprocessed_folder, exist_ok=True)
    print(f"Processing {len(files)} file(s)...")

    for file_path in files:
        try:
            df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
            print(f"\nLoaded {file_path} with shape {df.shape}")
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue

        # Ensure label exists
        if 'label' not in df.columns:
            if 'Spam/Ham' in df.columns:
                df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})
            else:
                df['label'] = 0  # default if missing

        # Replace empty object columns with empty string
        for col in df.columns:
            if df[col].dtype == object:
                if df[col].isna().all() or df[col].str.strip().eq('').all():
                    df[col] = ""

        # Build unified text column
        if 'subject' in df.columns and 'body' in df.columns:
            df['email_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
        elif 'body' in df.columns:
            df['email_text'] = df['body'].fillna('')
        else:
            text_col = next((c for c in df.columns if df[c].dtype == object), None)
            df['email_text'] = df[text_col].fillna('') if text_col else ''

        # Clean text
        df['clean_email_text'] = df['email_text'].apply(clean_dataset_basic)

        # Keep final columns
        final_columns = ['clean_email_text', 'label']
 
        df_final = df[final_columns].rename(columns={'clean_email_text': 'text'})

        # Save CSV
        preprocessed_name = os.path.basename(file_path).replace('.csv', '_preprocessed.csv')
        output_path = os.path.join(preprocessed_folder, preprocessed_name)
        df_final.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved preprocessed file to: {output_path}")


        print(df_final.head(3))

        print("\nData pre-processing successfully completed.")
        
if __name__ == "__main__":
    try:
        preprocess_all()
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
