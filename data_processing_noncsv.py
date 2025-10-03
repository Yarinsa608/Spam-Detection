import os
import re
import pandas as pd
import argparse
from email import policy
from email.parser import BytesParser

#Command-line arguments
parser = argparse.ArgumentParser(description="Clean raw email files for spam/ham detection")
parser.add_argument("spam_folder", type=str, help="Path to the folder containing spam emails")
parser.add_argument("ham_folder", type=str, help="Path to the folder containing ham emails")
parser.add_argument("--output", type=str, default="./processed_emails.csv", help="Output CSV file path")
args = parser.parse_args()

#Load Raw Email Files
def load_emails_from_folder(folder_path, label):
    emails = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not filename.endswith(".txt") and not filename.endswith(".eml"):
            continue
        try:
            with open(file_path, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)
                subject = msg['subject'] if msg['subject'] else ""
                body = msg.get_body(preferencelist=('plain'))
                body_text = body.get_content() if body else ""
                text = subject + " " + body_text
                emails.append({"text": text, "label": label})
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return emails

# Load emails using folder paths from command line
spam_emails = load_emails_from_folder(args.spam_folder, 1)
ham_emails  = load_emails_from_folder(args.ham_folder, 0)

# Combine into one DataFrame
df_raw = pd.DataFrame(spam_emails + ham_emails)
print("Loaded emails:", df_raw.shape)

#Clean Email Text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'from:.*|to:.*|subject:.*|return-path:.*|received:.*|date:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'-----original message-----', '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_raw['cleaned_text'] = df_raw['text'].apply(clean_text)

# Feature engineering
df_raw['text_length'] = df_raw['cleaned_text'].apply(len)
df_raw['punc_count'] = df_raw['text'].apply(lambda x: len([c for c in str(x) if c in '!?$']))

# Save to CSV
df_raw.to_csv(args.output, index=False)
print(f"dataset saved to {args.output}")