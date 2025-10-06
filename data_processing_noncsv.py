import os
import re
import csv
from bs4 import BeautifulSoup
from email import policy
from email.parser import BytesParser

# config
INPUT_DIRS = {
    "spam": "spam_emails",
    "ham": "ham_emails"
}
OUTPUT_CSV = "cleaned_spam_assassin_emails.csv"


def clean_text(text):
    text = str(text).lower()
    # Remove typical email headers/footers
    text = re.sub(r'-----original message-----|from:.*|sent:.*|to:.*|subject:.*', '', text, flags=re.MULTILINE)
    # Remove all non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_body_from_email(filepath):
    with open(filepath, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    subject = msg['subject'] if msg['subject'] else ""
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                body = part.get_content()
                break
            elif ctype == "text/html":
                html = part.get_content()
                body = BeautifulSoup(html, "html.parser").get_text()
                break
    else:
        ctype = msg.get_content_type()
        if ctype == "text/plain":
            body = msg.get_content()
        elif ctype == "text/html":
            html = msg.get_content()
            body = BeautifulSoup(html, "html.parser").get_text()

    # Combine subject and body before cleaning
    return clean_text(subject + " " + body)

def main():
    emails = []
    for label, folder in INPUT_DIRS.items():
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                try:
                    cleaned = extract_body_from_email(filepath)
                    # Convert label to 0/1
                    label_numeric = 1 if label == "spam" else 0
                    emails.append([cleaned, label_numeric])
                except Exception as e:
                    print(f"Failed to process {filepath}: {e}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(emails)

    print(f"Cleaned {len(emails)} emails saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()