import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
import joblib
import os

#------------------Helper Functions-------------------

# Load CSV file into a DataFrame
def load_csv(filepath):
    df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
    print(f"\nLoaded {len(df)} rows from {filepath}")
    return df

# Determine which column contains the main email text
def get_text_column(df):
    if 'cleaned_text' in df.columns:
        return 'cleaned_text'
    elif 'text' in df.columns:
        return 'text'
    elif 'body' in df.columns and 'subject' in df.columns:
        df['coconut'] = df['subject'].astype(str) + " " + df['body'].astype(str)
        return 'coconut'
    else:
        raise ValueError("No text related columns found: 'cleaned_text', 'text', 'subject', or 'body'")

# Allow user to select a row to preview/test
def select_row_text(df, text_col): #Allow user to select a row to preview/test
    row_number = int(input(f"Please enter a row number from (0-{len(df)-1}): "))
    sampled = df.loc[row_number, text_col]
    print(f"\nSelected Email Test from row {row_number}:\n{sampled}")
    return sampled

# Automatically guess the label column for classification
def prepare_label(df):
    label_col_guess = df.columns[-1]
    if 'label' in label_col_guess.lower() or 'phish' in label_col_guess.lower() or 'spam/ham' in label_col_guess.lower():
        label_col = label_col_guess
    else:
        label_col = label_col_guess
        print(f"Warning: Label column assumed to be '{label_col}'")
    return label_col

# Generate visualizations for a dataset
def generate_visualizations(df, label_col, dataset_name):
    print("\nGenerating visualizations...")

    # 1. Class Imbalance Plot (Bar Chart with annotations)
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x=label_col, data=df)
    ax.set_xticklabels(['Ham (0)', 'Spam (1)'])
    plt.title(f'{dataset_name} – Class Distribution', fontsize=14)
    plt.xlabel('Email Class', fontsize=12)
    plt.ylabel('Number of Emails', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    os.makedirs('./visualizations', exist_ok=True)
    plt.savefig(f'./visualizations/{dataset_name}_class_dist.png')
    plt.close()

    # 2. Text Length Distribution (Box Plot)
    if 'text' in df.columns:
        df['text_length'] = df['text'].astype(str).apply(len)
        plt.figure(figsize=(8,6))
        sns.boxplot(x=label_col, y='text_length', hue=label_col, data=df, palette=['green', 'red'], legend=False)
        plt.title(f'{dataset_name} – Text Length')
        plt.xlabel('Email Class (0=Ham, 1=Spam)', fontsize=12)
        plt.ylabel('Email Character Length', fontsize=12)
        plt.ylim(0, df['text_length'].quantile(0.95) * 1.1)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset_name}_text_length.png')
        plt.close()

    # 3. Word Cloud (Spam)
    if 'text' in df.columns:
        spam_text = " ".join(df[df[label_col] == 1]['text'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{dataset_name} – Spam Word Cloud')
        plt.savefig(f'./visualizations/{dataset_name}_spam_wordcloud.png')
        plt.close()

    # 4. Phishing Feature Box Plot (Number of dots in URL)
    if 'qty_dot_url' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=label_col, y='qty_dot_url', data=df)
        plt.title('Number of Dots in URL: Safe vs. Phishing', fontsize=14)
        plt.xlabel('Class (0=Safe, 1=Phishing/Malicious)', fontsize=12)
        plt.ylabel('Quantity of Dots in URL', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset_name}_phish_dot_url.png')
        plt.close()
    else:
        print("Skipping 'qty_dot_url' plot- as column not found.")

    # 5. Correlation Heatmap (Top numerical features)
    numerical_df = df.select_dtypes(include=['number'])
    if label_col in numerical_df.columns:
        corr_to_target = numerical_df.corr()[label_col].sort_values(ascending=False).head(10).index
        corr_matrix_subset = numerical_df[corr_to_target].corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr_matrix_subset, annot=True, cmap='viridis', fmt=".2f")
        plt.title(f'{dataset_name} – Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset_name}_corr_heatmap.png')
        plt.close()
    else:
        print(f"Warning Label column '{label_col}' not numeric or missing.")

    print(f"Visualizations saved for {dataset_name}")

#------------------Main Training and Processing Function-------------------
def vectorize_and_train(df, label_col, dataset_name):
    label_col = prepare_label(df)
    text_col = get_text_column(df)
    
    # Preprocessing: replace NaN in text with empty string
    X = df[text_col].fillna('')
    y = df[label_col]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vectorization complete for {dataset_name}")
# MODEL A: Naive Bayes
    model_nb = MultinomialNB()
    model_nb.fit(X_train_vec, y_train)
    preds_nb = model_nb.predict(X_test_vec)
    acc_nb = accuracy_score(y_test, preds_nb)
    print(f"{dataset_name} - Naive Bayes Accuracy: {acc_nb:.4f}")

    # MODEL B: Logistic Regression
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train_vec, y_train)
    preds_lr = model_lr.predict(X_test_vec)
    acc_lr = accuracy_score(y_test, preds_lr)
    print(f"{dataset_name} - Logistic Regression Accuracy: {acc_lr:.4f}")

    # Save best model and vectorizer
    best_model = model_lr
    os.makedirs('./models', exist_ok=True)
    joblib.dump(best_model, f'./models/{dataset_name}_logreg_model.joblib')
    joblib.dump(vectorizer, f'./models/{dataset_name}_vectorizer.joblib')
    print(f"{dataset_name} models and vectorizer saved.")

    # Optional: Gradient Boosting + KMeans clustering
    model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    model_gb.fit(X_train_vec, y_train)
    preds_gb = model_gb.predict(X_test_vec)
    acc_gb = accuracy_score(y_test, preds_gb)
    print(f"{dataset_name} - Gradient Boosting Accuracy: {acc_gb:.4f}")

    # KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_train_vec)
    ari_score = adjusted_rand_score(y_train, kmeans.labels_)
    print(f"{dataset_name} - KMeans ARI: {ari_score:.4f}")

    return X_train_vec, X_test_vec, y_train, y_test



def choose_csv_and_run(folder_path='./preprocessed/'):
    """
    List all CSV files in the preprocessed folder,
    ask user to choose one, then run full processing, modeling, and visualization pipeline.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in folder:", folder_path)
        return None

    print("\nAvailable CSV files:")
    for i, file in enumerate(csv_files):
        print(f"{i}: {file}")

    while True:
        try:
            choice = int(input(f"\nEnter the number of the CSV to process (0-{len(csv_files)-1}): "))
            if 0 <= choice < len(csv_files):
                break
            else:
                print("Invalid number, try again.")
        except ValueError:
            print("Please enter a valid integer.")

    selected_file = csv_files[choice]
    dataset_name = os.path.splitext(selected_file)[0]
    file_path = os.path.join(folder_path, selected_file)
    print(f"\nYou selected: {selected_file}\nProcessing dataset '{dataset_name}'...\n")

    # Load CSV
    df = load_csv(file_path)

    # Get text column
    text_col = get_text_column(df)

    # Prepare label column
    label_col = prepare_label(df)

    # Generate visualizations
    generate_visualizations(df, label_col, dataset_name)

    # Vectorize and train models
    X_train_vec, X_test_vec, y_train, y_test = vectorize_and_train(df, label_col, dataset_name)

    print(f"\nProcessing complete for dataset: {dataset_name}\n")
    return df, dataset_name, X_train_vec, X_test_vec, y_train, y_test

#------------------Run the pipeline-----------------------------
if __name__ == "__main__":
    df, dataset_name, X_train_vec, X_test_vec, y_train, y_test = choose_csv_and_run()