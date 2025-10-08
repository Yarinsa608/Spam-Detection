

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, adjusted_rand_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, adjusted_rand_score

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
    # Automatically guess the label column for classification
def prepare_label(df):
    possible_labels = [col for col in df.columns if col.lower() in ['label','spam','phish','spam/ham','ham']]
    if not possible_labels:
        raise ValueError("No valid categorical label found in dataset!")
    return possible_labels[0]
#-------------------------Model training and evaluation------------------
def vectorize_and_train(df, label_col, dataset_name):
    #---Enron classification ML type1 text--
    label_col = prepare_label(df)
    text_col = get_text_column(df)
     # Preprocessing: replace NaN in text with empty string
    # CRITICAL FIX: Replace NaN values in the text column with an empty string.
    # This prevents the "np.nan is an invalid document" ValueError in TfidfVectorizer.
    X = df[text_col].fillna('')
    y = df[label_col]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vector = TfidfVectorizer(max_features=5000)
    X_train_vec = vector.fit_transform(X_train)
    X_test_vec = vector.transform(X_test)
    print(f"Vectorization complete for {dataset_name}")

    # Dense conversion for scaling
    X_train_dense = X_train_vec.toarray()
    X_test_dense = X_test_vec.toarray()

    # FIX 2: Save the newly fitted scaler for future consistency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_test_scaled = scaler.transform(X_test_dense)
    print(f"Scaling complete for {dataset_name}.\n")
    #Train Model A (Baseline: Naive Bayes)
    model_nb = MultinomialNB()
    model_nb.fit(X_train_vec, y_train)
    preds_nb = model_nb.predict(X_test_vec)
    acc_nb = accuracy_score(y_test, preds_nb)
    print(f"{dataset_name} - Naive Bayes Accuracy: {acc_nb:.4f}")
    #Train Model B (Advanced: Logistic Regression)
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train_vec, y_train)
    preds_lr = model_lr.predict(X_test_vec)
    acc_lr = accuracy_score(y_test, preds_lr)
    print(f"{dataset_name} - Logistic Regression Accuracy: {acc_lr:.4f}")

    # Save models and vector
    os.makedirs('./models', exist_ok=True)
    joblib.dump(model_lr, f'./models/{dataset_name}_logreg_model.joblib')
    joblib.dump(vector, f'./models/{dataset_name}_vector.joblib')
    joblib.dump(scaler, f'./models/{dataset_name}_scaler.joblib')
    print(f"{dataset_name} models and vector saved.\n")
#---Phishing classification, clustering ML type2,3--
#Train Phishing Classifier (Model C: Gradient Boosting)
    model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    model_gb.fit(X_train_vec, y_train)
    preds_gb = model_gb.predict(X_test_vec)
    acc_gb = accuracy_score(y_test, preds_gb)
    print(f"{dataset_name} - Gradient Boosting Accuracy: {acc_gb:.4f}")
    joblib.dump(model_gb, f'./models/{dataset_name}_gradientBabe.joblib')
    #Clustering (ML Type 2: Unsupervised Learning for complexity)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_train_vec)
    ari_score = adjusted_rand_score(y_train, kmeans.labels_)
    print(f"{dataset_name} - KMeans ARI: {ari_score:.4f}\n")
    # Return vectors and labels for further use (e.g., testing)
    return X_train_vec, X_test_vec, y_train, y_test


    #------------------Visualization-------------------------------
def generate_visualizations(df, label_col, dataset_name):
    print("\nGenerating visualizations...")
    # Save folder two levels up
    output_folder = os.path.join( 'visualizations')
    os.makedirs(output_folder, exist_ok=True)
    print(output_folder)
    
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
    plt.savefig(os.path.join(output_folder, f'{dataset_name}_class_dist.png'))
    plt.close()
# 2. Text Length Distribution (Box Plot - Cleaner than Histogram)
    if 'text' in df.columns:
        df['text_length'] = df['text'].astype(str).apply(len)
        plt.figure(figsize=(8,6))# The UserWarning about palette is due to an updated seaborn version; this is minor and harmless
        sns.boxplot(x=label_col, y='text_length', hue=label_col, data=df, palette=['green', 'red'], legend=False)
        plt.title(f'{dataset_name} – Text Length')
        plt.xlabel('Email Class (0=Ham, 1=Spam)', fontsize=12)#I do not like this but it works so do not under any cirm touch it 
        plt.ylabel('Email Character Length', fontsize=12)# Zoom into the 95th percentile for better visualization
        plt.ylim(0, df['text_length'].quantile(0.95) * 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{dataset_name}_text_length.png'))
        
        plt.close()

# 3. Word Cloud (Spam)
    if 'text' in df.columns:
        spam_text = " ".join(df[df[label_col] == 1]['text'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{dataset_name} – Spam Word Cloud')
        plt.savefig(os.path.join(output_folder, f'{dataset_name}_spam_wordcloud.png'))
        plt.close()

 
#no longer valid due to constraints
# 4. Phishing Feature Box Plot (Number of dots in URL)
    if 'qty_dot_url' in df.columns:
        plt.figure(figsize=(12,6))
        sns.boxplot(x=label_col, y='qty_dot_url', data=df, palette=['green', 'red'])
        plt.title('Number of Dots in URL: Safe vs. Phishing', fontsize=14)
        plt.xlabel('Class (0=Safe, 1=Phishing/Malicious)', fontsize=12)
        plt.ylabel('Quantity of Dots in URL', fontsize=12)
        plt.ylim(0, df['qty_dot_url'].quantile(0.95) * 1.2)  # zoom to 95th percentile
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{dataset_name}_phish_dot_url.png'))
        plt.close()
    else:
        print("Skipping 'qty_dot_url' plot- as column not found.")

# 5. Correlation Heatmap (Top features only)
    numerical_df = df.select_dtypes(include=['number'])
    if label_col in numerical_df.columns:
        # Calculate correlation with the target variable, sort, and take the top features
        corr_to_target = numerical_df.corr()[label_col].abs().sort_values(ascending=False).head(12).index
        corr_matrix_subset = numerical_df[corr_to_target].corr()
        plt.figure(figsize=(14,12))
        sns.heatmap(corr_matrix_subset, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'shrink':0.8}, linewidths=0.5)
        plt.title(f'{dataset_name} – Correlation Heatmap (Top Features)', fontsize=18)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{dataset_name}_corr_heatmap.png'))
        plt.close()
    else:
        print(f"Warning Label column '{label_col}' not numeric or missing.")

    print(f"Visualizations saved for {dataset_name}")

def test_email(df, text_col, label_col=None, dataset_name=None):
        # Ask user to select an email by row or input custom text
    while True:
        choice = input("Enter 'row' to select a row number or 'text' to input custom text: ").strip().lower()
        if choice in ['row', 'text']:
            break
        else:
            print("Invalid choice. Please type 'row' or 'text'.")

    if choice == 'row':
        row_number = int(input(f"Enter row number (0-{len(df)-1}): "))
        email_text = df.loc[row_number, text_col]
        true_label = df[label_col].iloc[row_number] if label_col and label_col in df.columns else None
        print(f"\nSelected Email from row {row_number}:\n{email_text}")
    else:
        email_text = input("Enter your custom email text: ") # honestly did not know what to put this is more or less temp for final
        true_label = None

    # Ensure models/vector exist; if not, train
    vector_path = f'./models/{dataset_name}_vector.joblib'
    if not os.path.exists(vector_path):
        print("Models not found. Training models now...")
        vectorize_and_train(df, label_col, dataset_name)

    # Load vector
    vector = joblib.load(vector_path)
    email_vec = vector.transform([email_text])

    # Load models
    predictions = {}
    model_paths = {
        'Logistic Regression': f'./models/{dataset_name}_logreg_model.joblib',
        'Naive Bayes': f'./models/{dataset_name}_nb_model.joblib',
        'Gradient Boosting': f'./models/{dataset_name}_gradientBabe.joblib'
    }

    for name, path in model_paths.items():
        if os.path.exists(path):
            model = joblib.load(path)
            predictions[name] = model.predict(email_vec)[0]

    # Majority vote
    counts = {0: 0, 1: 0}
    for val in predictions.values():
        counts[val] += 1
    final_label = "Not Spam (Ham)" if counts[0] > counts[1] else "Spam"

    # Output
    print("\nPredictions per model:")
    for model_name, label in predictions.items():
        print(f"{model_name}: {label} ({'Spam' if label == 1 else 'Ham'})")
    print(f"\nFinal Email Prediction: {final_label}")
    if true_label is not None:
        print(f"True label: {true_label}")

    return email_text, predictions, final_label
def choose_csv_and_run(folder_path='./data/preprocessed/'):
    # List CSV files
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found.")
        return None, None, None, None, None, None, None, None

    print("\nAvailable CSV files:")
    for i, f in enumerate(csv_files):
        print(f"{i}: {f}")

    # Ask user which CSV to process
    while True:
        try:
            choice = int(input(f"Enter CSV number (0-{len(csv_files)-1}): "))
            if 0 <= choice < len(csv_files):
                break
            else:
                print("Invalid number, try again.")
        except ValueError:
            print("Please enter a valid integer.")

    selected_file = csv_files[choice]
    dataset_name = os.path.splitext(selected_file)[0]  # Important!
    file_path = os.path.join(folder_path, selected_file)
    print(f"\nProcessing dataset '{dataset_name}'...\n")

    # Load CSV and prepare columns
    df = load_csv(file_path)
    text_col = get_text_column(df)
    label_col = prepare_label(df)

    # Ask action
    action = int(input("Test or Train Model (1=Test, 2=Train): "))
    match action:
        case 1:
            print("Testing\n")
            # Pass dataset_name here
            email_text, predictions, final_label = test_email(df, text_col, label_col, dataset_name)
            return df, dataset_name, None, None, None, None, email_text, predictions
        case 2:
            print("Training\n")
            generate_visualizations(df, label_col, dataset_name)
            X_train_vec, X_test_vec, y_train, y_test = vectorize_and_train(df, label_col, dataset_name)
            print(f"\nProcessing complete for dataset: {dataset_name}\n")
            return df, dataset_name, X_train_vec, X_test_vec, y_train, y_test, None, None
        case _:
            print("Invalid choice.")
            return None, None, None, None, None, None, None, None

# ------------------ Main Run -----------------
if __name__ == "__main__":
    df, dataset_name, X_train_vec, X_test_vec, y_train, y_test, email_text, predictions = choose_csv_and_run()

    if predictions is not None:
        print(f"\nInput email:\n{email_text}\n")
        spam_count = 0
        ham_count = 0

        for model_name, label in predictions.items():
            print(f"{model_name}: {label} ({'Spam' if label==1 else 'Ham'})")
            if label == 1:
                spam_count += 1
            else:
                ham_count += 1

        final_label = "Spam" if spam_count > ham_count else "Ham"
        print(f"\nFinal Prediction based on majority vote:")
        print(f"SPAM votes: {spam_count}, HAM votes: {ham_count}")
        print(f"Overall Predicted: {final_label}")

    elif X_train_vec is not None:
        print("Training completed. Models and vector are saved.")
    else:
        print("No action performed.")




