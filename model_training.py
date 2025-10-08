import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# --- Ensure output directories exist ---
os.makedirs('./models', exist_ok=True)
os.makedirs('./visualizations', exist_ok=True)
print("Output directories checked/created.")

# --- Initial setup ---
try:
    # Load processed data
    df_enron = pd.read_csv('./data/preprocessed/processed_enron.csv')
    df_phish = pd.read_csv('./data/dataset_full.csv')
    
    # Use a robust way to determine the phishing label column
    label_col_guess = df_phish.columns[-1]
    if 'label' in label_col_guess.lower() or 'phish' in label_col_guess.lower():
        label_col = label_col_guess
    else:
        label_col = label_col_guess 
        print(f"Warning: Phishing label column assumed to be '{label_col}'. Verify this is correct.")
        
except FileNotFoundError:
    print("FATAL ERROR: Data files not found. Run data_processing.py first.")
    exit()

# Load scaled data/scaler to avoid re-splitting/re-scaling if possible
try:
    scaler = joblib.load('./models/phish_scaler.joblib')
    
    # Recreate split and scale here for consistent logic, using the loaded scaler
    X_phish = df_phish.drop(columns=[label_col])
    y_phish = df_phish[label_col]
    X_train_ph, X_test_ph, y_train_ph, y_test_ph = train_test_split(X_phish, y_phish, test_size=0.2, random_state=42)
    X_train_scaled = scaler.transform(X_train_ph)
    X_test_scaled = scaler.transform(X_test_ph)

except FileNotFoundError:
    # Fallback if scaler wasn't saved (will rerun the scaling logic)
    print("Warning: Phishing scaler not found. Re-splitting, re-scaling, and saving new scaler...")
    X_phish = df_phish.drop(columns=[label_col])
    y_phish = df_phish[label_col]
    X_train_ph, X_test_ph, y_train_ph, y_test_ph = train_test_split(X_phish, y_phish, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_ph)
    X_test_scaled = scaler.transform(X_test_ph)
    
    # FIX 2: Save the newly fitted scaler for future consistency
    joblib.dump(scaler, './models/phish_scaler.joblib')
    print("New phishing scaler saved to ./models/phish_scaler.joblib.")


#------------------Visualization-------------------------------
print("\nGenerating visualizations...")

# 1. Class Imbalance Plot (Bar Chart with annotations)
plt.figure(figsize=(7, 5))
ax = sns.countplot(x='label', data=df_enron)
ax.set_xticklabels(['Ham (0)', 'Spam (1)'])
plt.title('Distribution of Email Classes in Enron Dataset', fontsize=14)
plt.xlabel('Email Class', fontsize=12)
plt.ylabel('Number of Emails', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.savefig('./visualizations/enron_class_distribution.png')
plt.close() 

# 2. Text Length Distribution (Box Plot - Cleaner than Histogram)
plt.figure(figsize=(9, 6))
# The UserWarning about palette is due to an updated seaborn version; this is minor and harmless
sns.boxplot(x='label', y='text_length', data=df_enron, palette=['green', 'red'])
plt.title('Email Length Distribution: Ham vs. Spam (Box Plot)', fontsize=14)
plt.xlabel('Email Class (0=Ham, 1=Spam)', fontsize=12)
plt.ylabel('Email Character Length', fontsize=12)
# Zoom into the 95th percentile for better visualization
plt.ylim(0, df_enron['text_length'].quantile(0.95) * 1.1) 
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('./visualizations/enron_text_length.png')
plt.close()

# 3. Word Cloud (Spam)
spam_text = " ".join(df_enron[df_enron['label'] == 1]['cleaned_text'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Words in Spam Emails', fontsize=14)
plt.savefig('./visualizations/enron_spam_wordcloud.png')
plt.close()

# 4. Phishing Feature Box Plot (Number of dots in URL)
plt.figure(figsize=(8, 6))
sns.boxplot(x=label_col, y='qty_dot_url', data=df_phish) 
plt.title('Number of Dots in URL: Safe vs. Phishing', fontsize=14)
plt.xlabel('Class (0=Safe, 1=Phishing/Malicious)', fontsize=12)
plt.ylabel('Quantity of Dots in URL', fontsize=12)
plt.tight_layout()
plt.savefig('./visualizations/phish_dot_url_boxplot.png')
plt.close()

# 5. Correlation Heatmap (Top features only)
plt.figure(figsize=(10, 8))
# Calculate correlation with the target variable, sort, and take the top features
corr_to_target = df_phish.corr()[label_col].sort_values(ascending=False).head(10).index
corr_matrix_subset = df_phish[corr_to_target].corr()

sns.heatmap(corr_matrix_subset, 
            annot=True, 
            cmap='viridis', 
            fmt=".2f",
            linewidths=.5)

plt.title('Correlation Heatmap of Top 10 Features to Phishing Status', fontsize=14)
plt.tight_layout()
plt.savefig('./visualizations/phish_correlation_heatmap.png')
plt.close()

print("Visualizations saved.")

#-------------------------Model training and evaluation------------------
print("\n--- Starting Model Training and Evaluation ---")

#---Enron classification ML type1 text--

#Prepare Enron data for training
X_enron = df_enron['cleaned_text']
y_enron = df_enron['label']

#Replace NaN values in the text column with an empty string.
X_enron = X_enron.fillna('') 

X_train_en, X_test_en, y_train_en, y_test_en = train_test_split(X_enron, y_enron, test_size=0.2, random_state=42)

#Vectorization (Feature Extraction)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train_en)
X_test_vec = vectorizer.transform(X_test_en)

#Train Model A (Baseline: Naive Bayes)
model_nb = MultinomialNB()
model_nb.fit(X_train_vec, y_train_en)
preds_nb = model_nb.predict(X_test_vec)
print("\n--- Enron: Naive Bayes Report ---")
print(classification_report(y_test_en, preds_nb))

#Train Model B (Advanced: Logistic Regression)
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_vec, y_train_en)
preds_lr = model_lr.predict(X_test_vec)
print("\n--- Enron: Logistic Regression Report ---")
print(classification_report(y_test_en, preds_lr))

#Save the best model and vectorizer
best_enron_model = model_lr #Assuming LR is slightly better
joblib.dump(best_enron_model, './models/final_enron_model.joblib')
joblib.dump(vectorizer, './models/enron_vectorizer.joblib')
print("Enron models and vectorizer saved to ./models/")


#---Phishing classification, clustering ML type2,3--

#Train Phishing Classifier (Model C: Gradient Boosting)
model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
model_gb.fit(X_train_scaled, y_train_ph)
preds_gb = model_gb.predict(X_test_scaled)
print("\n--- Phishing: Gradient Boosting Report ---")
print(classification_report(y_test_ph, preds_gb))

#Save the best model
joblib.dump(model_gb, './models/final_phish_model.joblib')
print("Phishing model saved to ./models/")


#Clustering (ML Type 2: Unsupervised Learning for complexity)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)
cluster_labels = kmeans.labels_

#Evaluate clustering alignment (measures how well clusters match true labels)
ari_score = adjusted_rand_score(y_train_ph, cluster_labels)
print(f"\n--- Phishing: K-Means Clustering Score (Adjusted Rand Index) ---")
print(f"Alignment Score: {ari_score:.4f} (Demonstrates Unsupervised Learning)")

print("\nModel training and evaluation successfully completed.")