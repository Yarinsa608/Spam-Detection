import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

#------------------Visualization-------------------------------
df_enron = pd.read_csv('./data/processed_enron.csv')
# Class imbalance plot (Bar Chart)
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df_enron)
plt.title('Enron Spam vs. Ham Distribution (0=Ham, 1=Spam)')
plt.savefig('./visualizations/enron_class_distribution.png')
plt.show()

#Text length distribution (Histogram)
plt.figure(figsize=(10, 5))
sns.histplot(df_enron[df_enron['label'] == 0]['text_length'], color='blue', label='Ham', kde=True, bins=50)
sns.histplot(df_enron[df_enron['label'] == 1]['text_length'], color='red', label='Spam', kde=True, bins=50)
plt.title('Text Length Distribution by Class')
plt.legend()
plt.savefig('./visualizations/enron_text_length.png')
plt.show()

#Word Cloud (Spam)
spam_text = " ".join(df_enron[df_enron['label'] == 1]['cleaned_text'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Words in Spam Emails')
plt.savefig('./visualizations/enron_spam_wordcloud.png')
plt.show()
#-----------------------------------------------------
#Phishing Feature Box Plot (Number of dots in URL)
plt.figure(figsize=(8, 6))
sns.boxplot(x=label_col, y='qty_dot_url', data=df_phish) 
plt.title('Number of Dots in URL: Safe vs. Phishing')
plt.xlabel('Class (0=Safe, 1=Phishing)')
plt.ylabel('qty_dot_url')
plt.savefig('./visualizations/phish_dot_url_boxplot.png')
plt.show()

#Correlation Heatmap (Top features only)
plt.figure(figsize=(10, 8))
# Use only the feature columns and the label column
corr_matrix = df_phish.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap (Phishing Data)')
plt.savefig('./visualizations/phish_correlation_heatmap.png')
plt.show()

#-------------------------Model training and evaluation------------------
#---Enron classification ML type1 text--
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

#Prepare Enron data for training
X_enron = df_enron['cleaned_text']
y_enron = df_enron['label']
X_train_en, X_test_en, y_train_en, y_test_en = train_test_split(X_enron, y_enron, test_size=0.2, random_state=42)

#Vectorization (Feature Extraction)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train_en)
X_test_vec = vectorizer.transform(X_test_en)

#Train Model A (Baseline: Naive Bayes)
model_nb = MultinomialNB()
model_nb.fit(X_train_vec, y_train_en)
preds_nb = model_nb.predict(X_test_vec)
print("--- Enron: Naive Bayes Report ---")
print(classification_report(y_test_en, preds_nb))
#Save metrics for the report

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

#---Phishing classification, clustering ML type2,3--
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score # For evaluating clustering

#Train Phishing Classifier (Model C: Gradient Boosting)
model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
model_gb.fit(X_train_scaled, y_train_ph)
preds_gb = model_gb.predict(X_test_scaled)
print("--- Phishing: Gradient Boosting Report ---")
print(classification_report(y_test_ph, preds_gb))
# Save metrics for the report

#Clustering (ML Type 2: Unsupervised Learning for complexity)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)
cluster_labels = kmeans.labels_

#Evaluate clustering alignment (measures how well clusters match true labels)
print(f"\n--- Phishing: K-Means Clustering Score (Adjusted Rand Index) ---")
print("Alignment Score:", adjusted_rand_score(y_train_ph, cluster_labels))

#Save the best model and scaler
joblib.dump(model_gb, './models/final_phish_model.joblib')
joblib.dump(scaler, './models/phish_scaler.joblib')