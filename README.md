# Spam-Detection
Session 15 Group 9 Assignment 2
 How to Run the Code
The project is executed using two main Python scripts that must be run sequentially from the command line.

❗ IMPORTANT: Ensure your terminal is positioned in the root directory of the project (e.g., AI4Cyber_Assignment2/) before running any commands.

Step 1: Execute Data Processing (data_processing.py)

This script handles all data cleaning, feature engineering, and scaling. It is crucial to run this first, as it prepares the input files for the model training stage and creates the necessary output directories (data/, models/, visualizations/).

Bash
python3 data_processing.py
Outcome:

The raw data is processed and a combined intermediate file is saved (./data/processed_enron.csv).

The StandardScaler for the numerical Phishing features is fitted and saved (./models/phish_scaler.joblib).

Step 2: Execute Model Training and Visualization (model_training.py)

Once the data is processed and saved, run the second script. This will load the processed data, generate all required visualizations, train all the machine learning models, and print the final evaluation metrics.

Bash
python3 model_training.py
Outcome:

All required charts (e.g., class distribution, box plots, correlation heatmap) are generated and saved as .png files in the ./visualizations/ folder.

The final trained models (Logistic Regression, Gradient Boosting) and the TF-IDF vectorizer are saved to the ./models/ folder.

Model performance reports (Classification Reports and Adjusted Rand Index) are displayed in the terminal.

Required Libraries
To successfully run both the data processing and model training scripts, you need the following Python libraries.

You can install all required packages at once using pip:

Bash
pip install pandas scikit-learn matplotlib seaborn wordcloud joblib
Library	Purpose
pandas	Essential for loading, cleaning, and manipulating dataframes (e.g., loading enron_spam_data.csv).
scikit-learn (sklearn)	The core machine learning framework, providing all models (LR, MNB, GBC, KMeans) and preprocessing tools (StandardScaler, TfidfVectorizer).
matplotlib & seaborn	Used for creating all visualizations (Bar charts, Box plots, Correlation Heatmap).
wordcloud	Used specifically for generating the visual representation of the most common words in spam emails.
joblib	Used for efficiently saving and loading trained models and preprocessing objects (StandardScaler, TfidfVectorizer) to the ./models directory.
re	Used for regular expressions, specifically in the data cleaning process to remove email headers and non-alphanumeric characters.
