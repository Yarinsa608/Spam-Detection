import pandas as pd


#nltk.download("stopwords") add to the req file
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

#By default What makes a spam email
#Theres not always going to be JUST TEXT
#For this reason having a background OCR might be helpful to read over just to check if there is anything that might have been missed



#Loa
#https://www.geeksforgeeks.org/nlp/removing-stop-words-nltk-python/
#Could hardcode
#MAYEVB also sir,mam , or occupant
stop_words=set(stopwords.words('english'))

#Load dataset
email_df=pd.read_csv("enron_spam_data.csv")


#Preprocess Data
#Handle Null values

print(email_df.head())

#Data cleaning

#Remove duplicates

#Align fields and data structures for consistency


#Data transformation

#Data scaling

#normlisation


#data categorisation



#Convert Spam/Ham column to boolean
value_detected="spam"
email_df.fillna(value_detected)#Fill any column having NA/null data
email_df.drop_duplicates()#Remove any duplicate rows

spam_detected=email_df.query('`Spam/Ham` == @value_detected')

#Create new table? so we can split them?

df=email_df



def preprocess_text(text):
    if pd.isnull(text):
        return 'invalid'
    text=text.lower()#Convert to lowercase
    #BUTR WHAT AQBOUT SPAM EMAILS THAT ARE ALL IN CAPS THATS IMMEDIATE
    














#print(f"Total spam messages: {len(spam_detected)}")
#df.plot()
#plt.show()

#cheatsheet
#https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf


