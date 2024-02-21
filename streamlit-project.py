import streamlit as st
from joblib import load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import email
import string
from bs4 import BeautifulSoup
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve
import os
import warnings
from PIL import Image
warnings.filterwarnings('ignore')
np.random.seed(49)


##
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
class email_to_clean_text(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        text_list = []
        for mail in X:
            b = email.message_from_string(mail)
            body = ""

            if b.is_multipart():
                for part in b.walk():
                    ctype = part.get_content_type()
                    cdispo = str(part.get('Content-Disposition'))

                    # skip any text/plain (txt) attachments
                    if ctype == 'text/plain' and 'attachment' not in cdispo:
                        body = part.get_payload(decode=True)  # get body of email
                        break
            # not multipart - i.e. plain text, no attachments, keeping fingers crossed
            else:
                body = b.get_payload(decode=True) # get body of email
            #####################################################
            soup = BeautifulSoup(body, "html.parser") #get text from body (HTML/text)
            text = soup.get_text().lower()
            #####################################################
            text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE) #remove links
            ####################################################
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text, flags=re.MULTILINE) #remove email addresses
            ####################################################
            text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
            ####################################################
            text = ''.join([i for i in text if not i.isdigit()]) # remove digits
            ####################################################
            stop_words = stopwords.words('english')
            words_list = [w for w in text.split() if w not in stop_words] # remove stop words
            ####################################################
            words_list = [lemmatizer.lemmatize(w) for w in words_list] #lemmatization
            ####################################################
            words_list = [stemmer.stem(w) for w in words_list] #Stemming
            text_list.append(' '.join(words_list))
            #st.write(words_list)
            #st.write(text_list)
        #st.write(text_list)
        string_mail = ' '.join(text_list)
        wordcloud = WordCloud(background_color='white').generate(string_mail)
        # Display the generated image:
        st.write("### Word Cloud")
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off') # Remove axes
        st.pyplot(fig)

        dataframe = pd.read_csv("features.csv")
        first_column = dataframe.iloc[:, 0]
        liste = first_column.tolist()
        #st.write(liste)
        
        intersection = set(liste).intersection(set(words_list))
        text = ' '.join(intersection)
        #st.write(text)
        wordcloud = WordCloud(background_color='white').generate(text)
        # Display the generated image:
        st.write("### Intersection Word Cloud with Important Features Words with Mail Words")
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off') # Remove axes
        st.pyplot(fig)


        # Convert the result back to a list (if needed)
        intersection_list = list(intersection)


        return text_list
        
##


# Load your pre-trained pipeline
pipeline = load('model_pipeline.joblib')

#from ChatGPT
#my_pipeline = Pipeline(steps=[
#    ('text', email_to_clean_text()),  # Assuming this is a custom transformer
#    ('vector', vectorizer),           # Your vectorizer
#    ('model', rfc)                    # Your model, e.g., RandomForestClassifier
#])# Step 1: Create a truncated pipeline for the 'text' step
text_step_pipeline = Pipeline(steps=[
    ('text', email_to_clean_text())
])

#end ChatGPT

# Set up your Streamlit interface
st.header('My SpApp')
st.subheader('This app uses the (almost) latest LLM to detect Spam mail.')
st.subheader('Check your mail here before opening it!')

# Assuming the input is text. Adjust based on your model's input

how = st.radio(
    "How do you want to input your mail?", ["Type / Paste", "Upload"])

if how == "Type / Paste":
    user_input = st.text_input("Type or paste your mail for prediction...")
    #st.write(user_input)
else:
    uploaded_file = st.file_uploader("...or upload mail as text file")
    if uploaded_file is not None:
        user_input = uploaded_file.read().decode("utf-8")#selbes Format wie oben?
        #st.write(user_input)

# Button to make predictions
if st.button('Predict'):
    # Make sure there is input before making a prediction
    #if user_input:
    #if user_input.lower() == "this is ham":
    #    st.write(f'Prediction: Mail is ham.')
    #    st.image("Ham.png")
        #cleaned_texts = pipeline.named_steps['text'].transform(user_input)
        #st.write(cleaned_texts)
        #plot_WordCloud(cleaned_texts)
        #st.write(words)

    if user_input: 
        # Use the pipeline to make predictions
        prediction = pipeline.predict([user_input]) 
        # Display the prediction
        if prediction[0] == 0:
            st.subheader(f'Prediction: Mail is ham.')
            #st.image("Ham.png")
            image = Image.open("Ham.png")
            st.image(image, width=500)  # Adjust the width as needed
            #word cloud
            

            

        elif prediction[0] == 1:
            st.subheader(f'Prediction: Mail is spam.')
            image = Image.open("Spam.png")
            st.image(image, width=500)  # Adjust the width as needed

            
        else:
            st.write("Please enter input for prediction.")

# Transform  data using the truncated pipeline
#transformed_email = text_step_pipeline.transform(user_input)# Step 3: Print the output
#st.write(transformed_email)
#print(transformed_emails)
#string_mail = ' '.join(transformed_email)
#wordcloud = WordCloud(background_color='white').generate(string_mail)
# Display the generated image:
#st.write("### Word Cloud")
#fig, ax = plt.subplots()
#ax.imshow(wordcloud, interpolation='bilinear')
#ax.axis('off') # Remove axes
#st.pyplot(fig)

##
# Assuming you have raw email data stored in `raw_emails`
# and email_to_clean_text is properly implemented as a transformer








