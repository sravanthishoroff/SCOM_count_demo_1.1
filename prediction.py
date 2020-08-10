import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re, pickle
le = LabelEncoder()
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import pickle
import random, json , requests


def doc_path(file_name):
    results = []
    df = pd.read_excel(file_name,encoding='latin-1')

    input_alarm_count = {"title":"Total count",
                            'count': df.shape[0]}
    results.append(input_alarm_count)

    df['Alert'].dropna(how='all')
    df['Alert Type(ignorable)'].dropna(how='all')

    # preprocessing
    df['Alert Type(ignorable)'] = df['Alert Type(ignorable)'].replace(to_replace=['NO','no','No(needs discussion)'],value='No')
    df['Alert Type(ignorable)'] = df['Alert Type(ignorable)'].replace(to_replace=['yes','yes(server decom)','yes( server decom)'],value='Yes')
    df.drop(df.loc[df['Alert Type(ignorable)'] == 'doubt(needs discussion)'].index,inplace=True)
    df.dropna(subset=['Alert Type(ignorable)'],inplace = True)
    df['Alert'] = df['Alert'].apply(lambda x: x.lower())
    df['Alert'] = df['Alert'].apply(lambda s: re.sub(r"[^a-zA-Z0-9]"," ",s))

    # lemmetization
    lemmatizer=WordNetLemmatizer()
    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        tokens =[w for w in tokens if not w in stop] # [w for w in
        lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in tokens])
        return lemmatized_output

    df['Alert'].apply(tokenize)

    # label_encoding
    le = preprocessing.LabelEncoder()
    alert = le.fit_transform(df['Alert'])

    tf = pickle.load(open("labelencoder.pkl", "rb"))
    
    #taking X and y
    X = alert.reshape(-1,1)
    y = df['Alert Type(ignorable)']
    # print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    ## Load model
    from sklearn.ensemble import VotingClassifier
    
    # load model 
    pkl_file = 'C:\\Users\\SRAVANTHI SHOROFF\\Desktop\\sravanthi\\SCOM_ML_v_1\\votingclassifierMod1.pkl'
    with open(pkl_file,'rb') as file:
        votingclassifier = pickle.load(file)

    # votingMod = votingMod.fit(X_train, y_train)
    
    # predict class
    y_pred = votingclassifier.predict(X)
    # print(y_pred)
    print(classification_report(y, y_pred))

    output = pd.DataFrame(columns=['y_pred'],data = y_pred)
    # print("output")
    output['pred_val'] = output['y_pred'].apply(lambda x:1 if (x) =="Yes" else 0)

    #Generate false alarm excel
    false_output = output[output['pred_val']==0]
    # print(false_output.shape)
    false_alarm_count = {"title":"False Alarm",
                            'count' : false_output.shape[0]}
    results.append(false_alarm_count)

    output_df_false = df[df.index.isin(false_output.index)]
    output_df_false.to_excel('false_alarm_report.xls')

    #Generating true alarms from predicted output
    true_output = output[output['pred_val']==1]
    # print(true_output.shape)
    true_alarm_count = {'title':"True Alarm ",
                            'count':true_output.shape[0]}
    results.append(true_alarm_count)

    output_df_true =df[df.index.isin(true_output.index)]
    # print(output_df_true)
    print(output_df_true.shape)

    data = {'result':results}

    return data
