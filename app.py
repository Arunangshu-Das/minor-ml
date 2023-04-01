from flask import Flask
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from flask import request
from flask_cors import CORS

app=Flask(__name__)

CORS(app)

@app.route('/',methods = ['GET'])
def welcome():
    return "Minor project 1)Arunangshu 2)Anik 3)Ratul 4)Kisalay"

@app.route('/search',methods = ['POST'])
def finding():
    try:
        x=request.get_json()
        x=x['data']
        x=x.lower()
        symptom=pd.read_csv('symptom.csv')
        model=pickle.load(open('final_model.h5','rb'))
        symptom['Symptom'][0]=' '
        user_input=x.split(',')
        if len(user_input)<17:
            a=17-len(user_input)
            for i in range(a):
                user_input.append(' ')
        else:
            user_input=user_input[:17]
        for i in range(17):
            user_input[i]=symptom['Encoded_value'][symptom['Symptom']==user_input[i]].values[0]
        symptom['Encoded_value'][symptom['Symptom']=='cold_hands_and_feets'].values[0]
        df=pd.read_csv('dataframe.csv')
        df.drop(['Unnamed: 0','Disease'],axis=1,inplace=True)
        df=df.append(pd.DataFrame([user_input],columns=['Symptom_1', 'Symptom_2', 'Symptom_3','Symptom_4', 'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8','Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13','Symptom_14', 'Symptom_15', 'Symptom_16', 'Symptom_17']),ignore_index=True)

        df['idx']=df.index
        df1=pd.DataFrame(df.sample(frac=1).reset_index())
        idx=df1[df1['idx']==4920].index[0]
        if idx<20:
            df1=df1[0:50]
        else:
            df1=df1[idx-20:idx+20]
        df1.drop(['index','idx'],axis=1,inplace=True)
        imputer=KNNImputer(n_neighbors=20)
        df=imputer.fit_transform(df1)
        df=pd.DataFrame(df)
        df.columns=['Symptom_1', 'Symptom_2', 'Symptom_3','Symptom_4', 'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8','Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13','Symptom_14', 'Symptom_15', 'Symptom_16', 'Symptom_17']

        for i in df.columns:
            df[i] = df[i].astype(np.int64)
        a=pd.DataFrame(df1==user_input)
        a=pd.DataFrame([a['Symptom_1']==True]).T
        idx=a[a['Symptom_1']==True].index[0]-df1.iloc[:1].index[0]
        return model.predict(pd.DataFrame(df.iloc[idx]).T)[0]
    except:
        return "Error"
