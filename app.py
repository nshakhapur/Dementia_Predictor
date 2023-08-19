import streamlit as st
import numpy as np
import sklearn
import pandas as pd

st.title("Dementia Predictor")
st.subheader("Used Machine Learning to compute the Dementia using Medical Report")
st.markdown("## Enter the Patient Details")

st.markdown("### Enter the Patient Personal Details: ")




visit=st.number_input("Number of times you have visited neurologist: ",step=1)
MRD = st.number_input("Time taken to difference in MRI Scans (in minutes):")
    
gender = st.radio("Gender:", ['Male', 'Female'])
if gender == 'Male':
        gen = 0
else:
        gen = 1
    
Age = st.number_input("Age:",step=1)
educ = st.number_input("Age when completed Education (12-25):",step=1)
if educ>23:
        educ=23 #Done in accordance to the dataset
ses = st.slider("Socioeconomic Status Number:",1,5)
mmse = st.number_input("Mini-Mental State Examination Score:")
 
cdr = st.slider("How Forgetful are you (0-Nil and 2 Highly)",0,2)
tiv = st.number_input("Estimated Total Intracranial Volume as mentioned in MRI:")
wbv = st.number_input("Normalized Whole Brain Volume as mentioned in MRI:")
asf = 1.20

predict_button = st.button("Predict")



if predict_button:
  dataset = pd.read_csv('dementia_dataset.csv')
  dataset=dataset.drop(['Subject ID','MRI ID','Hand'],axis=1)


  dataset.replace({'Group':{'Nondemented':0,'Demented':1,'Converted':2}},inplace=True)

  dataset.replace({'M/F':{'M':0,'F':1}},inplace=True)
  dataset = dataset.dropna()

  X = dataset.iloc[:, 1:].values
  y = dataset.iloc[:, 0:1].values

  y=y.ravel()

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  from sklearn.svm import SVC
  classifier = SVC(kernel = 'rbf', random_state = 0)
  classifier.fit(X_train, y_train)

  #1,0,0,68,12,2,27,0.5,1457,0.806,1.205

  res=classifier.predict(sc.transform([[visit, MRD, gen, Age, educ, ses, mmse, cdr, tiv, wbv, asf]]))

  st.markdown("## Prediction Result:")

  if res == 0:
    st.markdown("### Patient is Non Demented")
  else:
    st.markdown("### Patient is Demented")

  
