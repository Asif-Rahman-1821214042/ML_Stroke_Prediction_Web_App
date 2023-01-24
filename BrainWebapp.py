import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#loading the csv data to a Pandas DataFrame
brain_stroke = pd.read_csv('full_data_brain.csv')



dataChanginggender = {
    "Male" : 1,
    "Female" : 0
}

brain_stroke['gender'] = brain_stroke['gender'].map(dataChanginggender)




dataChangingmarrried = {
    "Yes" : 1,
    "No" : 0
}

brain_stroke['ever_married'] = brain_stroke['ever_married'].map(dataChangingmarrried)




dataChangingworktype= {
    "Private" : 0,
    "Self-employed" : 1,
    "Govt_job" : 2,
    "children" : 3
}

brain_stroke['work_type'] = brain_stroke['work_type'].map(dataChangingworktype)




dataChangingResidence_type= {
    "Urban" : 0,
    "Rural" : 1
}

brain_stroke['Residence_type'] = brain_stroke['Residence_type'].map(dataChangingResidence_type)


X = brain_stroke.drop(columns = ['stroke','smoking_status'], axis=1)
Y = brain_stroke['stroke']

s=SMOTE()
X_balanced, Y_balanced = s.fit_resample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_balanced, Y_balanced, stratify = Y_balanced, test_size = 0.2, random_state = 49) 

dtm = DecisionTreeClassifier(criterion = 'entropy', max_depth = 15)

dtm.fit(X_train, Y_train)

def brainStroke(input_data):
    input_np=np.asarray(input_data)
    input_rs=input_np.reshape(1,-1)
    prd=dtm.predict(input_rs)
    if(prd==0):
      return 'Patient has not stroke'
    else:
      return 'Patient has stroke'



def main():
    st.title('Brain Stroke Predictiom Web App')
    
    gender=st.text_input("Gender")
    age=st.text_input("Age")
    hypertension=st.text_input("Hypertension")
    heart_disease=st.text_input("Heart Disease")
    ever_married=st.text_input("Married Status")
    work_type=st.text_input("Work Type")
    Residence_type=st.text_input("Residence Type")
    avg_glucose_level=st.text_input("Avg Glucose")
    bmi=st.text_input("BMI")
    
    diagnosis= ''
    if st.button("Check"):
        diagnosis=brainStroke([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi])
        
    st.success(diagnosis)


if __name__=='__main__':
    main()