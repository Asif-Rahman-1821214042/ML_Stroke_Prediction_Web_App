import numpy as np
import pickle
import streamlit as st

#load model
loaded_model= pickle.load(open('model.sav','rb'))

#stroke prediction function
def brainStroke(input_data):
    input_np=np.asarray(input_data,dtype=object)
    input_rs=input_np.reshape(1,-1)
    try:
      prd=loaded_model.predict(input_rs)
    except:
      return 'Fill the required space or use correct string'
    #giving result according to condition
    if(prd==0):
      return 'Patient has not stroke'
    else:
      return 'Patient has stroke'

def main():
    st.title('Brain Stroke Prediction Web App')
    gender=st.radio("Gender",["Male","Female"])
    age=st.text_input("Age")
    hypertension=st.selectbox("Hypertension",["Yes","No"])
    heart_disease=st.selectbox("Heart Disease",["Yes","No"])
    ever_married=st.selectbox("Married Status",["Yes","No"])
    work_type=st.selectbox("Work Type",["Private","Self-employed","Children","Govt_job"])
    Residence_type=st.selectbox("Residence Type",["Urban","Rural"])
    avg_glucose_level=st.text_input("Avg Glucose")
    bmi=st.text_input("BMI")
    diagnosis= ''
    if st.button("Check"):
        #for gender conversion
        if gender == 'Male':
            gender = 1
        else:
            gender = 0
            
        # for hypertension conversion
        if hypertension == 'Yes':
            hypertension = 1
        else:
            hypertension = 0

        # for heart_disease conversion
        if heart_disease == 'Yes':
            heart_disease = 1
        else:
            heart_disease = 0

        # for ever_married conversion
        if ever_married == 'Yes':
            ever_married = 1
        else:
            ever_married = 0

        # for Residence_type conversion
        if Residence_type == 'Urban':
            Residence_type = 0
        else:
            Residence_type = 1

        #for work_type conversion
        if work_type == 'Private':
            work_type = 0
        elif work_type == 'Self-employed':
            work_type = 1
        elif work_type == 'Children':
            work_type = 2
        else:
            work_type = 3

        diagnosis=brainStroke([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi])
    st.success(diagnosis)


if __name__=='__main__':
    main()