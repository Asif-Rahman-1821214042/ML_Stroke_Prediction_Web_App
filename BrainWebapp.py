import numpy as np
import pickle
import streamlit as st

loaded_model= pickle.load(open('D:/Model/model.sav','rb'))


def brainStroke(input_data):
    input_np=np.asarray(input_data)
    input_rs=input_np.reshape(1,-1)
    prd=loaded_model.predict(input_rs)
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