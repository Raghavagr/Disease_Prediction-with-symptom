import pickle
import streamlit as st
import pandas as pd
import numpy as np

st.title("Disease Prediction Using Symtoms observed")

#create navigation bar
nav = st.sidebar.radio("Navigation", ["Home", "Description"])

@st.cache(allow_output_mutation = True)
def load_model():
    with open("disease_predDT.pkl","rb") as f:
        model = pickle.load(f)

    with open("symptoms.pkl", "rb") as f:
        symptoms = pickle.load(f) 
    
    with open("diseases.pkl", "rb") as f:
        diseases = pickle.load(f) 

    with open("diseaseWithSymp.pkl", "rb") as f:
        disease_symptom = pickle.load(f) 

    return model, symptoms, diseases, disease_symptom

with st.spinner("Loading Files..."):
        model, symptoms, diseases, disease_symptom = load_model()

if nav == "Home":
    st.header("Enter the symptoms which you observe")
    sort_symptom = sorted(symptoms)
    sympton1 = st.selectbox("select symptom-1", sort_symptom)

    symptom2 = st.selectbox("select symptom-2", sort_symptom)

    symptom3 = st.selectbox("select symptom-3", sort_symptom)

    symptom4 = st.selectbox("select symptom-4", sort_symptom)

    symptom5 = st.selectbox("select symptom-5", sort_symptom)

    key_list = list(diseases.keys())
    val_list = list(diseases.values())
    cols_dict = {}
    for sympt in symptoms:
        if(sympt == sympton1 or sympt == symptom2 or sympt == symptom3 or sympt==symptom4 or sympt == symptom5):
            cols_dict[sympt] = 1
        else:
            cols_dict[sympt] = 0

    df = pd.DataFrame(cols_dict, columns=cols_dict.keys(), index=cols_dict.values())
    test = np.array(df.iloc[0])
    output = model.predict([test])

    #extract the name of disease which predicted feom list.
    position = val_list.index(output)
    final_output = key_list[position]
    st.write("\n")
    text = "<strong><span style='color:green'>Accroding the symptoms predicted disease can be</span></strong>"
    st.markdown(text, unsafe_allow_html=True)
    st.write(final_output)
    st.write("\n")
    st.write("\n")
    st.write("\t\t Project made with ❤️ by Raghav Agrawal")

if nav == "Description":
    st.header("Disease Description with symptoms")
    st.text("showing the complete file with symptoms")
    st.write(disease_symptom)
    st.write("\n")
    st.write("\t\tThank You!")
    st.write("\t\tProject made with ❤️ by Raghav Agrawal")