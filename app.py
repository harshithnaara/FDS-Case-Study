import joblib
import pandas as pd
import streamlit as st

bundle = joblib.load('titanic_model.pkl')
model = bundle['model']
columns = bundle['columns']

st.title('Titanic Survival Prediction')

pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Gender', ['male', 'female'])
age = st.number_input('Age', min_value=0.0, max_value=100.0, value=bundle['age_median'])
sibsp = st.number_input('Siblings / Spouses Aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('Parents / Children Aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

if st.button('Predict'):
    row = pd.DataFrame([{
        'Pclass': pclass,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Sex_male': int(sex == 'male'),
        'Embarked_Q': int(embarked == 'Q'),
        'Embarked_S': int(embarked == 'S'),
    }])[columns]
    probability = model.predict_proba(row)[0, 1]
    st.write('Survival probability: {:.1f}%'.format(probability * 100))
    st.write('Prediction: ' + ('Survived' if probability >= 0.5 else 'Did not survive'))
