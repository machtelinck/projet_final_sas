import streamlit as st
import pandas as pd


import numpy as np



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

from sklearn import tree, linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import *
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pickle




@st.cache(suppress_st_warning=True)


def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction'])
if app_mode=='Home':
    st.title('LOAN PREDICTION :')  
    st.markdown('Dataset :')
    data=pd.read_csv('df_clear.csv')
    st.write(data.head())
    st.markdown('Interior_space VS price')
    st.bar_chart(data[['interior_space','price']].head(20))


elif app_mode == 'Prediction':

    st.write('''
  # La prédiction du prix d'une maison
  ''')
    st.sidebar.header("Entrez les paramètres de votre maison : ")



    def user_input():
        bedrooms=st.sidebar.slider("Nombre de chambres",1,11,3)
        bathrooms=st.sidebar.slider("Nombre de salles de bains, où 0,5 correspond à une chambre avec toilettes mais sans douche",2,8,3)
        interior_space=st.sidebar.slider("L'espace de vie intérieur",290,13540,1910)
        floors=st.sidebar.slider('Nombre etage ',1, 13, 7)
        see_sea=st.sidebar.checkbox("Vue sur la mer : Oui/No", 0,1)
        view_qual=st.sidebar.slider("Vue, un indice de 0 à 4 de la qualité de la vue de la propriété",0,4,0)
        grade=st.sidebar.slider('Niveau de qualité de la construction ',1, 13, 7)
        space_above=st.sidebar.slider("L'espace intérieur du logement qui est au-dessus du niveau du sol",290,9410,1560)
        sqft_basement=st.sidebar.slider("La superficie en pieds carrés de l'espace intérieur du logement qui est sous le niveau du sol",0, 3260, 0)
        yr_renovated=st.sidebar.slider('annee de renovation',1, 13, 7)
        lat=st.sidebar.slider('Lattitude',47.1559, 47.7776, 47.1559)
        space_int_15=st.sidebar.slider('space int 15 ',1, 13, 7)
        zipcode_98004=st.sidebar.checkbox('La code postale de la maison est 98004', 0,1)
        zipcode_98006=st.sidebar.checkbox('La code postale de la maison est 98006', 0,1)
        zipcode_98039=st.sidebar.checkbox('La code postale de la maison est 98039', 0,1)
        zipcode_98040=st.sidebar.checkbox('La code postale de la maison est 98040', 0,1)
        zipcode_98112=st.sidebar.checkbox('La code postale de la maison est 98112', 0,1)

      



        data={
      'bedrooms':bedrooms,
      'bathrooms':bathrooms,
      'interior_space':interior_space,
      'floors':floors,
      'see_sea' : see_sea,
      'view_qual':view_qual,
      'grade':grade,
      'space_above':space_above,
      'sqft_basement':sqft_basement,
      'yr_renovated':yr_renovated,
      'lat':lat,
      'space_int_15':space_int_15,
      'zipcode_98004': zipcode_98004,
      'zipcode_98006': zipcode_98006,
      'zipcode_98039': zipcode_98039,
      'zipcode_98040': zipcode_98040,
      'zipcode_98112': zipcode_98112,
      }

        maison_parametres=pd.DataFrame(data,index=[0])

        return maison_parametres
    df=user_input()

    

    st.subheader("Le prix de la maison est:")
    pickle_in = open('pickle_model.pkl', 'rb') 
    my_pipe_lasso = pickle.load(pickle_in)
    st.write((my_pipe_lasso.predict(df))[0])
    



