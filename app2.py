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
        sqft_living=st.sidebar.slider("L'espace de vie intérieur",290,13540,1910)
        grade=st.sidebar.slider('Niveau de qualité de la construction ',1, 13, 7)
        sqft_above=st.sidebar.slider("L'espace intérieur du logement qui est au-dessus du niveau du sol",290,9410,1560)
        sqft_living15=st.sidebar.slider("La superficie de l'espace de vie intérieur du logement pour les 15 voisins les plus proches" ,399,6210, 1490)
        bathrooms=st.sidebar.slider("Nombre de salles de bains, où 0,5 correspond à une chambre avec toilettes mais sans douche",2,8,3)
        view=st.sidebar.slider("Vue, un indice de 0 à 4 de la qualité de la vue de la propriété",0,4,0)
        sqft_basement=st.sidebar.slider("La superficie en pieds carrés de l'espace intérieur du logement qui est sous le niveau du sol",0, 3260, 0)
        bedrooms=st.sidebar.slider("Nombre de chambres",1,11,3)
        lat=st.sidebar.slider('Lattitude',47.1559, 47.7776, 47.1559)
        see_sea=st.sidebar.checkbox("Vue sur la mer : Oui/No", 0,1)

        zipcode_98004=st.sidebar.checkbox('La code postale de la maison est 98004', 0,1)
        zipcode_98039=st.sidebar.checkbox('La code postale de la maison est 98039', 0,1)
        zipcode_98040=st.sidebar.checkbox('La code postale de la maison est 98040', 0,1)
        zipcode_98006=st.sidebar.checkbox('La code postale de la maison est 98006', 0,1)
        floors=st.sidebar.slider('Nombre etage ',1, 13, 7)
        yr_renovated=st.sidebar.slider('annee de renovation',1, 13, 7)
        space_int_15=st.sidebar.slider('space int 15 ',1, 13, 7)
        zipcode_98112=st.sidebar.checkbox('La code postale de la maison est 98112', 0,1)

      



        data={
      'bathrooms':bathrooms,
      'bedrooms':bedrooms,
      'interior_space':sqft_living,
      'floors':floors,
      'see_sea' : see_sea,
      'grade':grade,

  
      'space_above':sqft_above,
    
      'view_qual':view,
      'sqft_basement':sqft_basement,
      'yr_renovated':yr_renovated,
      'lat':lat,
      'space_int_15':space_int_15,
      'zipcode_98004': zipcode_98004,
      'zipcode_98039': zipcode_98039,
      'zipcode_98006': zipcode_98006,
      'zipcode_98040': zipcode_98040,
      'zipcode_98112': zipcode_98112,
      
      

      }

        maison_parametres=pd.DataFrame(data,index=[0])
        
        return maison_parametres
    df=user_input()

    st.subheader('On veut calculer le prix de cette maison')
    st.write(df)
    X = pd.read_csv("df_cleared.csv")
    y = pd.read_csv("df_clear.csv")
    y = y["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model1 = make_pipeline(StandardScaler(), Ridge())
    print 
    model1.fit(X_train, y_train)

    prediction = model1.predict(df)
    columns = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms',
        'view', 'sqft_basement', 'bedrooms', 'lat', 'waterfront', 'house_age','zipcode_98001', 'zipcode_98002', 'zipcode_98003', 'zipcode_98004',
          'zipcode_98005', 'zipcode_98006', 'zipcode_98007', 
        'zipcode_98008', 'zipcode_98010', 'zipcode_98011', 'zipcode_98014', 
      'zipcode_98019', 'zipcode_98022', 'zipcode_98023', 'zipcode_98024', 'zipcode_98027', 'zipcode_98028', 'zipcode_98029', 'zipcode_98030', 'zipcode_98031', 'zipcode_98032', 'zipcode_98033', 'zipcode_98034', 'zipcode_98038', 'zipcode_98039', 'zipcode_98040', 
      'zipcode_98042', 'zipcode_98045', 'zipcode_98052', 'zipcode_98053', 'zipcode_98055', 'zipcode_98056', 'zipcode_98058', 'zipcode_98059', 'zipcode_98065', 'zipcode_98070', 'zipcode_98072', 'zipcode_98074', 'zipcode_98075', 'zipcode_98077', 'zipcode_98092',
      'zipcode_98102', 'zipcode_98103', 'zipcode_98105', 'zipcode_98106', 'zipcode_98107', 'zipcode_98108', 'zipcode_98109', 'zipcode_98112', '9zipcode_8115', 'zipcode_98116', 'zipcode_98117', 'zipcode_98118', 'zipcode_98119', 'zipcode_98122', 'zipcode_98125',
      'zipcode_98126', 'zipcode_98133', 'zipcode_98136', 'zipcode_98144', 'zipcode_98146', 'zipcode_98148', 'zipcode_98155', 'zipcode_98166', 'zipcode_98168', 'zipcode_98177', 'zipcode_98178', 'zipcode_98188', 'zipcode_98198', 'zipcode_98199'
        'zipcode_98004', 'zipcode_98039', 'zipcode_98040']
    st.subheader("Le prix de la maison est:")
    st.write(int(prediction)/1000)



