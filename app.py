####################################################################
# import massif 
####################################################################
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
import pickle  #to load a saved model
import base64  #to open .gif files in streamlit app
import numpy as np # linear algebra
import pandas as pd 
import sklearn
from subprocess import check_output
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve



@st.cache(suppress_st_warning=True)


def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value
df = pd.read_csv('df_cleared.csv') 
df1 = pd.read_csv('df_clear.csv') 

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value




app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) 
####################################################
#    Partie Home                                                                        #
####################################################
if app_mode=='Home':
    st.title('LOAN PREDICTION :')  
    st.markdown('Dataset :')
    data=pd.read_csv('df_clear.csv')
    st.write(data.head())
    st.markdown('Interior_space VS price')
    st.bar_chart(data[['interior_space','price']].head(20))

###############################################################
################ partie prédiction 
#################################################################
elif app_mode == 'Prediction':

#############################################
############## ici on prend de l'utilisateur les paramètre il donne
####################################""
    st.subheader('vous devez completez toutes les données pour avoir une estimation du prix de la maison ')
    st.sidebar.header("rentrez les informations :")


    

    zipcode=st.sidebar.selectbox('le ZIP code',('98001', '98002', '98003', '98004', '98005', '98006', '98007', '98008', '98010', '98011', '98014', 
    '98019', '98022', '98023', '98024', '98027', '98028', '98029', '98030', '98031', '98032', '98033', '98034', '98038', '98039', '98040', 
    '98042', '98045', '98052', '98053', '98055', '98056', '98058', '98059', '98065', '98070', '98072', '98074', '98075', '98077', '98092',
     '98102', '98103', '98105', '98106', '98107', '98108', '98109', '98112', '98115', '98116', '98117', '98118', '98119', '98122', '98125',
     '98126', '98133', '98136', '98144', '98146', '98148', '98155', '98166', '98168', '98177', '98178', '98188', '98198', '98199'))

    latitude= st.sidebar.number_input("latitude")
    longitude= st.sidebar.number_input("longtitude")
    note_condition=st.sidebar.slider('condition',0,4,0,)
    note_grade=st.sidebar.slider('note attribué',0,4,0,)# non , il faut faire un dictionnaire puis du one hot encoding
    note_vue=st.sidebar.slider('note de la vue',0,4,0,)   # non , il faut faire un dictionnaire puis du one hot encoding
    view_qual =st.sidebar.selectbox("viewqual" , ( 'view_qual_0',
       'view_qual_1', 'view_qual_2', 'view_qual_3', 'view_qual_4',))
    
    interior_space=st.sidebar.slider('espace interieur',0,1000000,0,)
    space_above=st.sidebar.slider('espace au dessus du sol',0,10000,0,)
    space_below=st.sidebar.slider('espace en dessous du sol',0,1000000,0,)
    space_int_15=st.sidebar.slider('space_int_15',0,1000000,0,)
    space_land_15=st.sidebar.slider('space_land_15',0,1000000,0,)
    land_space=st.sidebar.slider('land_space',0,1000000,0,)
    
    floors=st.sidebar.slider('etage',0,10,0,)
    nombre_chambre=st.sidebar.slider('nombre de chambre',0,100,0,) 
    nombre_salle_de_bain=st.sidebar.slider('nombre de salle de bain',0,100,0,)

   
    house_age=st.sidebar.slider('house_age',0,1000,0,)
    year_renovated=st.sidebar.slider('year_renovated',0,1000,0,)
    year_build=st.sidebar.slider('yearbuild ',0,3000,0,)
    
   
    

#######################################
# faire un bouton pour confirmer les données
########################################
  
  #### Rajout de donnée de notre modèle ##########  

    ###########################################################################################
#####  Rajout de commentaire en fonction du sidebar
    ############################################################################################
    if note_grade:
        st.write("kamehameha")
    elif note_vue:
        st.write("genkidama")
    else:
        value = "No value selected"
    if int(space_above)>5000:
        st.write("alors,comme ça on est riche?")

######################################
# Traitement des informations #
#####################################

     









#########################
# partie commune aux deux 
########################

df1 = pd.read_csv('kc_house_data.csv')
df1['lon'] = df1['long']
columns=['latitude', 'longitude']
st.map(df1)
# une map de la zone 