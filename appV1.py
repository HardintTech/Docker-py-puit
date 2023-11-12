#Import affichage site
import pandas as pd
import streamlit as st
import plotly.express as px
import json
from urllib.request import urlopen
import matplotlib.pyplot as plt

#Import programme
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import numpy as np
import os

# On charge les données 
df = pd.read_csv('./data/data_tuto_streamlit.csv', sep = ';')
with urlopen('https://france-geojson.gregoiredavid.fr/repo/departements.geojson') as f : 
    geo_dep = json.load(f)


state_id_map = {}
for feature in geo_dep['features']:
 feature['id'] = feature['properties']['code']
 state_id_map[feature['properties']['nom']] = feature['id']


#Fonction qui prend en paramètre un puit et un pattern et qui renvoie le nombre de fois qu'il trouve le pettern dans le puit
 def match(wellFile, wellLegend):
  img_rgb = cv.imread(wellFile)
  assert img_rgb is not None, "file could not be read, check with os.path.exists()"
  img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
  template = cv.imread(wellLegend)
  assert template is not None, "file could not be read, check with os.path.exists()"

  template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
  _, template_trans = cv.threshold(template_gray, 200, 255, cv.THRESH_BINARY_INV)
#   scale_percent = 40 # percent of original size
#   width = int(template_gray.shape[1] * scale_percent / 100)
#   height = int(template_gray.shape[0] * scale_percent / 100)
#   dim = (width, height)
#   resized = cv.resize(template_gray, dim)
#   w, h = resized.shape[::-1]


#   res = cv.matchTemplate(img_gray,resized,cv.TM_CCOEFF_NORMED)
#   threshold = 0.8
#   loc = np.where( res >= threshold)
#   cpt = 0

  scale_percent = 40 # percent of original size
  threshold = 0.8
  scale_max = scale_percent
  cpt_max = 0
  for i in range(6):
    cpt = 0
    width = int(template_trans.shape[1] * scale_percent / 100)
    height = int(template_trans.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(template_trans, dim)
    w, h = resized.shape[::-1]
    # st.image(img_gray)
    # st.image(resized)
    res = cv.matchTemplate(img_gray,resized,cv.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
      cpt += 1
    if (cpt_max < cpt):
      cpt_max = cpt
      scale_max = scale_percent
    scale_percent += 5
  width = int(template_trans.shape[1] * scale_max / 100)
  height = int(template_trans.shape[0] * scale_max / 100)
  dim = (width, height)
  resized = cv.resize(template_trans, dim)
  w, h = resized.shape[::-1]
  
  res = cv.matchTemplate(img_gray,resized,cv.TM_CCOEFF_NORMED)
  for pt in zip(*loc[::-1]):
      cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    #   cpt += 1
#   txt = wellFile
#   x = txt.split("/")
#   ans=""
#   for y in x[1:-1]:
#       ans += "/" + y
#   ans += "/res.png"
#   print(cpt)
#   cv.imwrite(ans,img_rgb)
  return cpt,img_rgb


t1, t2, t3 = st.columns((0.05,0.1,0.05)) 


t1.image('./data/cytech.png', width = 120)

t2.title('Data Battle 2023')
t3.image('./data/iapau.png', width = 120)
st.header('Dashboard de visualisations des puits')
viz, comparatif = st.tabs(['Puits à analyser', 'Elements trouvé sur le puit'])
with viz : 
    # La selectbox prend en paramètres des Dataframe alors ne nous privons pas ! 
    # Je crée un sous dataframe qui contient seulement les colonnes d'indicateurs
    list_indicateurs = df.columns[2:]
    # J'ajoute une colonne qui sera celle par défaut : celle qui n'affiche rien
    list_indicateurs = list_indicateurs.insert(0, "Aucun")
    # Je crée la selectbox ET je stocke la valeur dans une variable (important pour la suite)
    indicateur = st.selectbox('Choisissez un puit', list_indicateurs)
    if indicateur != "Aucun" : 

        #Permet de récupérer dans la liste result le nombre de fois que chaque pattern apparait et dans labels leur nom associé
        
        file = indicateur + ".PNG"
        result = []
        labels = []
        for filename in os.listdir("Legende_short_15_2_1"):
        # Check if the file ends with ".png"
            if filename.endswith(".PNG"):
            # Do something with the file, for example print its name
                result.append(match(file,"Legende_short_15_2_1/"+ filename)[0])
                if (match(file,"Legende_short_15_2_1/"+ filename)[0] != 0):
                    labels.append(filename.rstrip(".PNG"))

        while 0 in result:
            result.remove(0)

        # Tracer un diagramme circulaire de la distribution des prédictions
        
        st.title('Distribution des prédictions sur l\'ensemble de test')
        fig, ax = plt.subplots()
        ax.pie(result, labels=labels,autopct='%1.1f%%')
        ax.axis('equal')
        ax.legend(labels,title='Legend')
        st.pyplot(fig)
        
with comparatif : 
    if indicateur != "Aucun" : 
        file = indicateur + ".PNG"
        for filename in os.listdir("Legende_short_15_2_1"):
            # Check if the file ends with ".png"
                if filename.endswith(".PNG"):
                # Do something with the file, for example print its name
                    if (match(file,"Legende_short_15_2_1/"+ filename)[0] != 0):
                        st.image(match(file,"Legende_short_15_2_1/"+ filename)[1])
            







