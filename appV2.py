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
import pytesseract
from pdf2image import convert_from_path
import PyPDF2 as pyPdf 
import io

# On charge les données 
df = pd.read_csv('./data/data_tuto_streamlit.csv', sep = ';')
with urlopen('https://france-geojson.gregoiredavid.fr/repo/departements.geojson') as f : 
    geo_dep = json.load(f)


state_id_map = {}
for feature in geo_dep['features']:
 feature['id'] = feature['properties']['code']
 state_id_map[feature['properties']['nom']] = feature['id']


def nbpagespdf(filename):
    with open(filename, 'rb') as file:
        reader = pyPdf.PdfReader(file)
        return len(reader.pages)

def nompuits(fichierpdf):
    y = 1
    x = 0
    while x == 0 and (y <= nbpagespdf(fichierpdf)):
        pages = convert_from_path(fichierpdf,300, first_page=y,last_page=y)
        image_name = "Page.jpg"  
        pages[0].save(image_name, "JPEG")
        y = y+1
        text = str(pytesseract.image_to_string(image_name, config='--psm 6'))
        name = ''
        for i in range (len(text)):
            if text[i] == '/' and (text[i+2] == '-' or text[i+3] == '-') and ord(text[i+1])<=57 and ord(text[i+1])>=48:
                j=i-1
                while ord(text[j])<=57 and ord(text[j])>=48:
                    j -= 1
                    
                j += 1
                while (ord(text[j])<=57 and ord(text[j])>=48) or text[j] == '/' or text[j] == '-':
                    name = name + text[j]
                    j += 1
                    x = 1
        
    return name

#Fonction qui prend en paramètre un puit et un pattern et qui renvoie le nombre de fois qu'il trouve le pettern dans le puit
def match(wellFile, wellLegend):
    img_rgb = cv.imread(wellFile)
    assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread(wellLegend)
    assert template is not None, "file could not be read, check with os.path.exists()"

    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    scale_percent = 40 # percent of original size
    width = int(template_gray.shape[1] * scale_percent / 100)
    height = int(template_gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(template_gray, dim)
    w, h = resized.shape[::-1]


    res = cv.matchTemplate(img_gray,resized,cv.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    cpt = 0
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cpt += 1
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


# Afficher le bouton pour télécharger le fichier PDF
uploaded_file = st.file_uploader("Télécharger un fichier PDF", type="pdf")

if uploaded_file is not None:
    # Récupérer le contenu du fichier
    file_contents = uploaded_file.read()

    # Afficher un message de confirmation
    st.success("Le fichier PDF a été téléchargé avec succès.")
    # st.text(nbpagespdf(file_contents))
    st.text(nompuits("test2.pdf"))



 
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
            







