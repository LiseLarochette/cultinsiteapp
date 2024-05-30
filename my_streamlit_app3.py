import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import requests
from io import BytesIO

# Configuration de la mise en page de Streamlit
st.set_page_config(page_title="CULT[IN]SITE", layout="wide")

# Charger les données au démarrage de l'application
@st.cache_data # Utilisation du cache pour accélérer le chargement de la page
def load_data():
    data_concatene = pd.read_csv("/workspaces/cultinsiteapp/data_concatene.csv")
    data_encoded = pd.read_csv("/workspaces/cultinsiteapp/data_encoded.csv")
    df_primaryName_actor = pd.read_csv('/workspaces/cultinsiteapp/df_primaryName_actor.csv')
    data_original = pd.read_csv("/workspaces/cultinsiteapp/df_french_films_comedy.csv")
    data_poster = pd.read_csv('/workspaces/cultinsiteapp/df_french_posters_final.csv')

    data = data_original.copy()
    data = data.drop(columns='Unnamed: 0', axis=1)
    data_concatene = data_concatene.drop(columns='Unnamed: 0', axis=1)
    data_encoded = data_encoded.drop(columns='Unnamed: 0', axis=1)
    df_primaryName_actor = df_primaryName_actor.drop(columns='Unnamed: 0', axis=1)
    
    return data_concatene, data_encoded, df_primaryName_actor, data_original, data_poster

data_concatene, data_encoded, df_primaryName_actor, data_original, data_poster = load_data()

# Définir les variables explicatives
X_title = data_concatene.iloc[:, 7:]  

# Entraîner le modèle pour la recommandation par titre
model_KNN_distance_title = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute")
model_KNN_distance_title.fit(X_title)


#---------------------------------------- FONCTION DE RECHERCHE -------------------------------------#

# Définir la fonction de recherche de films similaires :

@st.cache_data # Utilisation du cache pour accélérer le système de recherche
def recommandations_de_films(titre_de_film, k=4):
    film_choisi = data_concatene[data_concatene['originalTitle'].str.contains(titre_de_film, case=False)]
    
    if film_choisi.empty:
        return None
    
    film_choisi_data = film_choisi.iloc[:, 7:X_title.shape[1]+7]
    
    # Calculer les distances et indices des plus proches voisins à une distance de +1
    distances, indices = model_KNN_distance_title.kneighbors(film_choisi_data, n_neighbors=k+1)
    
    # Exclure le premier indice (qui est l'indice du film lui-même)
    indices = indices[0][1:]
    
    # Indices films
    id_films = data_concatene.iloc[indices]['tconst'].values
    
    recommendations = []
    
    # Recherche du poster du film, du titre, de l'année de sortie et la note moyenne du film recommandé
    for tconst in id_films:
        poster_path = data_poster[data_poster['tconst'] == tconst]['poster_path'].values
        poster_url = 'https://image.tmdb.org/t/p/w500' + poster_path[0]
        title = data_poster[data_poster['tconst'] == tconst]['originalTitle'].values[0]
        year = data_original[data_original['tconst'] == tconst]['startYear'].values[0]
        rating = data_original[data_original['tconst'] == tconst]['averageRating'].values[0]
        recommendations.append((poster_url, title, year, rating))
    
    return recommendations



#---------------------------------------- INTERFACE STREAMLIT -------------------------------------#


# Style de la page
st.markdown(
    """
    <style>

/* The animation code */


    .main {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: white;
        padding: 20px;
        max-width: 1200px;
        margin: 0 auto;
        min-height: 100vh; 
        position: relative; 
  }
        animation-name: example;
        animation-duration: 25s;
        animation-iteration-count: infinite; 
    }

.content {
    max-width: auto;
    width: 100%;
}
    .st-emotion-cache-dcpkew .e1f1d6gn4{
            animation: fadeIn 5s;
    }
    

    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    h1 {
        color : white!important;
        text-align : center!important;
        font-family : Poppins, sans-serif;
        font-weight :  900;
        font-size : 4.5em;
        animation: fadeIn 5s;
    }

    h2 {
        color : white!important;
        font-family : Poppins, sans-serif;
        text-align : center!important;      

        animation: fadeIn 5s;

    }

    h3 {
        font-family : Poppins, sans-serif;
        color : white!important;
        text-align : center!important;    
        animation: fadeIn 5s;


    }

    .st-emotion-cache-1jmvea6 p{
        color : white!important;
        text-align : center!important;     
    }

    .st-emotion-cache-183lzff{
        font-family : "Fira Sans", sans-serif!important;
        font-weight : 700;
        font
        color : white!important;
        text-align : center!important;          
    }

    .st-emotion-cache-7ym5gk{
        border-radius : 25px!important;

    }

    .st-emotion-cache-j6qv4b p{
        border-radius : 25px!important;
        color : #30334A!important;
        text-align : center!important;   
    }

    .st-emotion-cache-1f80lg1{
        text-align : center!important;   
    }

    .st-emotion-cache-1r4qj8v{
        background-color: #091734;
        text-align : center!important;   
    
    }

    st-emotion-cache-7ym5gk{
        text-align : center!important;   
        color : #30334A!important;
    }

    .st-emotion-cache-183lzff{
        font-family : "Fira Sans", sans-serif!important;
        color : white!important;
        text-align : center!important;  
    }



    </style>
    """,
    unsafe_allow_html=True
)

# Titre et sous-titres
st.title("CULT[IN]MOVIES")
st.header("Trouvez le prochain film à diffuser dans votre cinéma")
st.subheader("Recherchez une recommandation par titre de film")

# Ajouter l'année de sortie à côté des titres des films
data_concatene['title_with_year'] = data_concatene['originalTitle'] + ' (' + data_concatene['startYear'].astype(str) + ')'

# Menu déroulant pour la recherche de film
film_title_with_year = st.selectbox('Veuillez entrer ou sélectionner un film de votre choix', data_concatene['title_with_year'].unique())

# Extraire le titre du film sans l'année
film_title = film_title_with_year.rsplit(' (', 1)[0]

# Bouton de validation de la recherche
if st.button("Rechercher des recommandations"):
    recommendations = recommandations_de_films(film_title)
    
    if recommendations is None:
        st.error(f"Le film '{film_title}' n'a pas été trouvé dans les titres de films.")
    else:
        cols = st.columns(4)
        for idx, (poster_url, title, year, rating) in enumerate(recommendations):
            with cols[idx]:
                response = requests.get(poster_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, use_column_width=True)
                st.text(f"{title} ({year})")
                st.text(f"Note moyenne : {rating}")

