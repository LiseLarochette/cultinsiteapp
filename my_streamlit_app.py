import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Charger les données
data_concatene = pd.read_csv("/workspaces/cultinsiteapp/data_concatene.csv")
data_encoded = pd.read_csv("/workspaces/cultinsiteapp/data_encoded.csv")
df_primaryName_actor = pd.read_csv('/workspaces/cultinsiteapp/df_primaryName_actor.csv')
data_original = pd.read_csv("/workspaces/cultinsiteapp/df_french_films_comedy.csv")
data = data_original.copy()
data = data.drop(columns='Unnamed: 0', axis=1)
data_concatene = data_concatene.drop(columns='Unnamed: 0', axis=1)
data_encoded = data_encoded.drop(columns='Unnamed: 0', axis=1)
df_primaryName_actor = df_primaryName_actor.drop(columns='Unnamed: 0', axis=1)


# Définir les variables explicatives
X_title = data_concatene.iloc[:, 7:]  # les 7 premières colonnes sont les variables numériques

# Entraîner le modèle pour la recommandation par titre
model_KNN_distance_title = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute")
model_KNN_distance_title.fit(X_title)

# Fonction pour trouver des films similaires par titre
def quid_film_similaire(titre_de_film, k=4):
    film_choisi = data_concatene[data_concatene['originalTitle'].str.contains(titre_de_film, case=False)]
    
    if film_choisi.empty:
        return f"Le film '{titre_de_film}' n'a pas été trouvé dans les titres de films."
    
    film_choisi_data = film_choisi.iloc[:, 7:]
    
    # Calculer les distances et indices des plus proches voisins à une distance de +1
    distances, indices = model_KNN_distance_title.kneighbors(film_choisi_data, n_neighbors=k+1)
    
    # Exclure le premier indice (qui est l'indice du film lui-même)
    indices = indices[0][1:]
    
    # Retourner les titres des films similaires
    films_similaires = data_concatene.iloc[indices]['originalTitle'].values
    
    return films_similaires


# Entrainement pour recommandations par acteurs
X_actor = data_encoded.iloc[:, 7:]

model_KNN_distance_actor = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute")
model_KNN_distance_actor.fit(X_actor)

# Fonction pour recommander des films par acteur
def recommander_films_par_acteur(nom_acteur, k=4):
    colonnes_acteur = df_primaryName_actor.columns[df_primaryName_actor.columns.str.contains(nom_acteur, case=False)]
    
    if colonnes_acteur.empty:
        return f"L'acteur '{nom_acteur}' n'a pas été trouvé dans les données."
    
    acteur_choisi_data = pd.DataFrame(0, index=[0], columns=X_actor.columns)
    acteur_choisi_data.loc[:, colonnes_acteur] = 1
    
    distances, indices = model_KNN_distance_actor.kneighbors(acteur_choisi_data, n_neighbors=k+1)
    indices = indices[0][1:]
    
    films_recommandes = data_original.iloc[indices]['originalTitle'].values
    
    return films_recommandes



# CSS pour la personnalisation
st.markdown("""
    <style>
        body {
            background-color: #ffffff;
            color: #141414;
            font-family: Arial, sans-serif;
        }
        .main {
            background-color: #ffffff;
        }
        h1 {
            color: #000000;
            text-align: center;
        }
        h2, h3, h4 {
            color: #e50914;
            text-align: center;
        }
        .stButton>button {
            background-color: #e50914;
            color: #ffffff;
            border-radius: 5px;
        }
        .stTextInput>div>input {
            background-color: #f0f0f0;
            color: #141414;
            border-radius: 5px;
        }
        .stSelectbox>div {
            background-color: #f0f0f0;
            color: #141414;
            border-radius: 5px;
        }
        .stSlider > div {
            color: #141414;
        }
        .css-1d391kg, .css-18e3th9 {
            color: #141414;
        }
    </style>
""", unsafe_allow_html=True)

# Préparer l'interface utilisateur
st.title("Cult [In] Site")
st.header("Trouvez des films similaires en fonction du titre ou d'un acteur")

# Rechercher un film par titre
st.subheader("Recherche par titre de film français")
film_titre = st.text_input("Recherchez un film par titre")

# Bouton pour la recherche de similarité par film
if st.button("Trouver les films similaires"):
    if film_titre:
        recommandations_titre = quid_film_similaire(film_titre)
        if isinstance(recommandations_titre, str):
            st.write(recommandations_titre)
        else:
            st.subheader("Films similaires:")
            cols = st.columns(4)  # Create 4 columns
            for i, film in enumerate(recommandations_titre):
                with cols[i]:
                    # Poster du film a insérer ici
                    st.write(film)
    else:
        st.write("Veuillez entrer un titre de film.")

# Sélectionner un acteur préféré
st.subheader("Recherche par acteur")
acteur_prefere = st.text_input("Entrez le nom de l'acteur")

# Bouton de recherche de films avec l'acteur sélectionné
if st.button("Trouver les films avec cet acteur"):
    if acteur_prefere:
        recommandations_acteur = recommander_films_par_acteur(acteur_prefere)
        if isinstance(recommandations_acteur, str):
            st.write(recommandations_acteur)
        else:
            st.subheader(f"Voici {len(recommandations_acteur)} films recommandés avec l'acteur '{acteur_prefere}':")
            cols = st.columns(4)  # Create 4 columns
            for i, film in enumerate(recommandations_acteur):
                with cols[i]:
                    st.write(film)
    else:
        st.write("Veuillez entrer le nom d'un acteur.")

# Ajouter des espaces pour une meilleure disposition
st.markdown("<br><br>", unsafe_allow_html=True)
