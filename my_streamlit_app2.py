# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Charger les données

data_concatene_original = pd.read_csv("data_concatene.csv")
data_concatene = data_concatene_original.drop(columns="Unnamed: 0")

df_primaryName_actor = pd.read_csv('/workspaces/cultinsiteapp/df_primaryName_actor.csv')
df_primaryName_actor = df_primaryName_actor.drop(columns='Unnamed: 0', axis=1)

data_original = pd.read_csv("/workspaces/cultinsiteapp/df_french_films_comedy.csv")

data = data_original.copy()

data_encoded = pd.read_csv("/workspaces/cultinsiteapp/data_encoded.csv")
data_encoded = data_encoded.drop(columns='Unnamed: 0', axis=1)


# ------------------------------------------| MACHINE LEARNING |----------------------------------------

# Définir les variables explicatives
X = data_concatene.iloc[:, 7:]  # les 7 premières colonnes sont les variables numériques

# Entrainer le modèle
model_KNN_distance = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute")
model_KNN_distance.fit(X)

# Fonction pour trouver des films similaires
def quid_film_similaire(titre_de_film, k=4):
    film_choisi = data_concatene[data_concatene['originalTitle'].str.contains(titre_de_film, case=False)]
    
    if film_choisi.empty:
        return f"Le film '{titre_de_film}' n'a pas été trouvé dans les titres de films."
    
    film_choisi_data = film_choisi.iloc[:, 7:]
    
    # Calculer les distances et indices des plus proches voisins a une distance de +1
    distances, indices = model_KNN_distance.kneighbors(film_choisi_data, n_neighbors=k+1)
    
    # Exclure le premier indice (qui est l'indice du film lui-même)
    indices = indices[0][1:]
    
    # Retourner les titres des films similaires
    films_similaires = data_concatene.iloc[indices]['originalTitle'].values
    
    return films_similaires

nom_de_film = input("Veuillez entrer un nom de film")
recomandation_de_film = quid_film_similaire(nom_de_film)

if type(recomandation_de_film) == str:
    print(recomandation_de_film)
    
else:
    affichage_recommandations = list(recomandation_de_film)
    i = 1
    while i < len(recomandation_de_film)+2:
        affichage_recommandations.insert(i, '\n')
        i += 2 

    print(f"Voici les {len(recomandation_de_film)} films qui sont recommandés pour {nom_de_film} :\n ")
    print(''.join(affichage_recommandations))



# ------------------------------------------| INTERFACE |----------------------------------------

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
