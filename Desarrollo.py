from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors




app = FastAPI()
# Cargar los DataFrames desde archivos Parquet
df_1_steam_games = pd.read_parquet('/Users/luisalbertocerelli/Desktop/00-Todo/Data Science/01-Full Time/TI_1/Datasets_para_Render/steam_games.parquet')
df_2_user_reviews = pd.read_parquet('/Users/luisalbertocerelli/Desktop/00-Todo/Data Science/01-Full Time/TI_1/Datasets_para_Render/user_reviews.parquet')
df_3_users_items = pd.read_parquet('/Users/luisalbertocerelli/Desktop/00-Todo/Data Science/01-Full Time/TI_1/Datasets_para_Render/users_items.parquet')

@app.get("/developer/{desarrollador}")
def developer(desarrollador: str):
    result = df_1_steam_games[df_1_steam_games['developer'] == desarrollador].groupby('release_year').agg({
        'appid': 'count',
        'price': lambda x: (x == 0).mean() * 100
    }).rename(columns={'appid': 'Cantidad de Items', 'price': 'Contenido Free'}).reset_index()
    return result.to_dict(orient='records')


@app.get("/userdata/{User_id}")
def userdata(User_id: str):
    user_data = df_3_users_items[df_3_users_items['user_id'] == User_id]
    total_spent = user_data['price'].sum()
    recommendation_percentage = user_data['recommend'].mean() * 100
    item_count = user_data['appid'].nunique()
    return {
        "Usuario": User_id,
        "Dinero gastado": total_spent,
        "% de recomendación": recommendation_percentage,
        "Cantidad de items": item_count
    }


@app.get("/UserForGenre/{genero}")
def UserForGenre(genero: str):
    genre_data = df_3_users_items[df_3_users_items['genres'].str.contains(genero, na=False)]
    top_user = genre_data.groupby('user_id')['playtime_forever'].sum().idxmax()
    hours_per_year = genre_data[genre_data['user_id'] == top_user].groupby('release_year')['playtime_forever'].sum().reset_index()
    return {
        "Usuario con más horas jugadas para Género": top_user,
        "Horas jugadas": hours_per_year.to_dict(orient='records')
    }


@app.get("/best_developer_year/{año}")
def best_developer_year(año: int):
    year_data = df_2_user_reviews[df_2_user_reviews['release_year'] == año]
    top_developers = year_data[year_data['sentiment_analysis'] == 2].groupby('developer')['review_id'].count().nlargest(3).reset_index()
    top_developers.columns = ['Desarrollador', 'Recomendaciones Positivas']
    return top_developers.to_dict(orient='records')



@app.get("/developer_reviews_analysis/{desarrolladora}")
def developer_reviews_analysis(desarrolladora: str):
    dev_reviews = df_2_user_reviews[df_2_user_reviews['developer'] == desarrolladora]
    positive_count = dev_reviews[dev_reviews['sentiment_analysis'] == 2].shape[0]
    negative_count = dev_reviews[dev_reviews['sentiment_analysis'] == 0].shape[0]
    return {
        desarrolladora: {
            "Negative": negative_count,
            "Positive": positive_count
        }
    }
#SISTEMAS de RECOMENDACION:
# Sistema de Recomendacion ITEM-ITEM


# Concatenar tags en una cadena de texto
df_1_steam_games['tags_combined'] = df_1_steam_games['tags'].apply(lambda x: ' '.join(x)).str.lower().fillna('')

# Vectorización
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_1_steam_games['tags_combined'])

# Cálculo de la similitud del coseno
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.get("/recomendacion_juego/{product_id}")
def recomendacion_juego(product_id: int):
    idx = df_1_steam_games[df_1_steam_games['id'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_5_similar = [df_1_steam_games.iloc[i[0]]['id'] for i in sim_scores[1:6]]
    return top_5_similar


# ESTE FUNCIONA.. ESTE FUNCIONA.. ESTE FUNCIONA... SIIIIIII

# Sistema de Recomendacion USUARIO-ITEM


# Cargar el DataFrame desde un archivo Parquet
# users_items = pd.read_parquet('path_to_your_file.parquet')

# Expandir la columna 'items' para obtener un DataFrame donde cada fila es un juego jugado por un usuario
user_game_list = []

for index, row in df_3_users_items.iterrows():
    user_id = row['user_id']
    items = row['items']
    for item in items:
        item_id = item['item_id']
        user_game_list.append({'user_id': user_id, 'appid': item_id})

# Convertir a DataFrame
user_game_df = pd.DataFrame(user_game_list)

# Eliminar duplicados
user_game_df = user_game_df.drop_duplicates(subset=['user_id', 'appid'])

# Crear matriz de usuario-juego
user_game_matrix = user_game_df.pivot(index='user_id', columns='appid', values='appid').notnull().astype(int)

# Modelo de vecinos más cercanos
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_game_matrix)


@app.get("/recomendacion_usuario/{user_id}")
def recomendacion_usuario(user_id: str):
    user_index = user_game_matrix.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(user_game_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=6)
    similar_users = [user_game_matrix.index[i] for i in indices.flatten()][1:]
    
    # Obtener recomendaciones basadas en juegos jugados por usuarios similares
    similar_users_games = user_game_df[user_game_df['user_id'].isin(similar_users)]
    recommendations = similar_users_games['appid'].value_counts().head(5).index.tolist()
    return recommendations
