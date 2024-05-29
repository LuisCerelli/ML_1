import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Query, Path
import json
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


app = FastAPI(title='Luis Cerelli - DTFT22')


# Cargar los DataFrames desde archivos Parquet
df_1_steam_games = pd.read_parquet('Datasets_para_Render/steam_games.parquet')
df_user_reviews = pd.read_parquet('Datasets_para_Render/user_reviews.parquet')
df_user_items = pd.read_parquet('Datasets_para_Render/users_items.parquet')

# Convertir 'release_date' a datetime y extraer el año
df_1_steam_games['release_date'] = pd.to_datetime(df_1_steam_games['release_date'], format='%Y-%m-%d', errors='coerce')
df_1_steam_games['Año'] = df_1_steam_games['release_date'].dt.year


@app.get("/developer", response_class=HTMLResponse)
def developer(desarrollador : str = Query(default='Stainless Games Ltd')):
    # Filtrar los juegos por el desarrollador
    dev_games = df_1_steam_games[df_1_steam_games['developer'] == desarrollador]

    # Calcular la cantidad de ítems por año
    items_per_year = dev_games.groupby('Año')['item_id'].count().reset_index(name='Cantidad de Items')
    
    # Calcular el porcentaje de contenido gratuito por año
    free_content_per_year = dev_games[dev_games['price'] == 0].groupby('Año')['item_id'].count().reset_index(name='Free Items')
    total_items_per_year = dev_games.groupby('Año')['item_id'].count().reset_index(name='Cantidad Items')
    
    # Unir los DataFrames
    merged = pd.merge(total_items_per_year, free_content_per_year, on='Año', how='left').fillna(0)
    merged['Contenido Free'] = (merged['Free Items'] / merged['Cantidad Items']) * 100

    # Preparar el resultado
    result = merged[['Año', 'Cantidad Items', 'Contenido Free']].to_string(index=False)
    
    return "<pre>" + result + "</pre>"

@app.get("/userdata")
def user_stats_endpoint(user_id : str= Query(default='paliusz')):
    #print(f"User ID: {user_id}")  # Imprimir el user_id

    """
    Endpoint que devuelve las estadísticas de un usuario específico.
    
    Args:
        user_id (int): ID del usuario.
        
    Returns:
        dict: Diccionario con las estadísticas del usuario.
    """
       
    # Filtrar los ítems del usuario
    user_items = df_user_items[df_user_items['user_id'] == user_id]
    #print(f"User items: {user_items}")  # Imprimir los ítems del usuario
    
    # Calcular el dinero gastado
    item_ids = user_items['item_id'].tolist()
    #print(f"Item IDs: {item_ids}")  # Imprimir los IDs de los ítems

    # Asumiendo que item_ids se define en la línea anterior
    #item_ids = ...

    # Verificar si los item_ids en user_items existen en df_1_steam_games
    item_ids_in_df = df_1_steam_games['item_id'].isin(item_ids)

    if item_ids_in_df.any():
        print("Algunos item_ids en user_items existen en df_1_steam_games")
    else:
        print("Ningún item_id en user_items existe en df_1_steam_games")


    # Convertir los item_ids a flotantes
    df_1_steam_games['item_id'] = df_1_steam_games['item_id'].astype(float)
    item_ids = [float(item_id) for item_id in item_ids]

    # Tu línea original
    game_prices = pd.to_numeric(df_1_steam_games[df_1_steam_games['item_id'].isin(item_ids)]['price'])


    # Tu línea original
    game_prices = pd.to_numeric(df_1_steam_games[df_1_steam_games['item_id'].isin(item_ids)]['price'])

    #game_prices = df_1_steam_games[df_1_steam_games['item_id'].isin(item_ids)]['price']
    game_prices = pd.to_numeric(df_1_steam_games[df_1_steam_games['item_id'].isin(item_ids)]['price'])

    print(f"Game prices: {game_prices}")  # Imprimir los precios de los juegos

    # Imprimir los precios de los juegos
    #print(game_prices)

    money_spent = game_prices.sum()
    
    # Calcular el porcentaje de recomendación
    user_reviews = df_user_reviews[df_user_reviews['user_id'] == user_id]
    recommended = user_reviews[user_reviews['sentiment_analysis'] == 2].shape[0]
    not_recommended = user_reviews[user_reviews['sentiment_analysis'] == 0].shape[0]
    total_reviews = user_reviews.shape[0]
    if total_reviews > 0:
        recommendation_percentage = (recommended / total_reviews) * 100
    else:
        recommendation_percentage = 0

    # Calcular la cantidad de horas jugadas:
    total_playtime = user_items['playtime_forever'].sum()

    
    # Calcular la cantidad de ítems jugados
    num_items_played = user_items.shape[0]
    
    return {
        'Usuario': user_id,
        'Dinero gastado': money_spent,
        'Porcentaje de recomendación': recommendation_percentage,
        'cantidad de items': num_items_played,
        'cantidad de horas jugadas': total_playtime
    }

#*********************************************************************************************************************

def str_to_list(s):
    # Comprobar si 's' es una cadena de texto
    if isinstance(s, str):
        # Remover los corchetes y las comillas
        s = s.replace("[", "").replace("]", "").replace("'", "")
        # Dividir la cadena por las comas y eliminar los espacios en blanco
        return [x.strip() for x in s.split(",")]
    # Si 's' no es una cadena de texto, devolver 's' tal cual
    return s

@app.get("/userforgenre")
def user_for_genre(genero: str = Query(default='Indie')):
    # Convertir las cadenas de texto a listas
    df_1_steam_games['genres'] = df_1_steam_games['genres'].apply(str_to_list)
    
    # Crear un nuevo dataframe que tenga una fila por cada combinación de juego y género
    df_genre_expanded = df_1_steam_games.explode('genres')
  
    # Filtrar juegos por género
    genre_games = df_genre_expanded[df_genre_expanded['genres'] == genero].copy()
    
    # Reemplazar None con 0 en 'item_id'
    df_user_items['item_id'] = df_user_items['item_id'].fillna(0)
    genre_games['item_id'] = genre_games['item_id'].fillna(0)

    # Asegúrate de que 'item_id' en ambos dataframes sean del mismo tipo de datos
    df_user_items['item_id'] = df_user_items['item_id'].apply(lambda x: int(float(x)))
    genre_games['item_id'] = genre_games['item_id'].apply(lambda x: int(float(x)))
   
    # Luego, intenta ejecutar tu código original de nuevo, pero usando 'item_id' en lugar de 'items_count'
    user_playtime = df_user_items[df_user_items['item_id'].isin(genre_games['item_id'])].groupby('user_id')['playtime_forever'].sum().reset_index()
    print(f'el df user_playtime ha quedado asi:\n {user_playtime}')

    # Comprobar si user_playtime está vacío
    if user_playtime.empty:
        return {"error": "No se encontraron usuarios para el género proporcionado"}

    # Encontrar el usuario con más horas jugadas
    top_user = user_playtime.sort_values(by='playtime_forever', ascending=False).iloc[0]

    # Convertir 'item_id' a float y luego a int en ambos dataframes
    df_user_items['item_id'] = df_user_items['item_id'].apply(lambda x: int(float(x)))
    df_1_steam_games['item_id'] = df_1_steam_games['item_id'].apply(lambda x: int(float(x)))

    # Filtrar los juegos del usuario con más horas jugadas
    top_user_games = df_user_items[(df_user_items['user_id'] == top_user['user_id']) & (df_user_items['item_id'].isin(genre_games['item_id']))]
    playtime_per_year = top_user_games.merge(df_1_steam_games[['item_id', 'Año']], on='item_id').groupby('Año')['playtime_forever'].sum().reset_index()

    playtime_list = playtime_per_year.to_dict(orient='records')

    return {
        "user_id": top_user['user_id'],
        "total_playtime": top_user['playtime_forever'],
        "playtime_per_year": playtime_list
    }
    


#*********************************************************************************************************************
#SISTEMAS de RECOMENDACION:

# Sistema de Recomendacion USUARIO-ITEM


# Expandir la columna 'items' para obtener un DataFrame donde cada fila es un juego jugado por un usuario
user_game_list = []

df_user_items_copy = df_user_items.copy()
df_user_items_copy.columns = df_user_items_copy.columns.astype(str)

for index, row in df_user_items_copy.iterrows():
    user_id = row['user_id']
    item_id = row['item_id']
    user_game_list.append({'user_id': user_id, 'appid': item_id})

# Convertir a DataFrame
user_game_df = pd.DataFrame(user_game_list)

# Eliminar duplicados
user_game_df = user_game_df.drop_duplicates(subset=['user_id', 'appid'])

# Crear matriz de usuario-juego
user_game_matrix = user_game_df.pivot(index='user_id', columns='appid', values='appid').notnull().astype(int)

# Convertir los nombres de las columnas a cadenas
user_game_matrix.columns = user_game_matrix.columns.astype(str)

# Modelo de vecinos más cercanos
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_game_matrix)




@app.get("/recomendacion_usuario/")
async def recomendacion_usuario(user_id: str = Query('DeEggMeister')):
    user_index = user_game_matrix.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(user_game_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=6)
    similar_users = [user_game_matrix.index[i] for i in indices.flatten()][1:]
    
    # Obtener recomendaciones basadas en juegos jugados por usuarios similares
    similar_users_games = user_game_df[user_game_df['user_id'].isin(similar_users)]
    recommendations = similar_users_games['appid'].value_counts().head(5).index.tolist()
    
    # Mapear 'appid' a nombres de juegos
    game_names = df_user_items[df_user_items['item_id'].isin(recommendations)]['item_name'].tolist()
    
    return {"Juegos similares que te pueden interesar a ti " + user_id: game_names}





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


