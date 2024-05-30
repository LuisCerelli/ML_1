# Machine Learning Operations (MLOps)

![MLOps](1704994327715.png)


## Contexto:
Hemos logrado desarrollar un modelo de recomendación que abarca todas las etapas fundamentales del proceso de Data Engineering, Análisis Exploratorio de Datos (EDA) y entrenamiento del modelo. Para ello, recibimos tres datasets en formato JSON, los cuales leímos y transformamo para poder alcanzar un MVP (Producto Mínimo Viable).

# Transformación de Datos

## Descripción

Se realizó un exhaustivo trabajo de transformación de datos en los datasets proporcionados en formato JSON, que presentaban inconsistencias y desorden. El objetivo principal fue minimizar las celdas nulas y convertir los datos al tipo más adecuado para su posterior uso en diversas funciones. Además, se implementaron las siguientes mejoras:

1. **Identificación de Datasets**: Se añadió un número al nombre de cada dataset para facilitar su identificación, sin embargo, en el dataset final se respetó su nombre original para no generar confusiones.
2. **Preservación Completa de Filas en Algunos Datasets**:
    - **steam_games** y **user_reviews** (datasets 1 y 2): Se conservaron todas las filas debido a las características particulares de estos datos.
3. **Transformación y Muestra de Datos en el Dataset 3**:
    - **users_items**: Este dataset presentaba una columna 'items' con diccionarios anidados y múltiples errores. Al abrir esta columna, el número de filas se incrementaba de aproximadamente 83,000 a cerca de 5,000,000. Para gestionar el tamaño y peso de los datos, se tomó una muestra aleatoria de 2,000 filas, facilitando así el despliegue en Render.

Este proceso asegura que los datos sean manejables y aptos para su uso en las siguientes etapas del proyecto, optimizando tanto la precisión como la eficiencia del trabajo con ellos.

[Transformacion de Datos: proyectos_reales_individuales](proyectos_reales_individuales)

# Procesamiento de Datos
A continuación, se describen las funciones que implementamos para interactuar con los datos procesados:

### Función developer

""
    Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.
    """
### Función userdata

"""
    Devuelve la cantidad de dinero gastado por el usuario, el porcentaje de recomendación 
    en base a reviews.recommend y la cantidad de items.
    """

### Función UserForGenre

"""
    Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de 
    la acumulación de horas jugadas por año de lanzamiento.
    """

 ## Análisis Exploratorio de Datos (EDA)
Realizamos un EDA para analizar las distintas variables entre los datasets, complementándolo con gráficos que permiten una mejor visualización de los datos, a continuación uno de los ejemplos, mas ejemplos en [Análisis Exploratorio de Datos: EDA.ipynb](EDA.ipynb)

![Nube de palabras](nube%20de%20palabras.png)

#### Análisis de la Gráfica
Frecuencia de Palabras: Las palabras más grandes en la nube representan las palabras más frecuentemente usadas en los títulos de los juegos. Por ejemplo, "Pack", "DLC", "Soundtrack", y "Edition" son algunas de las palabras más prominentes, lo que indica que estos términos aparecen con frecuencia en los títulos de los juegos del dataset.
Variedad de Términos: La nube de palabras también muestra una variedad de términos relacionados con los juegos, como "Simulator", "Train", "Game", "Add", "Steam", entre otros.
Esta gráfica es útil para obtener una visión rápida de las palabras más comunes en los títulos de los juegos, lo que puede ayudar a identificar tendencias y patrones en los nombres de los juegos en el dataset.

## Modelo de Recomendación User-Item
Implementamos un modelo de recomendación basado en la relación usuario-item. La función principal para realizar recomendaciones es:

   """
    Ingresando el id de un usuario, devuelve una lista con 5 juegos recomendados para dicho usuario.
    """

## Implementación y Despliegue
Toda la funcionalidad ha sido renderizada utilizando FastAPI y desplegada en RENDER. Puedes visualizar el despliegue final del trabajo en el siguiente enlace:

[Documentación del Deployment](https://ml-1-icy1.onrender.com/docs#/default/user_for_genre_userforgenre_get)

Contribuciones
Las contribuciones son bienvenidas. Por favor, abre un issue para discutir lo que te gustaría cambiar.
