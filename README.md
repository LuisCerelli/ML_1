# Machine Learning Operations (MLOps)

![Nube de Palabras](1704994327715.png)


## Contexto:
Hemos logrado desarrollar un modelo de recomendación que abarca todas las etapas fundamentales del proceso de Data Engineering, Análisis Exploratorio de Datos (EDA) y entrenamiento del modelo. Para ello, recibimos tres datasets en formato JSON, los cuales leímos y transformamo para poder alcanzar un MVP (Producto Mínimo Viable).

Transformación y Procesamiento de Datos
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
Realizamos un EDA exhaustivo para analizar las distintas variables entre los datasets, complementándolo con gráficos que permiten una mejor visualización de los datos.

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
