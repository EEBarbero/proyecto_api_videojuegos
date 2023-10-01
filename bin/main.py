###################################################################
## Proyecto 1 del LAB de Henry  ##
## Desarrollo de la API de un "MVP"para Steam, una plataforma multinacional de videojuegos.
##
## Autor: Edgar Eduardo Barbero
## Carrera: Data Science
## Cohorte: 15
## Período de desarrollo: 26 de septiembre-03 de octubre de 2023
##
## Última actualización: 30/09/2023
###################################################################

# Importación de librerías 
from fastapi import FastAPI
from typing import Union
import numpy as np
import pandas as pd
import warnings

# Desactivación de warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

# Instancio FastApi
app = FastAPI()

# Cargo los archivos de datos
df_reviews = pd.DataFrame(pd.read_csv(r"../data/aur_api.csv"))
df_users = pd.DataFrame(pd.read_csv(r"../data/aui_api.csv"))
df_items = pd.DataFrame(pd.read_csv(r"../data/osg_api.csv"))

# Hago los merge para que trabajen las funciones
df_UsersItems = (pd.merge(df_users, df_items, left_on="item_id",right_on="item_id")[["user_id","genres","anio_lanzamiento","playtime_forever"]])
df_ReviewsItems = (df_reviews.merge(df_items, left_on="item_id", right_on= "item_id")[["anio_post", "recommend", "sentiment_analysis", "item_id", "title", "anio_lanzamiento"]])

# Creo la función de bienvenida para el root
@app.get("/")
def read_root():
    return {"Hola":"Bienvenido a la API de Edgar Barbero"}

# Creo la Función del PlayTimeGenre
# Retorna el "año" de lanzamiento de los jurgos con mas horas jugadas para un género solicitado
@app.get("/play_time_genre/{genero}")
def PlayTimeGenre(genero: str):
    # Creo el filtro
    filtro = df_UsersItems["genres"].apply(lambda x: True if genero.strip().lower() in list(str(x).split(",")) else False)
    # Obtengo los resultados, los agrupo y ordeno
    resultado = (df_UsersItems[filtro].groupby(["anio_lanzamiento"]).sum()[["playtime_forever"]].
                 sort_values(by="playtime_forever", ascending=False).reset_index())
    # Verifico si la consulta está vacía y creo el retorno para ese caso
    if resultado.empty:
        retorno = {"Error":f"No se dispone de datos para el Género {genero.title()}. Por favor compuebe si no hay errores de tipeo."}
    else:
        # Creo el retorno para la consulta
        retorno = {f"Año de lanzamiento con más horas jugadas para Género {genero.title()}": str(int(resultado.iloc[0]["anio_lanzamiento"])), "Cantidad de horas jugadas": format(int(resultado.iloc[0]["playtime_forever"]), ',d').replace(",",".")}
    return retorno

# Creo la función UserForGenre
# Retorna el usuario que acumula más horas jugadas para el género dado y una lista de la composición  
# de horas jugadas agrupadas por año de lanzamiento de los juegos
@app.get("/user_for_genre/{genero}")
def UserForGenre(genero: str):
    # Creo el filtro
    filtro = df_UsersItems["genres"].apply(lambda x: True if genero.strip().lower() in list(str(x).split(",")) else False)
    # Obtengo los resultados
    resultado = df_UsersItems[filtro]
    # Verifico si la consulta está vacía y creo el retorno para ese caso    
    if resultado.empty:
        retorno = {"Error":f"No se dispone de datos para el Género {genero.title()}. Por favor compuebe si no hay errores de tipeo."}
    else:
        # Guardo en una variable el usuario con más horas jugadas
        usuario = df_UsersItems.groupby(["user_id"]).sum()[["playtime_forever"]].reset_index().sort_values(by="playtime_forever", ascending=False).iloc[0]["user_id"]
        # Creo el filtro compuesto para genero y usuario
        filtro = (df_UsersItems["genres"].apply(lambda x: True if genero.strip().lower() in list(str(x).split(",")) else False)) & (df_UsersItems["user_id"].str == str(usuario))
        # Obtengo el resultado y lo agrupo por año
        resultado2 = df_UsersItems[filtro].groupby(["anio_lanzamiento"]).sum()[["playtime_forever"]].reset_index().sort_values(by="playtime_forever", ascending=False).reset_index()
        # Cargo en un vector el retorno de cada uno de los registros de año
        vector = [{"Año": str(int(resultado.iloc[i]["anio_lanzamiento"])),"Horas":format(int(resultado.iloc[i]["playtime_forever"]), ',d').replace(",",".")} for i in range(0,len(resultado2.index)+1)]
        # Creo el retorno para la consulta
        retorno = {f"Usuario con más horas jugadas para el Género {genero.title()}":usuario,"Horas jugadas por año de lanzamiento de los juegos":vector}
    return retorno

# Creo la función UsersRecommend
# Retorna los 3 de juegos MÁS recomendados por usuarios para el año dado con comentarios 
# positivos o neutrales
@app.get("/users_recommend/{anio}")
def UsersRecommend(anio: int):
    # Creo el filtro con el año pasado, recomendación en "True" y análisis de sentimientos positivos 
    # o neutrales
    filtro = ((df_ReviewsItems["anio_post"] == anio) & (df_ReviewsItems["recommend"] == True) & ((df_ReviewsItems["sentiment_analysis"] == 1) | (df_ReviewsItems["sentiment_analysis"] == 2)))
    #Obtengo la colsulta y lo agrupo por año de post
    resultado = df_ReviewsItems[filtro].groupby(["item_id", "title"]).count()[["recommend"]].reset_index().sort_values("recommend", ascending=False).reset_index().drop("index", axis=1)
    if resultado.empty:
        retorno = {"Error":f"No se dispone de datos para el Año {anio}."}
    else:
        # Preparo la salida a retornar
        retorno = {f"Los 3 juegos MÁS recomendados para el año {anio}": [{f"Puesto {i+1}": {"Identificador del Juego": str(resultado["item_id"][i]),"Título":resultado["title"][i]}} for i in range(min(3,len(resultado.index)))]}
    return retorno

# Creo la función UsersNotRecommend
# Retorna los 3 de juegos MENOS recomendados por usuarios para el año dado con comentarios 
# negativos
@app.get("/users_not_recommend/{anio}")
def UsersNotRecommend(anio: int):
    # Creo el filtro con el año pasado, recomendación en "False" y análisis de sentimientos negativos
    filtro = ((df_ReviewsItems["anio_post"] == anio) & (df_ReviewsItems["recommend"] == False) & (df_ReviewsItems["sentiment_analysis"] == 0))
    #Obtengo la colsulta y lo agrupo por año de post
    resultado = df_ReviewsItems[filtro].groupby(["item_id", "title"]).count()[["recommend"]].reset_index().sort_values("recommend", ascending=True).reset_index().drop("index", axis=1)
    if resultado.empty:
        retorno = {"Error":f"No se dispone de datos para el Año {anio}."}
    else:
        # Preparo la salida a retornar
        retorno = {f"Los 3 juegos MENOS recomendados para el año {anio}": [{f"Puesto {i+1}": {"Identificador del Juego": str(resultado["item_id"][i]),"Título":resultado["title"][i]}} for i in range(min(3,len(resultado.index)))]}
    return retorno

# Creo la función UsersNotRecommend
# Retorna las clasificación de comentarios en negativos, neutrales y positivos según año de 
# lanzamiento de los juegos
@app.get("/sentiment_analysis/{anio}")
def sentiment_analysis(anio: int):
    # Creo el filtro con el año pasado, recomendación en "False" y análisis de sentimientos negativos
    filtro = (df_ReviewsItems["anio_lanzamiento"] == anio)
    #Obtengo la colsulta y lo agrupo por año de post
    resultado = df_ReviewsItems[filtro].groupby(["sentiment_analysis"]).count()[["item_id"]].reset_index().set_index("sentiment_analysis")
    if resultado.empty:
        retorno = {"Error":f"No se dispone de datos para el Año {anio}."}
    else:
        # Preparo la salida a retornar {Negative = 182, Neutral = 120, Positive = 278}
        negativos = resultado.loc[0]["item_id"] if 0 in resultado.index else 0
        neutros = resultado.loc[1]["item_id"] if 1 in resultado.index else 0
        positivos = resultado.loc[2]["item_id"] if 2 in resultado.index else 0
        retorno = {f"Clasificación de comentarios para los juegos lanzados en el año {anio}":f"Negativos: {negativos}, Neutrales: {neutros}, Positivos: {positivos}"}        
    return retorno