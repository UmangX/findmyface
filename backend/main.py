from fastapi import FastAPI
from fastapi.responses import FileResponse
import pandas as pd
import sqlite3
import os 

app = FastAPI()
DB = './static/faces.db'

@app.get("/facesdb")
def getfacesdb():
    with sqlite3.connect(DB) as conn:
        selectQuery = "SELECT * FROM files;"
        facesdf = pd.read_sql_query(selectQuery, conn)
        return facesdf.drop(columns="embeddings",errors='ignore').to_dict(orient='records')

@app.get("/facesdb/{item_id}")
def get_faces_db(item_id: int):
    with sqlite3.connect(DB) as conn:
        query = "SELECT * FROM files WHERE id = ?;"
        facesdf = pd.read_sql_query(query, conn, params=[item_id])
    return facesdf.drop(columns="embeddings",errors='ignore').to_dict(orient="records")
        
@app.get("/facesdb/{item_id}/image")
def get_faces_db(item_id: int):
    with sqlite3.connect(DB) as conn:
        query = "SELECT * FROM files WHERE id = ?;"
        facesdf = pd.read_sql_query(query, conn, params=[item_id])
        file_path = facesdf.iloc[0]['name']
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        return FileResponse(path=file_path)
 
