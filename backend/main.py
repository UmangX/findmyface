from fastapi import FastAPI
from fastapi.responses import FileResponse
import pandas as pd
import sqlite3
import os 

app = FastAPI()
DB = './static/faces.db'
SELECTALL = "SELECT * FROM files;"

@app.get("/facesdb")
def getfacesdb():
    with sqlite3.connect(DB) as conn:
        facesdf = pd.read_sql_query(SELECTALL, conn)
        return facesdf.drop(columns="embeddings",errors='ignore').to_dict(orient='records')

@app.get("/facesdb/{item_id}")
def imageData(item_id: int):
    with sqlite3.connect(DB) as conn:
        query = "SELECT * FROM files WHERE id = ?;"
        facesdf = pd.read_sql_query(query, conn, params=[item_id])
    return facesdf.drop(columns="embeddings",errors='ignore').to_dict(orient="records")
        
@app.get("/facesdb/{item_id}/image")
def imageFile(item_id: int):
    with sqlite3.connect(DB) as conn:
        query = "select * from files where id = ?;"
        facesdf = pd.read_sql_query(query, conn, params=[item_id])
        file_path = facesdf.iloc[0]['name']
        if not os.path.exists(file_path):
            return {"error": "file not found"}
        return FileResponse(path=file_path)

