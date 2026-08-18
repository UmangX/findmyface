from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
import base64
import logging
import json
import numpy as np
import pandas as pd
import sqlite3
import os

logger = logging.getLogger(__name__);
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
        

@app.get("/facesdb/{item_id}/similar/images")
def getsimilarimages(item_id: int, limit: int = Query(default=10, ge=1, le=100)):
    with sqlite3.connect(DB) as conn:
        query = "select * from files where id = ?;"
        targetdf = pd.read_sql_query(query, conn, params=[item_id])
        if targetdf.empty:
            return {"error": "item not found"}
        targetdata = targetdf.iloc[0]
        if not os.path.exists(targetdata['name']):
            return {"error": "file not found"}
        targetemb = np.frombuffer(targetdata['embeddings'], dtype=np.float32)

    with sqlite3.connect(DB) as conn:
        facesdf = pd.read_sql_query(SELECTALL, conn)

    results = []
    for row in facesdf.itertuples(index=False):
        if row.id == item_id:
            continue
        if not os.path.exists(row.name):
            continue
        emb = np.frombuffer(row.embeddings, dtype=np.float32)
        similarity = float(np.dot(targetemb, emb) / (np.linalg.norm(targetemb) * np.linalg.norm(emb)))
        results.append({
            "id": row.id,
            "name": row.name,
            "similarity": round(similarity, 4),
            "x1": row.x1,
            "y1": row.y1,
            "x2": row.x2,
            "y2": row.y2,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    results = results[:limit]

    for result in results:
        with open(result['name'], "rb") as f:
            result["image"] = base64.b64encode(f.read()).decode("utf-8")

    return results

