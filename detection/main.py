import sys
from pathlib import Path
import hashlib
import cv2
import lancedb
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from tqdm import tqdm


def get_all_images(folder_path):
    image_extensions = {
        ".jpg",
        ".heic",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".tiff",
    }
    folder = Path(folder_path)
    return [
        str(file_path)
        for file_path in folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in image_extensions
    ]


def match_embeddings(emb1, emb2, threshold=0.4):
    """
    Matches two 512-d InsightFace embeddings using Cosine Distance.
    InsightFace vectors are typically normalized, so distance = 1 - dot_product.
    """
    cosine_distance = 1.0 - np.dot(emb1, emb2)
    return cosine_distance < threshold, cosine_distance


def embeddingFromFile(app, file):
    img = cv2.imread(file)
    faces = app.get(img)
    return faces[0].normed_embedding


def generateEmbeds(app, targetfolder, images):

    cachelocation = Path(targetfolder) / "findmyfacedb"
    # connecting the database is accessing the file
    db = lancedb.connect(str(cachelocation))

    # CACHE EXISTS
    if "embeddings" in db.table_names():
        print("Using existing embedding cache...")
        table = db.open_table("embeddings")
        return table.to_pandas()

    embeds = []
    for image_path in tqdm(images, desc="Generating the dataset", unit="img"):
        img = cv2.imread(image_path)
        if img is None:
            continue
        faces = app.get(img)
        for f in faces:
            embeds.append([image_path, f.normed_embedding.tolist()])
    if not embeds:
        return None
    embeds_df = pd.DataFrame(embeds, columns=["path", "embeds"])
    # Create persistent LanceDB table
    table = db.create_table("embeddings", data=embeds_df, mode="overwrite")
    return embeds_df


if __name__ == "__main__":

    images = []
    target = ""

    if len(sys.argv) >= 3:
        images = get_all_images(sys.argv[1])
        target = sys.argv[2]

    cachelocation = Path(sys.argv[1]) / "findmyfacedb"
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    # ctx_id=-1 for cpu and 0 for gpu

    # images is a list of target  images file path
    # embeds  index  : path (string): embeds (512 dimension ndarray)
    embeds_df = generateEmbeds(app, sys.argv[1], images)
    target_emb = embeddingFromFile(app, target)

    matches = pd.DataFrame()

    if embeds_df is not None and not embeds_df.empty:
        # 1. Converts series of 512-d lists/arrays to contiguous 2D NumPy matrix
        matrix = np.vstack(embeds_df["embeds"].values)

        # 2. Vectorized Cosine Distance computation (normalized embeddings)
        # this is done on the whole matrix which is the stacked NumPy Matrix
        # cosine_distance is the series of distance computed against target embeddings
        cosine_distances = 1.0 - (matrix @ target_emb)

        # 3. Filter matches under threshold (e.g., 0.4)
        embeds_df["distance"] = cosine_distances
        threshold = 0.4
        matches = embeds_df[embeds_df["distance"] < threshold].sort_values("distance")

        print(f"\nFound {len(matches)} matching faces:")

    db = lancedb.connect(str(cachelocation))     
    #convert the taret embedding to bytes and hash it using SHA-256 
    #use this for table name in the db
    targethash = hashlib.sha256(target_emb.tobytes())
    print(f"Target image hash: {targethash.hexdigest()}") 

    matches_table = db.create_table(targethash.hexdigest(), data=matches, mode="overwrite")     
    print(db.table_names())

    
    