import sqlite3
import sys
import pandas as pd
import cv2

def dbtodf(dbFile):
    with sqlite3.connect(dbFile) as conn:
        selectQuery = "SELECT * FROM files;"
        df = pd.read_sql_query(selectQuery, conn)
        return df

def showimage(df, id):
    row = df[df['id'] == id].iloc[0]
    img = cv2.imread(row['name'])
    if img is None:
        print(f"Could not load image: {row['name']}")
        return
    x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
    annotatedImg = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imshow("Face", annotatedImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide the Faces database file and try again")
        sys.exit(1)
    facesdb = dbtodf(sys.argv[1])
    print(facesdb)