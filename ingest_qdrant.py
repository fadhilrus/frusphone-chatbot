import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CSV_PATH = os.getenv("CSV_PATH", "products.csv")

genai.configure(api_key=GEMINI_API_KEY)

def get_embedding(text: str):
    if not text or not text.strip():
        raise ValueError("Empty text for embedding")
    result = genai.embed_content(
        model="gemini-embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    if result is None:
        raise ValueError("Embedding API returned None")
    if isinstance(result, dict):
        emb = result.get("embedding")
    else:
        emb = getattr(result, "embedding", None)
    if emb is None:
        raise ValueError("No embedding in response")
    return emb

def create_text_representation(row):
    return f"{row['name']}: {row['description']}. Price: {row['price']} IDR. Available: {row['available']}"

def main():
    print("=" * 60)
    print("STARTING DATA INGESTION TO QDRANT")
    print("=" * 60)
    print(f"CSV Path      : {CSV_PATH}")
    print(f"Collection    : {COLLECTION_NAME}")
    print(f"Embed Model   : gemini-embedding-001 (3072 dim)")
    print("=" * 60)

    try:
        df = pd.read_csv(CSV_PATH)
        print(f"[PREPROCESSING] Loaded {len(df)} products from {CSV_PATH}")
        print(f"[PREPROCESSING] Columns found: {list(df.columns)}")
    except Exception as e:
        print(f"Failed to read CSV from {CSV_PATH}: {e}")
        return

    required_cols = {'id', 'name', 'image_url', 'description', 'available', 'price'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing columns in CSV: {missing}")
        return

    df = df.fillna({
        'name': '', 'image_url': '', 'description': '', 'available': 0, 'price': 0
    })
    print("Missing values handled (filled with defaults)")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    if COLLECTION_NAME in collection_names:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=3072,
            distance=Distance.COSINE
        )
    )
    print(f"Created collection: {COLLECTION_NAME} (size=3072, distance=COSINE)")

    points = []
    failed = []
    for idx, row in df.iterrows():
        try:
            product_id = row['id']
            if product_id is None or (isinstance(product_id, float) and pd.isna(product_id)):
                print(f"Skipping row {idx}: missing id")
                failed.append(idx)
                continue
            product_id = int(product_id)

            text = create_text_representation(row)
            print(f"  -> Text representation: {text}")
            print(f"  -> Generating embedding for: {row['name']} (id={product_id})")
            embedding = get_embedding(text)
            print(f"  -> Embedding generated successfully (dim={len(embedding)})")

            point = PointStruct(
                id=product_id,
                vector=embedding,
                payload={
                    "name": str(row["name"]),
                    "image_url": str(row["image_url"]),
                    "description": str(row["description"]),
                    "available": int(row["available"]),
                    "price": float(row["price"]) if isinstance(row["price"], (int, float)) else 0.0
                }
            )
            points.append(point)
        except Exception as e:
            print(f"Failed to process product id={row.get('id', 'unknown')} name={row.get('name', 'unknown')}: {e}")
            failed.append(idx)

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"\n[UPSERT] Successfully inserted {len(points)} products into Qdrant")
    else:
        print("No products were successfully processed")

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Total products in CSV  : {len(df)}")
    print(f"Successfully embedded  : {len(points)}")
    print(f"Failed to process      : {len(failed)}")
    if failed:
        print(f"Failed indices         : {failed}")
    print("=" * 60)

if __name__ == "__main__":
    main()