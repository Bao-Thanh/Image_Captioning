import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = pymongo.MongoClient("mongodb://localhost:27017")

db = client["ImageCaption"]
collection = db["search"]

def save(caption, url):
    # Check if the URL already exists in the collection
    existing_document = collection.find_one({"url": url})
    
    if existing_document:
        print("URL already exists in the database. Skipping save.")
    else:
        # Create a new document with caption and URL
        document = {"caption": caption, "url": url}
        
        # Insert the document into the collection
        collection.insert_one(document)
        
        print("Caption and URL saved successfully.")

    