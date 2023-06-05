import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer

client = pymongo.MongoClient("mongodb://localhost:27017")

db = client["ImageCaption"]
collection = db["search"]

# Create text index on the "caption" field
collection.create_index([("caption", "text")])

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on all captions
documents = collection.find()
captions = [document["caption"] for document in documents]
vectorizer.fit(captions)

def search(input_caption, top_k=100):
    input_vector = vectorizer.transform([input_caption])

    # Perform text search using full-text index and limit the number of results
    documents = collection.find({"$text": {"$search": input_caption}}).limit(top_k)

    caption_scores = []
    for document in documents:
        _id = document["_id"]
        caption_text = document["caption"]
        url = document["url"]

        caption_vector = vectorizer.transform([caption_text])

        # Calculate the percentage similarity between the input caption and the caption in the document
        similarity_score = (input_vector * caption_vector.T).A[0][0]
        similarity_score_percent = round(similarity_score * 100, 2)

        caption_scores.append({
            "_id": _id,
            "caption": caption_text,
            "score": similarity_score_percent,
            "url": url
        })

    sorted_caption_scores = sorted(caption_scores, key=lambda x: x["score"], reverse=True)
    sorted_compare_caption_list_with_percent = [
        {
            "caption": cap["caption"],
            "score": f"{cap['score']}%",
            "url": cap["url"]
        }
        for cap in sorted_caption_scores if cap['score'] > 0
    ]

    return sorted_compare_caption_list_with_percent
