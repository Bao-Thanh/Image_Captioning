import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = pymongo.MongoClient("mongodb://localhost:27017")

db = client["ImageCaption"]
collection = db["search"]

def search(input_caption):
    documents = collection.find()

    caption = []

    for document in documents:
        _id = document["_id"]
        number = document["number"]
        caption_text = document["caption"]
        url = document["url"]

        caption_data = {
            "_id": _id,
            "number": number,
            "caption": caption_text,
            "url": url
        }

        caption.append(caption_data)

    def calculate_similarity_score(caption1, caption2):
        # Create a TfidfVectorizer
        vectorizer = TfidfVectorizer()

        # Fit the vectorizer on the two captions
        vectorizer.fit([caption1, caption2])

        # Transform the captions to vectors
        caption1_vector = vectorizer.transform([caption1])
        caption2_vector = vectorizer.transform([caption2])

        # Calculate the cosine similarity between the vectors
        similarity_score = cosine_similarity(caption1_vector, caption2_vector)[0][0]

        return round(similarity_score * 100, 2) 

    for cap in caption:
        cap["score"] = calculate_similarity_score(input_caption, cap["caption"])

    sorted_caption_list = sorted(caption, key=lambda x: x["score"], reverse=True)

    sorted_compare_caption_list_with_percent = [
        {
            "caption": cap["caption"],
            "number": cap["number"],
            "score": f"{cap['score']}%",
            "url": cap["url"]
        }
        for cap in sorted_caption_list if cap['score'] > 0
    ]

    if len(sorted_compare_caption_list_with_percent) > 100:
        sorted_compare_caption_list_with_percent = sorted_compare_caption_list_with_percent[:100]

    return sorted_compare_caption_list_with_percent

