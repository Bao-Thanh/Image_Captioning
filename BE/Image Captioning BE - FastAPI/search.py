import pymongo

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

    def replace_characters_with_space(caption):
        replacements = [',', '.', ';', '"', '~', '`', '-', '_', '+', '=', "'", "?", "/", "\\", "|", "<", ">"]
        for char in replacements:
            caption = caption.replace(char, ' ')
        return caption

    def remove_duplicate(caption):
        text = caption.lower()
        words = text.split()
        unique_words = list(set(words))
        result = ' '.join(unique_words)
        return result


    def get_words(caption):
        words = caption.split()
        return words

    def compare_caption(input_caption, caption):
        input_list = sorted(get_words(replace_characters_with_space(remove_duplicate(input_caption))))
        caption_list = sorted(get_words(replace_characters_with_space(remove_duplicate(caption))))
        caption_len = len(caption_list)
        common_elements = set(input_list).intersection(caption_list)
        count = len(common_elements)
        return round(((count/caption_len)*100), 2)   



    for cap in caption:
        cap["score"] = compare_caption(input_caption, cap["caption"])

    sorted_caption_list = sorted(caption, key=lambda x: x["score"], reverse=True)

    sorted_compare_caption_list_with_percent = [
        {
            "caption": cap["caption"],
            "number": cap["number"],
            "score": f"{cap['score']}%",
            "url": cap["url"]
        }
        for cap in sorted_caption_list
    ]
    return sorted_compare_caption_list_with_percent