import os
import base64
from pymongo import MongoClient

from caption import predict

client = MongoClient('mongodb://localhost:27017/')
db = client['ImageCaption']
collection = db['search']


def process_images():
    img_folder = 'img' 
    order = 1
    for filename in os.listdir(img_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(img_folder, filename)

            with open(img_path, 'rb') as f:
                img_data = f.read()

            encoded_img_data = base64.b64encode(img_data).decode('utf-8')

            imgUrl = 'data:image/jpeg;base64,' + encoded_img_data

            caption = predict(imgUrl)
            if caption:
                start = "<start>"
                end = "<end>"

                start_idx = caption.find(start) + len(start)
                end_idx = caption.find(end)

                result = caption[start_idx:end_idx].strip()
                print(result)
            doc = {
                'number': order,
                'caption': result,
                'url': imgUrl
            }
            collection.insert_one(doc)
            order += 1

process_images()

client.close()
