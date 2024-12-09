import os
import csv
from openai import APIError, OpenAI
import json

filename = "embeddings.json"

api_key = "OPEN AI KEY HERE"

client = OpenAI(api_key=api_key)



def load_embeddings_file():
    if os.path.exists(filename):
        with open(filename,
                    "r") as f:
                return json.load(f)
    else:
        return {}

embeddings_file = load_embeddings_file()

def get_text_embedding(text):
    if text in embeddings_file:
        print("found in file")
        return embeddings_file[text]
    else:
        try:
            print("not found in file, creating new")
            embedding = client.embeddings.create(input=text, model="text-embedding-3-large").data[0].embedding
            embeddings_file[text] = embedding
            with open(filename, "w") as f:
                json.dump(embeddings_file, f)
            return embedding
        except APIError as e:
            print(e)
            return None







