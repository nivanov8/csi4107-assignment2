from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import numpy as np
import tensorflow_hub as hubs

class USE:
    def __init__(self):
        revision = None
        self.modelPath = snapshot_download(repo_id="Dimitre/universal-sentence-encoder")
        self.model = hubs.KerasLayer(handle=self.modelPath)

    def get_embeddings(self, text):
        return self.model(text)

    def get_embedding_dict(self, text, text_to_doc_mapping):
        embeddings = self.get_embeddings(text)
        d = {}
        for i in range(len(embeddings)):
            #embedding = np.round(embeddings[i], decimals=5)
            embedding = embeddings[i]
            d[tuple(embedding)] = [text[i], text_to_doc_mapping[text[i]]]

        return d