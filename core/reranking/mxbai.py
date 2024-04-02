from sentence_transformers import SentenceTransformer
import numpy as np

class mxbai:
    def __init__(self):
        revision = None
        self.model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", revision=revision)

    def get_embeddings(self, text):
        return self.model.encode(text)
    
    def get_embedding_dict(self, text, text_to_doc_mapping):
        embeddings = self.get_embeddings(text)
        d = {}
        for i in range(len(embeddings)):
            #embedding = np.round(embeddings[i], decimals=5)
            if i == 0:
                continue
            embedding = embeddings[i]
            d[tuple(embedding)] = [text[i], text_to_doc_mapping[text[i]]]

        return d