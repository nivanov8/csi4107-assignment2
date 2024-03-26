from sentence_transformers import SentenceTransformer

class T5Base:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/sentence-t5-base')
  
    def encode(self, passages):
        return self.model.encode(passages)