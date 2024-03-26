from core.preprocessing.preprocessor import Preprocessor
from core.encoding.t5base import T5Base
import argparse

from sentence_transformers import SentenceTransformer

import numpy as np

import torch



def main():
    query = "Coping with overcrowded prisons"

    print("Start of info retrieval script...")
    folder_path = "collection"
    preprocessor = Preprocessor()
    print("Starting preprocessing and indexing...")
    passages, docs = preprocessor.preprocess(folder_path)

    encoder = T5Base()
    embeddings = encoder.encode(passages[:5000])

    query_embedding = encoder.encode(query)

    maxx = 0
    ind = None
    for i in range(len(embeddings)):
        passage = embeddings[i]
        dp = np.dot(passage, query_embedding)
        if dp > maxx:
            maxx = dp
            ind = i
        #print(torch.nn.functional.cosine_similarity(torch.tensor(passage), torch.tensor(query_embedding), dim=-1))

    print(maxx)
    print(passages[ind])
    #dp = np.dot(embeddings[1], query_embedding)

    #print(dp)







    

        

    
if __name__ == "__main__":
    main()

    



    



    # inverted_index, bm25_input, corpus = preprocessor.preprocess(folder_path)
    # print("Preprocessing complete...")
    # retriever = BM25(inverted_index, corpus)
    # retriever.computeDocumentVectors(bm25_input)
    # query = "Causes and treatments of multiple sclerosis"
    # retriever.retreive(query)
    

    # retriever = TFIDF()
    # retriever.compute(tf, df, num_documents)
    # print(df["gain"])

    # preprocessor = Preprocessor()
    # print(preprocessor.tokenize("I didn't go to the party because I hadn't finished my work, which includes underscores_like_this. I dont like it.''")
    




  