from core.preprocessing.preprocessor import Preprocessor
from core.encoding.t5base import T5Base
from core.bm25.BM25 import BM25
from core.reranking.minilmv2 import Minilmv2
from core.encoding.sfrmistral import SFRMistral
from core.filewriter.filewriter import Filewriter
import argparse

from sentence_transformers import SentenceTransformer, util

import numpy as np

import torch



def main():
    query = "Coping with overcrowded prisons."
    query2 = "Accusations of Cheating by Contractors on U.S. Defense Projects"
    print("Start of info retrieval script...")
    folder_path = "collection"
    preprocessor = Preprocessor()
    print("Starting preprocessing and indexing...")
    tokens_to_passage, passagesStemmed, passages, docs = preprocessor.preprocessBM25(folder_path)

    #tokens = tokens.keys()

    passage_to_doc_mapping = {}

    #print(len(set(tokens)) == len(tokens))

    for i in range(len(passages)):
        passage_to_doc_mapping[passages[i]] = docs[i]

    queries = preprocessor.processQueries()

    query = "Toys R Dangerous Document will report the dangers of certain toys to children and the activities by consumer safety groups and parents against production and retail of these dangerous toys."

    filewriter = Filewriter(tokens_to_passage, passages, docs, passage_to_doc_mapping, tokens_to_passage)
    filewriter.writeOutput(queries)

    #filewriter.writeOutputTest(query)

    #bm25 = BM25(tokens)
    #top = bm25.retrieve_top_n(query, 2000) #returns list of tuples in sorted order (score, index)

    #bm25Pairs = [passages[top[i][1]] for i in range(len(top))]

    #print(bm25Pairs)

    # reranker = Minilmv2()
    # embeddings_dict = reranker.get_embedding_dict(bm25Pairs) #tuple: string

    # query_embedding = reranker.get_embeddings([query])


    # rerank = []
    # for embedding in embeddings_dict.keys():
    #     sim_score = util.cos_sim(query_embedding, np.array(embedding))
    #     rerank.append([sim_score.item(), embedding])

    
    # sorted_rerank = sorted(rerank, key=lambda x: x[0], reverse=True)
    # print(embeddings_dict[sorted_rerank[0][1]])


    #model = SFRMistral()

    #queries = model.encodeQueries([query, query2])

    #embeddings = model.model.encode(queries + passages[:1])
    #scores = util.cos_sim(embeddings[:2], embeddings[2:]) * 100
    #print(scores.tolist())


    # encoder = T5Base()
    # embeddings = encoder.encode(passages[:1000])

    # query_embedding = encoder.encode(query)

    # maxx = 0
    # ind = None
    # for i in range(len(embeddings)):
    #     passage = embeddings[i]
    #     dp = np.dot(passage, query_embedding)
    #     if dp > maxx:
    #         maxx = dp
    #         ind = i
    #     #print(torch.nn.functional.cosine_similarity(torch.tensor(passage), torch.tensor(query_embedding), dim=-1))

    # print(maxx)
    # print(passages[ind])
    # #dp = np.dot(embeddings[1], query_embedding)

    # bm25Pairs.append((query, passages[ind]))
    # print(reranker.predict(bm25Pairs))

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
    




  