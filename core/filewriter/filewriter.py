from core.bm25.BM25 import BM25
from core.preprocessing.preprocessor import tokenize
from core.reranking.minilmv2 import Minilmv2
from core.reranking.GIST import GIST
from core.reranking.mxbai import mxbai

from sentence_transformers import util

import numpy as np

class Filewriter:
    def __init__(self, tokens, passages, docs, passage_to_doc_mapping, tokens_to_passage):
        self.passages = passages
        self.passage_to_doc_mapping = passage_to_doc_mapping
        self.docs = docs
        self.tokens_to_passage = tokens_to_passage
        self.bm25 = BM25(tokens)
        self.reranker = GIST()

    def writeOutput(self, queries):
        docs_retrieved = {}

        n = 1000

        
        for d in queries:
            query_num = d['num']
            query_title = d["title"]
            query_desc = d["desc"]

            


            query = query_title + query_desc

            top = self.bm25.retrieve_top_n(query, n)

            #ind_look = self.passages.index('    VaporSimac steam irons are being recalled because of danger of starting fires, the U.S. Consumer Product Safety Commission announced today.    About 10,000 VaporSimac irons have been sold nationwide by major department stores and sewing specialty centers, the agency said. The irons, which cost about $100 each, were sold between 1985 and 1987.    The recall was initiated following seven reports of fire, including one that caused a burn injury, the commission reported.    The irons come with a separate, transparent plastic water reservoir and are designed to operate both vertically and horizontally, the agency said.    Owners should immediately stop using the irons, and return them to Electra Craft, 250 Halsey St., Newark, N.J., 07102, for a refund including postage costs.    Persons needing more information can contact Electra Craft at 1-800-223-1898 or the Safety Commission at 1-800-638-2772. ')

            #query = "Represent this sentence for searching relevant passages: " + query

            bm25Pairs = [self.passages[top[i][1]] for i in range(len(top))]
            #bm25Pairs.insert(0, query)

            embeddings_dict = self.reranker.get_embedding_dict(bm25Pairs, self.passage_to_doc_mapping) #tuple: [text, docNo]


            query_embedding = self.reranker.get_embeddings([query])

            rerank = []
            for embedding in embeddings_dict.keys():
                #print(embeddings_dict[embedding][1])
                sim_score = util.cos_sim(query_embedding, np.array(embedding))
                rerank.append([np.round(sim_score.item(), decimals=4), embedding])
            #print(rerank)

            rerank_tuples = [(x[0], tuple(x[1])) for x in rerank]
            unique_rerank = list(set(rerank_tuples))

            sorted_rerank = sorted(unique_rerank, key=lambda x: x[0], reverse=True)[:1000]

            #for score, embedding in sorted_rerank:
            #    if score == 0.1953:
            #        print(embedding)
            
            retrieved = []
            scores = []

            for ret in sorted_rerank:
                score = ret[0]
                embedding = ret[1]

                doc_num = embeddings_dict[embedding][1]

                retrieved.append(doc_num)
                scores.append(score)

            docs_retrieved[query_num] = (retrieved, scores)
            print(query_num)
        
            
        f = open("Results.txt", "w")
        for query_num, tup in docs_retrieved.items():
            docs = tup[0]
            scores = tup[1]
            for j in range(len(docs)):
                f.write(f"{query_num} Q0 {docs[j]} {j+1} {scores[j]} BM25 \n")


    def writeOutputTest(self, query):
        docs_retrieved = {}

        n = 2000
        top = self.bm25.retrieve_top_n(query, n)

        print("----------------")
        # check = {}
        # for score, index in top:
        #     if self.docs[index] not in check:
        #         check[self.docs[index]] = 1
        #     else:
        #         check[self.docs[index]] += 1
        
        # for k,v in check.items():
        #     if v == 2:
        #         print(k)

        bm25Pairs = [self.passages[top[i][1]] for i in range(len(top))]
        
        reranker = Minilmv2()
        embeddings_dict = reranker.get_embedding_dict(bm25Pairs, self.passage_to_doc_mapping) #tuple: [text, docNo]

        query_embedding = reranker.get_embeddings([query])

        rerank = []
        for embedding in embeddings_dict.keys():
            sim_score = util.cos_sim(query_embedding, np.array(embedding))
            rerank.append([sim_score.item(), embedding])

        sorted_rerank = sorted(rerank, key=lambda x: x[0], reverse=True)[:1000]

        # check = {}
        # for score, embedding in sorted_rerank:
        #     if embeddings_dict[embedding][1] not in check:
        #         check[embeddings_dict[embedding][1]] = 1
        #     else:
        #         check[embeddings_dict[embedding][1]] += 1
        
        # for k, v in check.items():
        #     if v > 1:
        #         print(k)
        
        retrieved = []
        scores = []
        for ret in sorted_rerank:
            score = ret[0]
            embedding = ret[1]

            doc_num = embeddings_dict[embedding][1]

            retrieved.append(doc_num)
            scores.append(score)

        docs_retrieved[43] = (retrieved, scores)
        print(43)
    
        
        f = open("Results.txt", "w")
        for query_num, tup in docs_retrieved.items():
            docs = tup[0]
            scores = tup[1]
            for j in range(len(docs)):
                f.write(f"{query_num} Q0 {docs[j]} {j+1} {scores[j]} BM25 \n")

            