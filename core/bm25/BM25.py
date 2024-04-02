from rank_bm25 import BM25Okapi
import heapq
from core.preprocessing.preprocessor import tokenize

class BM25:
    def __init__(self, tokenized_corpus):
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve_top_n(self, query, n):
        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)
        top_n = self.top_k_indices(scores, n)
        return top_n


    def top_k_indices(self, arr, k):
        heap = []
        indices_added = set()
        for i, num in enumerate(arr):
            if len(heap) < k:
                heapq.heappush(heap, (num, i))
                indices_added.add(i)
            else:
                if num > heap[0][0] and i not in indices_added:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (num, i))
                    indices_added.add(i)
        return sorted(heap, reverse=True)
