import os
import re
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


ps = PorterStemmer()
vocabulary = set()
f_path = "stop-words.txt" 
with open(f_path, 'r') as f:
    stop_word = set() 
    for line in f:
        stop_word.update(line.strip().split())

class Preprocessor:
    def __init__(self):
        self.vocab = defaultdict(dict)
        self.corpus = {}
        self.docfreqs = defaultdict(int)
        

    #Best ive gotten it, removes punctuation, removes double apostrophes, removes special characters, one thing ive noticed since we removed punctuation
    # we also remove and tokenize abbreviations differently U.S. to ["U", "S"]
    
    # 1. Remove single words in corpus
    # 2. Deal with abbreviations
    # 3. Delete numbers
    # 4. Stemming
        
    # 5. Try adding description to query
    # 6. Try adding document title for document text 
    

        
    def preprocess(self, doc_folder_path):
        tokens = []
        corpus = []
        text_to_doc_mapping = {}
        docs = []
        for f in os.listdir(doc_folder_path):
            f_path = os.path.join(doc_folder_path, f)
            with open(f_path, 'r') as x:
                file = x.read()
                documents = file.split('<DOC>')
                for doc in documents[1:]:
                    lines = doc.replace('\n', ' ')
                    pattern_n = re.compile(r'<DOCNO>(.*?)</DOCNO>', re.DOTALL)
                    pattern_t = re.compile(r'<TEXT>(.*?)</TEXT>', re.DOTALL)
                    docno = pattern_n.findall(lines)
                    text_matches = pattern_t.findall(lines)
                    combined_text = []
                    if len(text_matches) >= 1:
                        for text in text_matches:
                            combined_text.append(text)
                        combined_text = ' '.join(combined_text)
                        corpus.append(combined_text)
                        docs.append(docno[0].strip())
                        #text_to_doc_mapping[combined_text] = docno[0].strip()
        
        print(len(corpus))
        #print(len(text_to_doc_mapping))
        return corpus, docs
                        
        #                 ret = self.tokenize(combined_text)
        #                 print(docno[0].strip())
        #                 tokens.append(ret)
        #                 text = " ".join(ret)
        #                 corpus.append(text)
        #                 text_to_doc_mapping[text] = docno[0].strip()
        # return tokens, corpus, text_to_doc_mapping
    
    def preprocessBM25(self, doc_folder_path):
        tokens = []
        corpus = []
        text_to_doc_mapping = {}
        docs = []
        for f in os.listdir(doc_folder_path):
            f_path = os.path.join(doc_folder_path, f)
            with open(f_path, 'r') as x:
                file = x.read()
                documents = file.split('<DOC>')
                for doc in documents[1:]:
                    lines = doc.replace('\n', ' ')
                    pattern_n = re.compile(r'<DOCNO>(.*?)</DOCNO>', re.DOTALL)
                    pattern_t = re.compile(r'<TEXT>(.*?)</TEXT>', re.DOTALL)
                    docno = pattern_n.findall(lines)
                    text_matches = pattern_t.findall(lines)
                    combined_text = []
                    if len(text_matches) >= 1:
                        for text in text_matches:
                            combined_text.append(text)
                        combined_text = ' '.join(combined_text)
                        
                        ret = tokenize(combined_text)
                        print(docno[0].strip())
                        tokens.append(ret)
                        text = " ".join(ret)
                        corpus.append(text)
                        docs.append(docno[0].strip())

        return tokens, corpus, docs
    

    # def removeSingles(self, vocab, docfreqs):
    #     for word, freq in docfreqs.items():
    #         if freq == 1:
    #             del docfreqs[word]
    #             del

    def processRelevanceScores(self):
        filepath = "relevance.txt"
        relevantDocs = {}

        with open(filepath, "r") as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()
            data = line.split(" ")
            relevant = data[3]
            if relevant == "1":
                queryNum = data[0]
                docNum = data[2]
                if queryNum in relevantDocs:
                    relevantDocs[queryNum].append(docNum)
                else:
                    relevantDocs[queryNum] = [docNum]
        
        return relevantDocs

    def processQueries(self):
        queries = []
        with open('./text-queries.txt', 'r') as f:
            query_info = {}
            for line in f:
                line = line.strip()
                if line.startswith("<num>"):
                    if query_info:  # If query_info is not empty, it means we finished processing the previous query
                        queries.append(query_info)
                        query_info = {}  # Reset query_info for the next query
                    query_info["num"] = int(line.split(">")[1])
                elif line.startswith("<title>"):
                    query_info["title"] = line.split(">")[1]
                elif line.startswith("<desc>"):
                    query_info["desc"] = ""
                elif line.startswith("<narr>"):
                    query_info["narr"] = ""
                elif line.startswith("</top>"):
                    pass  # End of query, ignore
                else:
                    if "desc" in query_info:
                        query_info["desc"] += line
                    elif "narr" in query_info:
                        query_info["narr"] += line
        # Append the last query after the loop
        if query_info:
            queries.append(query_info)
        return queries
    
def tokenize(text):
    # text = text.lower()
    # text = re.sub(r'[^\w\s\']|_', ' ', text)
    # text = re.sub(r'\'{2,}', ' ', text)
    # tokens = text.split()
    # return tokens
    
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens
