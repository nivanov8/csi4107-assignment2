from sentence_transformers import SentenceTransformer

class SFRMistral:
    def __init__(self):
        self.model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'

    def encodeQueries(self, queries):
        return [self.get_detailed_instruct(self.task, x) for x in queries]


    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'