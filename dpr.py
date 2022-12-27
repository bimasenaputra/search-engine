from transformers import DPRQuestionEncoder, AutoTokenizer, DPRContextEncoder
import torch

from haystack import Document
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore

class BioDPR:

	def __init__(self, save_dir = ''):
		self.retriever = DensePassageRetriever(
			document_store = InMemoryDocumentStore(),
			query_embedding_model='deepset/bert-small-mm_retrieval-question_encoder',
			passage_embedding_model='deepset/bert-small-mm_retrieval-passage_encoder',
			max_seq_len_query=512,
			max_seq_len_passage=512,
		)

	def load(self, save_dir):
		self.retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=InMemoryDocumentStore())

	def train(self, doc_dir, train_filename, save_dir = 'biodpr'):
		self.retriever.train(
			    data_dir=doc_dir,
			    train_filename=train_filename,
			    dev_split=0.2,
			    n_epochs=1,
			    batch_size=16,
			    grad_acc_steps=8,
			    save_dir=save_dir,
			    evaluate_every=3000,
			    num_positives=1,
			    num_hard_negatives=1,
		)

	def query_embeddings(self, query):
		embeddings = self.retriever.embed_queries([query])
		return embeddings[0]

	def passage_embeddings(self, docs):
		documents = []
		for doc in docs:
			documents.append(Document(content=doc,content_type='text'))
		embeddings = self.retriever.embed_documents(documents)
		return embeddings

	def similarity(self, query, docs):
		query_embeddings = self.query_embeddings(query)
		docs_embeddings = self.passage_embeddings(docs)
		scores = docs_embeddings.dot(query_embeddings)
		return scores.tolist()

class DPR:

	def __init__(self):
		self.query_tokenizer = AutoTokenizer.from_pretrained('deepset/bert-small-mm_retrieval-question_encoder', model_max_length=512)
		self.query_model = DPRQuestionEncoder.from_pretrained('deepset/bert-small-mm_retrieval-question_encoder')
		self.passage_tokenizer = AutoTokenizer.from_pretrained('deepset/bert-small-mm_retrieval-passage_encoder', model_max_length=512)
		self.passage_model = DPRContextEncoder.from_pretrained('deepset/bert-small-mm_retrieval-passage_encoder')

	def query_embeddings(self, query):
		input_ids = self.query_tokenizer(query, truncation=True, return_tensors='pt')['input_ids']
		embeddings = self.query_model(input_ids).pooler_output
		return embeddings[0]

	def passage_embeddings(self, doc):
		input_ids = self.passage_tokenizer(doc, truncation=True, padding=True, return_tensors='pt')['input_ids']
		embeddings = self.passage_model(input_ids).pooler_output
		return embeddings

	def similarity(self, query, docs):
		query_embeddings = self.query_embeddings(query)
		docs_embeddings = self.passage_embeddings(docs)
		scores = torch.matmul(docs_embeddings, query_embeddings)
		return scores.tolist()

if __name__ == "__main__":

	doc_dir = "bioasq"
	train_filename = "train.json"
	retriever = BioDPR()
	#retriever.load('biodpr')
	retriever.train(doc_dir, train_filename)
