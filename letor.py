import random
import pickle
import lightgbm as lgb
import numpy as np
from scipy.spatial.distance import cosine

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary

class NFCorpusDataset:
	"""
	Class untuk dataset nf corpus pada domain Medical Information Retrieval.
	Link dataset: https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/

	Attributes
	----------
	doc_dir(str): Path ke koleksi dokumen
	query_dir(str): Path ke koleksi query
	qrel_dir(str): Path ke koleksi query relevancy
	num_negatives(int): banyaknya dokumen tambahan yang tidak terkait pada suatu query 
	"""
	def __init__(self, doc_dir = "", query_dir = "", qrel_dir = "", num_negatives = 1):
		self.doc_dir = doc_dir
		self.query_dir = query_dir
		self.qrel_dir = qrel_dir
		self.num_negatives = num_negatives
		self.documents = None
		self.queries = None
		self.rels = None
		self.dataset = None
		self.model = None

	def parse_docs(self):
		"""
		Parsing terhadap text file dengan format (FORMAT: DOC_ID CONTENT).

		Returns
		-------
		Dict[str, List[str]]
			Mapping antara id dokumen dengan token dokumen tersebut.
		"""
		documents = {}
		with open(self.doc_dir, encoding="utf-8") as file:
		  for line in file:
		    doc_id, content = line.split("\t")
		    documents[doc_id] = content.split()
		return documents

	def parse_queries(self):
		"""
		Parsing terhadap text file dengan format (FORMAT: QUERY_ID QUERY_TEXT).

		Returns
		-------
		Dict[str, List[str]]
			Mapping antara id query dengan token query tersebut.
		"""
		queries = {}
		with open(self.query_dir, encoding="utf-8") as file:
		  for line in file:
		  	q_id, content = line.split("\t")
		  	queries[q_id] = content.split()
		return queries

	def parse_rels(self):
		"""
		Parsing terhadap text file dengan format (FORMAT: QUERY_ID 0 DOC_ID RELEVANCE_LEVEL).

		Returns
		-------
		Dict[str, List[Tuple[str, int]]]
			Mapping antara id query dengan kumpulan dokumen beserta relevansi dokumen tersebut
			yang terkait dengan id query tersebut.
		"""
		queries = self.getQueries()
		documents = self.getDocuments()
		q_docs_rel = {} # grouping by q_id terlebih dahulu
		with open(self.qrel_dir, encoding="utf-8") as file:
		  for line in file:
		    q_id, _, doc_id, rel = line.split("\t")
		    if (q_id in queries) and (doc_id in documents):
		      if q_id not in q_docs_rel:
		        q_docs_rel[q_id] = []
		      q_docs_rel[q_id].append((doc_id, int(rel)))
		return q_docs_rel

	def build(self):
		"""
		Membuat dataset untuk training LambdaMART model dengan format
		[(query_text, document_text, relevance), ...]
		relevance awalnya bernilai 1, 2, 3.
		relevance level: 3 (fully relevant), 2 (partially relevant), 1 (marginally relevant).
		"""
		queries = self.getQueries()
		documents = self.getDocuments()
		q_docs_rel = self.getRels()
		dataset = []
		for q_id in q_docs_rel:
		  docs_rels = q_docs_rel[q_id]
		  for doc_id, rel in docs_rels:
		    dataset.append((queries[q_id], documents[doc_id], rel))
		  # tambahkan num negative (random sampling saja dari documents)
		  for i in range(self.num_negatives):
		  	dataset.append((queries[q_id], random.choice(list(documents.values())), 0))
		self.dataset = dataset

	def getDocuments(self):
		if self.documents == None:
			self.documents = self.parse_docs()
		return self.documents

	def getQueries(self):
		if self.queries == None:
			self.queries = self.parse_queries()
		return self.queries

	def getRels(self):
		if self.rels == None:
			self.rels = self.parse_rels()
		return self.rels

	def getModel(self):
		"""
		Returns
		-------
		LsiBoWModel
			Latent Semantic Indexing/Analysis (LSI/A) model untuk representasi dokumen.
		"""
		if self.model == None:
			self.model = LsiBoWModel(self.getDocuments())
		return self.model

	def getX(self):
		"""
		Returns
		-------
		2d numpy array
			Array of word embedding suatu query dan dokumen.
		"""
		if self.dataset == None:
			raise Exception("Dataset has not been initialized. Call build() first!")
		X = []
		self.model = self.getModel()
		for (query, doc, _) in self.dataset:
			X.append(self.model.features(query, doc))
		return np.array(X)

	def getY(self):
		"""
		Returns
		-------
		numpy array
			Array yang berisi relevansi suatu dokumen terhadap query pada index X.
			relevance level: 3 (fully relevant), 2 (partially relevant), 1 (marginally relevant).
		"""
		if self.dataset == None:
			raise Exception("Dataset has not been initialized. Call build() first!")
		Y = []
		for (_, _, rel) in self.dataset:
		  Y.append(rel)
		return np.array(Y)

	def getGroup(self):
		"""
		Data group/query untuk task learning-to-rank. 
		jumlah(group) = n_sampel. 
		Misalnya, group = [10, 20, 40, 10, 10, 10]. 
		Itu berarti terdapat 6 grup. 
		10 catatan pertama berada di grup pertama, catatan 11- 30 berada di kelompok kedua, rekor 31-70 berada di kelompok ketiga, dst.

		Returns
		-------
		List[int]
			Data group/query untuk task learning-to-rank.
		"""
		q_docs_rel = self.getRels()
		group_qid_count = []
		for q_id in q_docs_rel:
		  docs_rels = q_docs_rel[q_id]
		  group_qid_count.append(len(docs_rels) + self.num_negatives)
		return group_qid_count

class LsiBoWModel:
	"""
	Latent Semantic Indexing/Analysis (LSI/A) model untuk representasi dokumen.

	Attributes
	----------
	documents(Dict[str, List[str]]): Mapping antara id dokumen dengan token dokumen tersebut
	num_latent_topics(int): Banyaknya topik abstrak yang tidak dapat diamati secara langsung. 
	Topik-topik ini muncul dari sebuah teks berdasarkan struktur semantik yang dapat diamati di dalam teks tersebut.
	"""
	def __init__(self, documents, num_latent_topics = 200):
		self.num_latent_topics = num_latent_topics
		self.dictionary = Dictionary()
		bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in documents.values()]
		self.model = LsiModel(bow_corpus, num_topics = self.num_latent_topics)

	def vector_rep(self, text):
		"""Mendapatkan word embedding sebuah text"""
		rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
		return rep if len(rep) == self.num_latent_topics else [0.] * self.num_latent_topics

	def features(self, query, doc):
		"""
		Mendapatkan word embedding sebuah query dan dokumen.

		word embedding = concat(vector(query), vector(doc)) + informasi lain.
		informasi lain -> cosine distance & jaccard similarity antara query dan dokumen.
		"""
		v_q = self.vector_rep(query)
		v_d = self.vector_rep(doc)
		q = set(query)
		d = set(doc)
		cosine_dist = cosine(v_q, v_d)
		jaccard = len(q & d) / len(q | d)
		return v_q + v_d + [jaccard] + [cosine_dist]

class Letor:
	"""
	Class yang mengimplementasikan model LambdaMART
	untuk meranking ulang top-K dokumen.

	Attributes
    ----------
    save_dir(str): Path ke pre-trained model untuk di load
    param(dict): Parameter untuk inisialisasi LambdaMART (akan diabaikan jika save_dir terisi)
	"""
	def __init__(self, save_dir = None, param = dict(objective="lambdarank",
                    boosting_type = "gbdt",
                    n_estimators = 100,
                    importance_type = "gain",
                    metric = "ndcg",
                    num_leaves = 40,
                    learning_rate = 0.02,
                    max_depth = -1)):
		if save_dir == None:
			self.ranker = lgb.LGBMRanker(**param)
			self.save_dir = "model.txt"
		else:
			self.save_dir = save_dir
			with open(self.save_dir, 'rb') as f:
				self.ranker = pickle.load(f)

	def save(self, save_dir = None):
		"""Menyimpan trained model ke save directory via pickle"""
		if save_dir == None:
			save_dir = self.save_dir

		with open(save_dir, 'wb') as f:
			pickle.dump(self.ranker, f)

	def train(self, X, y, group):
		"""
		Melakukan training learning-to-rank model tanpa cross-validation.

		Parameters
		----------
		X : 2d numpy array
			Array of word embedding suatu query dan dokumen.
		y : numpy array
			Array yang berisi relevansi suatu dokumen terhadap query pada index X.
			relevance level: 3 (fully relevant), 2 (partially relevant), 1 (marginally relevant).
		group : List[int]
			Data group/query untuk task learning-to-rank. 
			jumlah(group) = n_sampel. 
			Misalnya, group = [10, 20, 40, 10, 10, 10]. 
			Itu berarti terdapat 6 grup. 
			10 catatan pertama berada di grup pertama, catatan 11- 30 berada di kelompok kedua, rekor 31-70 berada di kelompok ketiga, dst.
		"""
		self.ranker.fit(X, y, group = group, verbose = 10)

	def rank(self, query_docs_vec, docs_id):
		"""
		Melakukan prediksi nilai relevansi suatu list of documents terhadap suatu query.
		Mengembalikan 

		Parameters
		----------
		query_docs_vec : 2d numpy array
			Array of word embedding suatu query dan dokumen.
		docs_id : List[str]
			Id dokumen pada untuk dokumen pada index query_docs_vec.

		Returns
		-------
		List[(int, str)]
            List of tuple: elemen pertama adalah score relevansi, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
		"""
		X_unseen = np.array(query_docs_vec)
		scores = self.ranker.predict(X_unseen)
		did_scores = [x for x in zip(scores, docs_id)]
		sorted_did_scores = sorted(did_scores, reverse = True)
		return sorted_did_scores

if __name__ == "__main__":

	doc_dir = "nfcorpus/train.docs"
	query_dir = "nfcorpus/train.vid-desc.queries"
	qrel_dir = "nfcorpus/train.3-2-1.qrel"
	corpus = NFCorpusDataset(doc_dir, query_dir, qrel_dir)
	corpus.build()
	params = dict(objective="lambdarank",
                    boosting_type = "goss",
                    n_estimators = 100,
                    importance_type = "gain",
                    metric = "ndcg",
                    num_leaves = 40,
                    learning_rate = 0.02,
                    reg_alpha = 0.005,
                    max_depth = -1)
	ranker = Letor(param = params)
	ranker.train(corpus.getX(), corpus.getY(), corpus.getGroup())
	ranker.save('model2.txt')