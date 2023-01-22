import os
import pickle
import contextlib
import heapq
import time
import math
import spacy

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from letor import NFCorpusDataset, Letor
from dpr import BioDPR
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    doc_dir(str): Path ke koleksi dokumen untuk kebutuhan word embedding
    ranker_dir(str): Path ke pre-trained model untuk di load
    model: Untuk word embedding
    ranker: Learning-to-rank model untuk memprediksi relevansi dokumen terhadap suatu query
    DPR: Dense Passage Retriever model untuk memprediksi kemiripan dokumen terhadap query
    """
    def __init__(self, data_dir, output_dir, postings_encoding, doc_dir = "", ranker_dir = None, dpr_dir = None, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        corpus = NFCorpusDataset(doc_dir)
        self.model = corpus.getModel()
        self.ranker = Letor(ranker_dir)
        self.DPR = BioDPR()
        self.DPR.load(dpr_dir)

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        td_doc_pair = []
        block_relative_path = self.data_dir + '\\' + block_dir_relative
        files = os.listdir(block_relative_path)

        nlp = spacy.load("en_core_web_sm")

        for file in files:
            with open(block_relative_path + '\\' + file) as f:
                text = f.read().lower()
                doc = nlp(text)
                for token in doc:
                    if not token.is_punct:
                        td_doc_pair.append((self.term_id_map[token.text], self.doc_id_map[block_relative_path + '\\' + file]))

        return td_doc_pair

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = {}
            if doc_id not in term_dict[term_id]:
                term_dict[term_id][doc_id] = 0    
            term_dict[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            term_doc_freq = sorted(zip(term_dict[term_id].keys(), term_dict[term_id].values()))
            index.append(term_id, [doc_freq[0] for doc_freq in term_doc_freq], [doc_freq[1] for doc_freq in term_doc_freq])

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve(self, query):
        """
        Melakukan boolean retrieval untuk mengambil semua dokumen yang
        mengandung semua kata pada query. Jangan lupa lakukan pre-processing
        yang sama dengan yang dilakukan pada proses indexing!
        (Stemming dan Stopwords Removal)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya adalah
                    boolean query "universitas AND indonesia AND depok"

        Result
        ------
        List[str]
            Daftar dokumen terurut yang mengandung sebuah query tokens.
            Harus mengembalikan EMPTY LIST [] jika tidak ada yang match.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.
        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)

        result = None

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            for token in doc:
                if not token.is_punct:
                    if result == None:
                        if token.text in self.term_id_map.str_to_id:
                            result = merged_index.get_postings_list(self.term_id_map[token.text])
                        else:
                            return []
                    else:
                        if token.text in self.term_id_map.str_to_id:
                            result = sorted_intersect(result, merged_index.get_postings_list(self.term_id_map[token.text]))
                        else:
                            return []

        if result == None:
            return []
        else:
            return [self.doc_id_map[doc_id] for doc_id in result]

    def retrieve_wand_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema DaaT (Document-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        nlp = spacy.load("en_core_web_sm")

        doc = nlp(query)

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            term_pointers = {}
            term_wtq = {}
            term_maxub = {}
            term_doc_tf_pair = {}
            for token in doc:
                if not token.is_punct and token.text in self.term_id_map.str_to_id:
                    doc_tf_pair = merged_index.get_postings_list(self.term_id_map[token.text])
                    term_pointers[token] = 0
                    term_wtq[token] = math.log(len(merged_index.doc_length)/len(doc_tf_pair), 10)
                    doc_ids, tfs, max_ub = doc_tf_pair
                    term_maxub[token] = max_ub
                    term_doc_tf_pair[token] = (doc_ids, tfs)
            
            # WAND-K Algorithm
            evaluated_docs = []
            threshold = curdoc = 0

            while True:
                postings = []
                for term in term_pointers.keys():
                    pointer = term_pointers[term]
                    if pointer < len(term_doc_tf_pair[term][0]):
                        doc_id = term_doc_tf_pair[term][0][pointer]
                        tf = term_doc_tf_pair[term][1][pointer]
                        postings.append((doc_id, tf, term))
                    else:
                        postings.append((float('inf'),float('inf'),term))
                postings = sorted(postings)
                pivot_idx = 0
                choose_pivot_threshold = term_maxub[postings[0][2]]
                while choose_pivot_threshold < threshold:
                    pivot_idx += 1
                    choose_pivot_threshold += term_maxub[postings[pivot_idx][2]]
                
                pivot = postings[pivot_idx][0] 
                if pivot == float('inf'):
                    break

                if pivot <= curdoc:
                    # Majukan term pertama pada posting sampai doc_id >= doc_id term pertama sekarang
                    picked_term = postings[0][2]
                    term_pointers[picked_term] += 1
                else:
                    if pivot == postings[0][0]:
                        # Full evaluation
                        score = 0
                        i = 0
                        while i < len(postings) and postings[i][0] == pivot:
                            score += term_wtq[postings[i][2]] * (1 + math.log(postings[i][1], 10))
                            i += 1
                        curdoc = pivot
                        if len(evaluated_docs) == 0:
                            threshold = score
                        else:
                            threshold = min(threshold, score)
                        evaluated_docs.append((score, self.doc_id_map[pivot]))
                    else:
                        # Majukan term pertama pada posting sampai doc_id >= pivot
                        picked_term = postings[0][2]
                        picked_term_new_ptr = term_pointers[picked_term] + 1
                        while picked_term_new_ptr < len(term_doc_tf_pair[picked_term][0]) and term_doc_tf_pair[picked_term][0][picked_term_new_ptr] < pivot:
                            picked_term_new_ptr += 1
                        term_pointers[picked_term] = picked_term_new_ptr

        topk = sorted(evaluated_docs, key = lambda x: -x[0])[:k]
        return topk

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)

        scores = {}

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            for token in doc:
                if not token.is_punct and token.text in self.term_id_map.str_to_id:
                    doc_tf_pair = merged_index.get_postings_list(self.term_id_map[token.text])
                    wtq = math.log(len(merged_index.doc_length)/len(doc_tf_pair), 10)
                    doc_ids, tfs, _ = doc_tf_pair
                    for doc_id, tf in zip(doc_ids, tfs):
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += wtq * (1 + math.log(tf, 10))

        topk = sorted(zip(scores.values(), [self.doc_id_map[doc_id] for doc_id in scores.keys()]), reverse=True)[:k]
        return topk

    def retrieve_bm25(self, query, k1 = 1.2, b = 0.75, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = ((k1 + 1) * tf(t, D)) / (k1 * ((1 - b) + b * dl / avdl) + tf(t, D))

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)

        scores = {}

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            avdl = sum(merged_index.doc_length.values())/len(merged_index.doc_length)
            for token in doc:
                if not token.is_punct and token.text in self.term_id_map.str_to_id:
                    doc_tf_pair = merged_index.get_postings_list(self.term_id_map[token.text])
                    wtq = math.log(len(merged_index.doc_length)/len(doc_tf_pair), 10)
                    doc_ids, tfs, _ = doc_tf_pair
                    for doc_id, tf in zip(doc_ids, tfs):
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += wtq * (((k1 + 1) * tf) / (k1 * ((1 - b) + b * merged_index.doc_length[doc_id] / avdl) + tf))

        topk = sorted(zip(scores.values(), [self.doc_id_map[doc_id] for doc_id in scores.keys()]), reverse=True)[:k]
        return topk

    def retrieve_letor(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        Kemudian, method akan melakukan reranking top-K retrieval results
        dengan model learning-to-rank.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        topk = self.retrieve_bm25(query, k = k)
        query_doc_vec = []
        for _, did in topk:
            with open(did, encoding="utf-8") as f:
                doc = f.read()
                query_doc_vec.append(self.model.features(query.split(), doc.split())) 
        doc_id = [did for (_, did) in topk]
        reranked = self.ranker.rank(query_doc_vec, doc_id)
        return reranked

    def retrieve_dpr(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        Kemudian, method akan melakukan reranking top-K retrieval results
        dengan model DPR.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        topk = self.retrieve_bm25(query, k = k)
        docs = []
        for _, did in topk:
            with open(did, encoding="utf-8") as f:
                doc = f.read()
                docs.append(doc) 
        doc_id = [did for (_, did) in topk]
        scores = self.DPR.similarity(query, docs)
        reranked = sorted([x for x in zip(scores, doc_id)], reverse = True)
        return reranked

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index', \
                              doc_dir = 'nfcorpus/train.docs', \
                              ranker_dir = 'model2.txt', \
                              dpr_dir = 'biodpr')
    BSBI_instance.index() # memulai indexing!
