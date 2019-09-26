import numpy as np

from collections import Counter, defaultdict
from sklearn.decomposition import TruncatedSVD
from gensim.models import KeyedVectors
import pickle
import logging
import os 
###########################################################################
# LOGGING CONFIG
###########################################################################

FORMAT = '[SENTENCE EMBEDDING] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

###########################################################################
# WORD EMBEDDING LOADERS
###########################################################################

class EmbeddingModel():
    """
    A class for loading pre-trained word embeddings.
    """

    def __init__(self, path, embedding_is_custom):
        self.guess_origin_from_folder_contents(path, embedding_is_custom)
        self.load_embeddings()

    def guess_origin_from_folder_contents(self, path, embedding_is_custom):

        folder_contents = os.listdir(path)
        if len(folder_contents) == 0:
            raise Exception("Pre-trained embeddings folder is empty!")
        elif len(folder_contents) > 1:
            raise Exception("Too many files in the pre-trained embeddings folder. It should only contain one file")            
            
        embedding_file = folder_contents[0]
        self.embedding_file_path = os.path.join(path, embedding_file)
        if embedding_is_custom:
            try:
                with open(self.embedding_file_path, 'r') as f:
                    pass
            except IOError:
                raise Exception('Custom embedding should be a readable .txt file.')
            
            self.origin = "custom"
        else:
            if embedding_file == "fastText_embeddings":
                self.origin = "fasttext"
            elif embedding_file == "GloVe_embeddings":
                self.origin = "glove"
            elif embedding_file == "Word2vec_embeddings":
                self.origin = "word2vec"
            elif embedding_file == "ELMo":
                self.origin = "elmo"
            else:
                raise ValueError("Something is wrong with the pre-trained embeddings. " +
                                 "Please make sure to either use the plugin macro to download the embeddings, " +
                                 "or tick the custom embedding box if you are using custom vectors.")

    def load_embeddings(self):

        if self.origin == "word2vec":

            #######################
            # Word2vec
            #######################

            logger.info("Loading Word2vec embeddings...")
            model = KeyedVectors.load_word2vec_format(
                self.embedding_file_path, binary=True)

            self.word2idx = {w: i for i, w in enumerate(model.index2word)}
            self.embedding_matrix = model.vectors

        elif self.origin in ["custom", "glove", "fasttext"]:

            #######################################
            # GloVe, FastText or Custom Embedding
            #######################################

            word2idx = {}
            embedding_matrix = []
            with open(self.embedding_file_path, 'r') as f:
                for i, line in enumerate(f):

                    if i == 0 and self.origin == "fasttext":
                        continue

                    if i != 0 and i % 100000 == 0:
                        logger.info("Loaded {} word embeddings".format(i))

                    split = line.strip().split(' ')
                    word, vector = split[0], split[1:]

                    word2idx[word] = i

                    embedding = np.array(vector).astype(float)
                    embedding_matrix.append(embedding)

                logger.info(
                    "Successfully loaded {} word embeddings!".format(i))

            self.embedding_matrix = np.array(embedding_matrix)
            self.word2idx = word2idx


        else:
            raise Exception("Something is wrong with the embedding origin.")

    def get_sentence_word_vectors(self, batch):

        if self.origin == "elmo":
            tensors = self.elmo(
                batch, signature="default", as_dict=True)["word_emb"]
            embeddings = self.sess.run(tensors)
            return embeddings.tolist()

        else:    
            indices = [self.word2idx[w]
                       for w in batch.split() if w in self.word2idx]
            return self.embedding_matrix[indices]

    def get_weighted_sentence_word_vectors(self, batch, weights):
        #Check if sentence contains at least one token and return None if not
        indices = [self.word2idx[w]
                   for w in batch.split() if w in self.word2idx]
        embeddings = self.embedding_matrix[indices]
        weights = [weights[w] for w in batch.split() if w in self.word2idx]
        return [w * e for w, e in zip(weights, embeddings)]

      
###########################################################################
# SENTENCE EMBEDDING COMPUTATION
###########################################################################

def preprocess_and_compute_sentence_embedding(texts, embedding_model, method, smoothing_parameter, npc):
    """
    Takes a dataframe, a column name, and embedding model and an aggregation
    method then returns a sentence embeddings using the chosen method.
    """

    def average_embedding(text):
        """Get average word embedding from models like Word2vec, Glove or FastText."""
        embeddings = embedding_model.get_sentence_word_vectors(text)
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding

    def weighted_average_embedding(text, weights):
        """Weighted average embedding for computing SIF."""
        embeddings = embedding_model.get_weighted_sentence_word_vectors(text, weights)    
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding

    def remove_first_principal_component(X):
        """Removes the first PC for computing SIF."""
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        X = np.array(X)
        logger.info(X.shape)
        svd.fit(X)
        u = svd.components_
        return X - X.dot(u.T).dot(u)

    def contruct_final_res(res,is_void):
        res_final = []
        j = 0
        for v in is_void:
            if v == 0:
                res_final.append(res[j])
                j+=1
            else:
                res_final.append(np.nan)
        return res_final

    #####################################################


    # Computing either simple average or weighted average embedding
    method_name = method + "_" + embedding_model.origin

    if method == 'simple_average':
        logger.info("Computing simple average embeddings...")
        res = list(map(average_embedding, texts))


    elif method == 'SIF':

        # Compute word weights
        word_weights = Counter()
        for s in texts:
            word_weights.update(s.split())

        n_words = float(sum(word_weights.values()))
        for k, v in word_weights.items():
            word_weights[k] /= n_words
            word_weights[k] = float(smoothing_parameter) / (smoothing_parameter + word_weights[k])

        # Compute SIF
        logger.info("Computing weighted average embeddings...")
        res = list(map(lambda s: weighted_average_embedding(
            s, word_weights), texts))

        #Remove empty sentences and save their indecies
        is_void = list(map(lambda x: 1 if x.shape == () else 0 , res))
        res = [x for x,y in zip(res,is_void) if y==0]
        logger.info("Removing vectors first principal component...")
        res = remove_first_principal_component(res)
        res = contruct_final_res(res,is_void)


    else:
        raise NotImplementedError(
            "Only available aggregation methods are: 'simple_average' and 'SIF'.")

    return res