# IMPORTS NEEDED
from datasets import load_dataset
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans

nltk.download('punkt_tab')

# HYPERPARAMETERS
TRAIN_SEED = 0
TEST_SEED = 1
SIZE = 1_000
EPOCHS = 500
VECTOR_SIZE=100

# GETTING THE DATA FROM HUGGINFACE
X = load_dataset("wikimedia/wikipedia", "20231101.en")['train']
X_train = X.shuffle(seed=TRAIN_SEED).select(range(SIZE)).to_pandas()[["text","id"]]
#X_test = X.shuffle(seed=TEST_SEED).select(range(10)).to_pandas()[["text","id"]]

# CREATING TAGS FOR EACH WIKIPEDIA ARTICLE
tagged_data = [TaggedDocument(  words=word_tokenize(doc.lower()),
                                tags=[str(i)]) for i,
                                doc in enumerate(X_train["text"])]

# CREATING UMAP 
reducer = umap.UMAP()

# CREATING AND TRAINING DOC2VEC MODEL TAKING THE HYPERPARAMETERS VECTOR_SIZE AND EPOCHS
model = Doc2Vec(vector_size=VECTOR_SIZE,
                min_count=2, epochs=EPOCHS)
model.build_vocab(tagged_data)
model.train(tagged_data,
            total_examples=model.corpus_count,
            epochs=model.epochs)

# FUNCTION TO BE APPLIED TO EVERY DOC TO CONVERT THEM TO A VECTOR
def applythis(doc):
    return model.infer_vector(word_tokenize(doc.lower()))

# CREATE COLUMN NAMES
columns = []
for x in range(1, VECTOR_SIZE+1):
    columns.append(f"v{x}")

# JOIN TRAINING DATA (TEXT AND ID) WITH THE VECTORS
X_train = pd.concat([X_train, pd.DataFrame(X_train["text"].apply(applythis).tolist(), columns=columns)], axis=1)

# TAKE THE VECTORS AND USE DIMENSIONALITY REDUCER (UMAP)
embedding = reducer.fit_transform(X_train[columns])

# LOOP FOR TESTING 1 TO 15 CLUSTERING (KMEANS)
inertias = []
for i in range(1,150):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_train[columns])
    inertias.append(kmeans.inertia_)


# PLOT ELBOW METHOD
plt.plot(range(1,150), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#X_test = pd.concat([X_test, pd.DataFrame(X_test["text"].apply(applythis).tolist(), columns=columns)], axis=1)
