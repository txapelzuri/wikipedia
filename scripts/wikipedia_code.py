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
TRAIN_SIZE = 1000
TEST_SIZE = 1000
EPOCHS = 500
VECTOR_SIZE=100

# GETTING THE DATA FROM HUGGINGFACE
X = load_dataset("wikimedia/wikipedia", "20231101.en")['train']
X_train = X.shuffle(seed=TRAIN_SEED).select(range(TRAIN_SIZE)).to_pandas()[["text","id","title"]]
X_test = X.shuffle(seed=TEST_SEED).select(range(TEST_SIZE)).to_pandas()[["text","id","title"]]

# CREATING TAGS FOR EACH WIKIPEDIA ARTICLE
tagged_data = [TaggedDocument(  words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(X_train["text"])]

# CREATING UMAP WITH 2 AND 3 COMPONENTS
reducer2 = umap.UMAP(n_components=2)
reducer3 = umap.UMAP(n_components=3)

# CREATING AND TRAINING DOC2VEC MODEL TAKING THE HYPERPARAMETERS VECTOR_SIZE AND EPOCHS
model = Doc2Vec(vector_size=VECTOR_SIZE, min_count=2, epochs=EPOCHS)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# FUNCTION TO BE APPLIED TO EVERY DOC TO CONVERT THEM TO A VECTOR
def applythis(doc):
    return model.infer_vector(word_tokenize(doc.lower()))

# CREATE COLUMN NAMES(columns = [v1,v2...])
columns = []
for x in range(1, VECTOR_SIZE+1):
    columns.append(f"v{x}")

# JOIN TEST DATA (TEXT AND ID) WITH THE VECTORS
X_test = pd.concat([X_test, pd.DataFrame(X_test["text"].apply(applythis).tolist(), columns=columns)], axis=1)

# TAKE THE VECTORS AND USE DIMENSIONALITY REDUCER (UMAP)
embedding2 = reducer2.fit_transform(X_test[columns])
embedding3 = reducer3.fit_transform(X_test[columns])

inertias = []
for i in range(1,50):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_test[columns])
    inertias.append(kmeans.inertia_)

# PLOT ELBOW METHOD
plt.plot(range(1,50), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# USING KMEANS WITH 9 GROUPS
kmeans = KMeans(n_clusters=9, n_init="auto")
X_kmeans = kmeans.fit(X_test[columns])

# PLOT DATA WITH CENTROIDS
fig, ax = plt.subplots()
ax.scatter(embedding2[:,0], embedding2[:,1], c=X_kmeans.labels_, cmap='nipy_spectral')
for i, txt in enumerate(X_test["title"]):
    ax.annotate(txt, (embedding2[i,0], embedding2[i,1]))
plt.show()

fig = plt.figure(figsize= (10,10))
ax = fig.add_subplot(projection = '3d')
ax.scatter(embedding3[:,0], embedding3[:,1], embedding3[:,2], c = X_kmeans.labels_, cmap='nipy_spectral')

x_center = (embedding3[:,0].max() + embedding3[:,0].min())/2
y_center = (embedding3[:,1].max() + embedding3[:,1].min())/2
z_center = (embedding3[:,2].max() + embedding3[:,2].min())/2
ax.plot([x_center,x_center],[y_center,y_center],[embedding3[:,2].min() - 2, embedding3[:,2].max() + 2],c = 'k',lw = 5)
ax.plot([x_center,x_center],[embedding3[:,1].min() - 2, embedding3[:,1].max() + 2],[z_center,z_center],c = 'k',lw = 5)
ax.plot([embedding3[:,0].min() - 2, embedding3[:,0].max() + 2],[y_center,y_center],[z_center,z_center],c = 'k',lw = 5)
ax.axis('off')
plt.show()
