# Files
## Temporal Files:
doc2vec_create_model.py:
Takes wikipedia articles, trains and saves the model.

doc2vec_create_vectors.py:
Using the model, creates vectors to each wikipedia article.

doc2vec_mostsimilar.py:
In the list of vectors, get the most similar articles.

doc2vec_vizz.py:
Visualizes the data using tsne and kmeans

## Official files
requirements.txt:
pip install requirements for the code to work.

wikipediacode.py:
Final code that takes the wikipedia articles and cluster them. 
Using 1_000 random wikipedia articles, trains doc2vec model after tagging the data, converts 1_000 random test data to a vector using the model and plots using UMAP and KMeans.

## Results and figures:


![Figure tsne1000](https://github.com/user-attachments/assets/8a2eaa36-dd5f-4140-a04e-ef019cadea41)
tsne with 1000 vectors:

![Figure kmeans100_000](https://github.com/user-attachments/assets/ef8515cb-3c7f-4ff7-aaf7-e94351c3c298)
kmeans with 100k vectors:

# WIKIPEDIA
Dataset: https://huggingface.co/datasets/wikimedia/wikipedia

## Objective: 

The ojectives of this project, other than learning the management of big data via Spark:

- **Doc2Vec** to vector Wikipedia articles.

- **CLUSTER** articles via *t-SNE*.  

- **CHANGE** *Vector_size* and *epoch* optimizing time and space.

- **PREDICT** vectors with shallow model.

- **SAVE** model to reduce time.

- **LOAD** model to predict new vectors.

- **CREATE** *Recommendation System* for articles.


### Regarding Extra Steps:

This are some steps that can be useful to apply to the project, but are time consuming:

-   **TRAIN** replaced data.

-   **ADD** *sentiment analyisis*, *length of the articles*, *word frequency*...

-   **NORMAL** text, running -> run

