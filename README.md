# WIKIPEDIA
Sentiment analysis on Wikipedia

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

## Results and figures:

This is the correlation matrix of 1_000 randomly selected vectors that each represent a single article, without any filter in length of document:
![Figure_1000](https://github.com/user-attachments/assets/94a6b030-4fcb-468f-a017-7c4a545c67e7)

This is 10_000 vectors:
![Figure_10000](https://github.com/user-attachments/assets/2685033e-97f7-4861-afb0-79a4c6400391)

tsne with 100 vectors:
![Figure tsne100](https://github.com/user-attachments/assets/cc4b379c-9c0f-4fa8-a614-a04ee69400b2)

### Regarding Extra Steps:

This are some steps that can be useful to apply to the project, but are time consuming:

-   **TRAIN** replaced data.

-   **ADD** *sentiment analyisis*, *length of the articles*, *word frequency*...

-   **NORMAL** text, running -> run



tsne with 1000 vectors:
![Figure tsne1000](https://github.com/user-attachments/assets/8a2eaa36-dd5f-4140-a04e-ef019cadea41)


