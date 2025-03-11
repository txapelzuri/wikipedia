# WIKIPEDIA
Sentiment analysis on Wikipedia

Dataset: https://huggingface.co/datasets/wikimedia/wikipedia

## Objective: 

The ojectives of this project, other than learning the management of big data via Spark:

- **CLUSTER** articles (*Brown clustering*,*t-SNE*?)
  
- **Doc2Vec** to vector Wikipedia articles.

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

## Results and figures:

This is the correlation matrix of 1_000 randomly selected vectors that each represent a single article, without any filter in length of document:
![Figure_1000](https://github.com/user-attachments/assets/94a6b030-4fcb-468f-a017-7c4a545c67e7)

This is 10_000 vectors:
![Figure_10000](https://github.com/user-attachments/assets/2685033e-97f7-4861-afb0-79a4c6400391)
