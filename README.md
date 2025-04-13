# Files

## Official files in github:
requirements.txt:

pip install requirements for the code to work.

wikipediacode.py:

Final code that takes the wikipedia articles and cluster them. 
using 1_000 random wikipedia articles, trains doc2vec model after tagging the data, converts 1_000 random test data to a vector using the model and plots using UMAP and KMeans.

## Results and figures:

The elbow method is done so we can see that the drop is steady after k equals 10, so we choose 9.
![Figure_elbow50](https://github.com/user-attachments/assets/e15e48c4-64e7-4e3d-9ba6-0fe877d24d46)


![Figure_elbow10](https://github.com/user-attachments/assets/25e64076-6099-4cd1-89de-08de95e5365c)

### Meaning of the Plots

if the articles are close and they have the same color, there's a high chance that tehy are related. These are some examples of articles being close and similar:

This plot shows villages from Iran, such as Pir Kohan, 
Balgeh Shir, Shir Daneh, 
Emamzadeh Ebrahim, Gilan, 
Dar Balut-e Olya,
Mian Chenar-e Mazkur, 
Garmejan. This plot is an example of close articles because their are short and are the same concept, villages:

![Figure_iran](https://github.com/user-attachments/assets/10e58edf-482b-4d8a-8c5e-2835ed94f120)

This image plots sports in general. Games, people and seasons. This plot shows that with short training time, we can group articles, but we can do it better by grouping more, for instance clustering the type of sport(as it shows baseball, basketball, football and hockey) or clustering the people playing the same sport.

Some players are: Anders Gozzi(hockey), 
Tim Layana(baseball) and Harry Neale(hockey).
Some teams/leagues are: 2007 Major League Baseball Home Run Derby, 1895 Kentucky State College Blue and White football team and 
2018–19 Milwaukee Panthers men's basketball team.

![Figure_football](https://github.com/user-attachments/assets/8a622e31-f0a9-44b4-9ff3-c160f746edfd)

This plot means that even articles that are dificult to classify and group because of their complexity, such as animal species and craters, the model is able to group them. Some animals are: Caddisfly, Cermatulus nasalis, 
Northern nail-tail wallaby and Tunga penetrans. Craters also are shown, probably because they are similarly described as animals. The crater are:
Klein, 
Garavito and 
Sheepshanks.
![Figure_nature](https://github.com/user-attachments/assets/7968f6eb-e1be-417b-894e-031494140035)

Engineered stuff are clustered in this plot(blue), but also show orange and grey points, showing that even though they phisically are close in the umap plot, kmeans achieves to represent the missing distance. This means that we can identify articles who do not belong to the group, such as ECPA, Transphobia in the United States and project on emerging nanotechnologies:
![Figure_engineer](https://github.com/user-attachments/assets/d9806aa1-f6df-45ba-b32e-cbdeab0a5b75)

This is a more complex plot to read, but thanks to kmeans, we can see that the blue navy dots are places around the world, clustered probably because they are described similarly:
![Figure_places_world](https://github.com/user-attachments/assets/08113aca-6c45-430c-8b75-d40bd10c4be6)

This plot easily shows songs and artists that are described similarly
![Figure_songs](https://github.com/user-attachments/assets/f71f707b-44a7-4763-8363-5aee8774316f)

This is a 3D plot of the UMAP using kmeans and 1_000. This plot tell us that as the wikipedia articles increase, the density will increase, meaning that the plot will become unatractive and another way to judge the model should be done. For example, checking a number of articles and seeing their neighbours using the vector of length 100.
![Figure_3d](https://github.com/user-attachments/assets/30f5d5ea-9a58-45b8-9439-fe49f77ba906)

# WIKIPEDIA
Dataset: https://huggingface.co/datasets/wikimedia/wikipedia

## Objective: 

The objectives of this project are:

Learn and dos:

- **Doc2Vec** to vector Wikipedia articles with a personalized trained model.

- **CLUSTER** articles while handling big data via *UMAP*, as *PCA* is linear and t-sne takes too long.

- **OPTIMIZE** *Vector_size*, meaning how long the vector should be that represents enough of the article and *epoch*, meaning many times do the model have to see the data to learn

- **PREDICT** vectors with shallow model.

- **SAVE** model to reduce time.

- **LOAD** model to predict new vectors.

To get to a point where we can:

- **CREATE** *Recommendation System* for articles using user database
- **LABEL** articles using unsupervised clustering
- **CREATE** good enough model that can take any article and converts to a vector that is a good representation of it

### Regarding Extra Steps:

This are some steps that can be useful to apply to the project, but are time consuming:

-   **TRAIN** more data.

-   **ADD** *sentiment analyisis*, *length of the articles*, *word frequency*...

-   **CHECK** *King – Man + Woman = Queen* works

###  Conclusion:

We prove that our model can work even with training as low as 10 articles and as high as 1_000. Testing 1_000 articles gave enough data to conclude that the clustering is working and, with enough training and testing data, we can imagine that the clustering will be almost flawless. 
