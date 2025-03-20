from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import os
import sys
from datasets import load_dataset
from pyspark.sql import SparkSession
import nltk
import time



nltk.download('vader_lexicon')
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

dicta = []
for i in range(41):
    if i < 10:
        dicta.append(f'train/data-0000{i}-of-00041.arrow')
    else:
        dicta.append(f'train/data-000{i}-of-00041.arrow')

if not 'spark' in locals():
    spark = SparkSession.builder.master("local[*]").config("spark.driver.memory","64G").getOrCreate()


t0 = time.time()

data = spark.createDataFrame(data=load_dataset("arrow", data_files=dicta)['train'].shuffle().select(range(500_000))).toPandas()['text'].tolist()#1_000_000 error, maybe should have REPLACEMENT=TRUE WHEN SHUFFLING/SELECTING

t1 = time.time()

total = t1-t0
print(f'dataframe created: {total} seconds')

t0 = time.time()

# preproces the documents, and create TaggedDocuments
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                              tags=[str(i)]) for i,
               doc in enumerate(data)]

t1 = time.time()

total = t1-t0
print(f'dataframe tagged: {total} seconds')

t0 = time.time()

# train the Doc2vec model
model = Doc2Vec(vector_size=20,
                min_count=2, epochs=50)
model.build_vocab(tagged_data)
model.train(tagged_data,
            total_examples=model.corpus_count,
            epochs=model.epochs)

#random 500000 articles trained
model.save('model500000.mod')

t1 = time.time()

total = t1-t0
print(f'model created: {total} seconds')

#100_000 docs used
#dataframe created: 16s
#dataframe tagged: 90s
#model created: 1091s
#total: 1197s ~ 20 minutes

#500_000 docs used
#dataframe created: 66s
#dataframe tagged 465s
#model created: 5875s
#total: 6406s ~ 106 minutes ~ 1.77 hours
