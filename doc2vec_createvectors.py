from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import os
import sys
from datasets import load_dataset
from pyspark.sql import SparkSession
import numpy as np
import time

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


#data = spark.createDataFrame(data=ds['train'].select(range(1000))).select('text').toPandas()['text'].tolist()#100000
#data = spark.createDataFrame(data=ds['train']).select('text').sample(True, 0.000001, seed=0).toPandas()['text'].tolist()
#data = spark.createDataFrame(data=load_dataset("arrow", data_files=dicta)['train'].select(range(1000))).toPandas()['text'].tolist()

#ds = load_dataset("arrow", data_files=dicta)['train']
#df = spark.createDataFrame(data=ds.shuffle())
#data = df.select(range(10)).toPandas()['text']
#data = spark.createDataFrame(data=load_dataset("arrow", data_files=dicta)['train'].shuffle()).select(range(10)).toPandas()['text']
#df = spark.createDataFrame(data=load_dataset("arrow", data_files=dicta)['train'].shuffle()).select(range(10)).toPandas()['id','text']

data = spark.createDataFrame(data=load_dataset("arrow", data_files=dicta)['train'].shuffle(seed=0).select(range(100_000))).toPandas()['text']
#delete text len less than 500

t1 = time.time()

total = t1-t0
print(f'dataframe created: {total} seconds')

t0 = time.time()

#random 500_000 articles trained
model = Doc2Vec.load('model500000.mod')

t1 = time.time()

total = t1-t0
print(f'model loaded: {total} seconds')

t0 = time.time()

#function
def applythis(doc):
    return model.infer_vector(word_tokenize(doc.lower()))

document_vectors = np.array(data.apply(applythis).tolist())
np.savetxt('vectors.txt',document_vectors)

t1 = time.time()

total = t1-t0
print(f'docs vectorized!: {total} seconds')

#44-48 appellate, answer... law stuff
#55-56 astronaut, apollo 8... space stuff
#112-115 plants
#151-152 abjad/abugida writing systems
#160-170 Astatine and other quemical elements
#2 list of days of the year...

# 100_000 docs used
#dataframe created: 23.119643211364746 seconds                                   
#model loaded: 4.467645883560181 seconds
#docs vectorized!: 2030.8331294059753 seconds