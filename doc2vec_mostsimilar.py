
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import os
import sys
from datasets import load_dataset
from pyspark.sql import SparkSession
import numpy as np
import time

v1 = [-0.02151850424706935883, 0.5600769519805908203, -0.05306205525994300842, 0.7027760148048400879, -1.311826705932617188, -1.611707806587219238, 0.6573483943939208984, -0.2949672043323516846, -1.112485051155090332, 0.9939249157905578613, 1.682537436485290527, -0.3464692234992980957, 0.7636488676071166992, 0.3207920789718627930, -0.8538008332252502441, 1.456900358200073242, -0.2956738770008087158, -1.448442816734313965, -1.210573315620422363, -1.076992273330688477]


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

data = spark.createDataFrame(data=load_dataset("arrow", data_files=dicta)['train'].shuffle(seed=0).select(range(438458))).toPandas()['text']

model = Doc2Vec.load("model500000.mod")
q = model.infer_vector(data[35022].split())
similar_doc = model.dv.most_similar([q],topn=10)
for i in similar_doc:
    print(data[int(i[0])])
    print('¡=============================================================================!')