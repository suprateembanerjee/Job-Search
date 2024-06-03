# Author: Suprateem Banerjee [www.github.com/suprateembanerjee]

from weaviate.util import generate_uuid5
import weaviate.classes as wvc
import json

# Loads data into a Weaviate collection from a JSON source
def load_data(collection,
			  path:str='../data/jobs.json',
			  num_docs:int=-1):

	with open(path) as f:
	    jobs = json.load(f)

	if num_docs != -1:
		try:
			jobs=jobs[:num_docs]
		except:
			pass

	with collection.batch.dynamic() as batch:
	    for job in jobs:
	        batch.add_object(
	        	properties=job, 
	        	uuid=generate_uuid5(job))

# Creates a Weaviate collection according to specification
def create_collection(client, 
					  collection_name:str='Jobs_Subset', 
					  collection_desc:str='various job postings',
					  vec_endpoint:str='http://host.docker.internal:11434',
					  vec_model:str='nomic-embed-text',
					  gen_endpoint:str='http://host.docker.internal:11434',
					  gen_model:str='llama3'):

	if client.collections.exists(collection_name):
	    print('Dropping pre-exisiting collection')
	    client.collections.delete(collection_name)
	    
	return client.collections.create(
		name=collection_name,
		description=collection_desc,

		vectorizer_config=[
		wvc.config.Configure.NamedVectors.text2vec_ollama(
			api_endpoint=vec_endpoint,
			model=vec_model,
			vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
				quantizer=wvc.config.Configure.VectorIndex.Quantizer.bq(),
				distance_metric=wvc.config.VectorDistances.L2_SQUARED),
			name='description_vector', 
			source_properties=['description', 'skills', 'industry', 'title'], 
			vectorize_collection_name=False)
		],

		generative_config=wvc.config.Configure.Generative.ollama(
			api_endpoint=gen_endpoint, 
			model=gen_model),

		reranker_config=wvc.config.Configure.Reranker.cohere(model='rerank-english-v3.0'),

		properties=[
		wvc.config.Property(
			name='job_id', 
			data_type=wvc.config.DataType.TEXT),
		wvc.config.Property(
			name='title', 
			data_type=wvc.config.DataType.TEXT),
		wvc.config.Property(
			name='description', 
			data_type=wvc.config.DataType.TEXT, 
			tokenization=wvc.config.Tokenization.WHITESPACE),
		wvc.config.Property(
			name='location', 
			data_type=wvc.config.DataType.TEXT),
		wvc.config.Property(
			name='type', 
			data_type=wvc.config.DataType.TEXT),
		wvc.config.Property(
			name='remote', 
			data_type=wvc.config.DataType.BOOL),
		wvc.config.Property(
			name='skills', 
			data_type=wvc.config.DataType.TEXT),
		wvc.config.Property(
			name='industry', 
			data_type=wvc.config.DataType.TEXT)])