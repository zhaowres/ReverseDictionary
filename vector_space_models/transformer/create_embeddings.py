from sentence_transformers import SentenceTransformer, util
import torch
import json
import pickle
import re 

# other models: all-mpnet-base-v2 multi-qa-mpnet-base-dot-v1  all-distilroberta-v1 all-MiniLM-L12-v2 multi-qa-distilbert-cos-v1
model_name = 'all-mpnet-base-v2'
embedder = SentenceTransformer(model_name)


f = open('../../data/data_full.json')
data = json.load(f)

sentences = []
words = []
for point in data:
    # Remove bracketed defintions
    sentences.append(re.sub("[\(\[].*?[\)\]]", "",point['definitions']))
    words.append(point['word'])

embeddings = embedder.encode(sentences, convert_to_tensor=True)
embeddings = embeddings.to('cuda')
embeddings = util.normalize_embeddings(embeddings)

#Store sentences & embeddings on disc
with open(f'../../transformer_embeddings/{model_name}_embeddings.pkl', "wb") as fOut:
    pickle.dump({'sentences': sentences, 'embeddings': embeddings, 'words': words}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

f.close()
