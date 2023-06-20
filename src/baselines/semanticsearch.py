from sentence_transformers import SentenceTransformer, util
import pickle
import argparse
model = SentenceTransformer('all-MiniLM-L6-v2', device = 'cuda')


#Load sentences & embeddings from disc
with open('embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_words = stored_data['words']
    stored_definitions = stored_data['definitions']
    stored_embeddings = stored_data['embeddings']

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--query', type=str, required=False)
# Parse the argument
args = parser.parse_args()

if args.query is None:
    query = ['man in space']
else:
    query = args.query
query_embedding = model.encode(query, convert_to_tensor=True)
print(query_embedding)
corpus_embeddings = stored_embeddings.to('cuda')
#corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

query_embeddings = query_embedding.to('cuda')
print(query_embeddings)
#query_embeddings = util.normalize_embeddings(query_embeddings)
hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=10)

#hits = hits[0] #Get the hits for the first query
for hitt in hits:
    for hit in hitt:
        print(stored_words[hit['corpus_id']], ":", stored_definitions[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))


