from sentence_transformers import SentenceTransformer, util
import pickle
import argparse
import json
import numpy as np
model = SentenceTransformer('all-MiniLM-L6-v2', device = 'cuda')

def evaluate(ground_truth, prediction):
    accu_1 = 0.
    accu_10 = 0.
    accu_100 = 0.
    length = len(ground_truth)
    for i in range(length):
        if ground_truth[i] in prediction[i][:100]:
            accu_100 += 1
            if ground_truth[i] in prediction[i][:10]:
                accu_10 += 1
                if ground_truth[i] == prediction[i][0]:
                    accu_1 += 1
    return accu_1/length*100, accu_10/length*100, accu_100/length*100

def evaluate_test(ground_truth, prediction):
    accu_1 = 0.
    accu_10 = 0.
    accu_100 = 0.
    length = len(ground_truth)
    pred_rank = []
    for i in range(length):
        try:
            pred_rank.append(prediction[i][:].index(ground_truth[i]))
        except:
            pred_rank.append(1000)
        if ground_truth[i] in prediction[i][:100]:
            accu_100 += 1
            if ground_truth[i] in prediction[i][:10]:
                accu_10 += 1
                if ground_truth[i] == prediction[i][0]:
                    accu_1 += 1
    return accu_1/length*100, accu_10/length*100, accu_100/length*100, np.median(pred_rank), np.sqrt(np.var(pred_rank))

#Load sentences & embeddings from disc
with open('embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_words = stored_data['words']
    stored_definitions = stored_data['definitions']
    stored_embeddings = stored_data['embeddings']

# Opening JSON file
# f = open('data_desc_c.json')
#f = open('data_test_500_rand1_seen.json')
f = open('data_test_500_rand1_unseen.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)

words = []
definitions = []
for i, value in enumerate(data):
    words.append(value["word"])
    definitions.append(value["definitions"])

queries = definitions

corpus_embeddings = stored_embeddings.to('cuda')
corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

predictions = []
for i, query in enumerate(queries):
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.to('cuda')
    query_embedding = util.normalize_embeddings(query_embedding)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=100, score_function=util.dot_score)
    hits = hits[0]
    top100 = []
    print("-----")
    print("query:", query, "actual_word:", words[i])
    for hit in hits[:10]:
        print(stored_words[hit['corpus_id']], ":", stored_definitions[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
        top100.append(stored_words[hit['corpus_id']])
    predictions.append(top100)

test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_test(words, predictions)
print('test_accu(1/10/100): %.2f %.2F %.2f %.2f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))