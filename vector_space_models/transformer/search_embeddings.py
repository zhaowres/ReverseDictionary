from sentence_transformers import SentenceTransformer, util
import pickle
import argparse
import json
import csv
import numpy as np
import torch

# other models: all-mpnet-base-v2 multi-qa-mpnet-base-dot-v1  all-distilroberta-v1 all-MiniLM-L12-v2 multi-qa-distilbert-cos-v1
model_name = 'multi-qa-mpnet-base-dot-v1'
model = SentenceTransformer(model_name)

def sbert(inputs):
    #Load sentences & embeddings from disc
    with open(f'../../transformer_embeddings/{model_name}.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_words = stored_data['words']
        stored_definitions = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']

    # # Create the parser
    # parser = argparse.ArgumentParser()
    # # Add an argument
    # parser.add_argument('--query', type=str, required=False)
    # # Parse the argument
    # args = parser.parse_args()

    # if args.query is None:
    #     query = ['man in space']
    # else:
    #     query = args.query


    predictions = []
    for query in inputs:
        query_embedding = model.encode(query, convert_to_tensor=True)
        corpus_embeddings = stored_embeddings.to('cuda')
        #corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

        query_embeddings = query_embedding.to('cuda')
        #query_embeddings = util.normalize_embeddings(query_embeddings)

   
        # We use cosine-similarity/dot product similary and torch.topk to find the highest 500 scores
        cos_scores = util.dot_score(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=500)

        single_prediction = []
        for score, idx in zip(top_results[0], top_results[1]):
            # print(stored_words[idx], "(Score: {:.4f})".format(score))
            if (stored_words[idx] not in single_prediction):
                single_prediction.append(stored_words[idx])

        predictions.append(single_prediction)
        # hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=10)

        # #hits = hits[0] #Get the hits for the first query
        # single_prediction = [] 
        # for hitt in hits:
        #     for hit in hitt:
        #         print(stored_words[hit['corpus_id']], ":", stored_definitions[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))

        # predictions.append(stored_words[hit['corpus_id']])

    
    return predictions
        
print(sbert(["source of light"]))

test_set_paths = ["../../data/data_test_500_rand1_seen.json", "../../data/data_test_500_rand1_unseen.json", "../../data/data_desc_c.json"]
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

def evaluate():
   

    test_sets = ["seen", "unseen", "description"]
    for i,test_set in enumerate(test_set_paths):
    
        inputs = []
        words = []
        with open(test_set) as f:
            data = json.load(f)
            for point in data:
                inputs.append(point['definitions'])
                words.append(point['word'])
            
        predictions = sbert(inputs)
        with open(f'../../results/vector_space/{model_name}_dot_{test_sets[i]}_results_.csv', 'w') as results:
            writer = csv.writer(results)
            writer.writerow(['Description', 'Solution', 'Prediction rank', 'Predictions'])

            total = len(words)
            correct = 0
            rank = [0] * total
            for i, value in enumerate(predictions):
                for j, value2 in enumerate(value):
                    if (value2 == words[i]):
                        correct += 1
                        rank[i] = j+1


                writer.writerow([inputs[i], words[i], rank[i], predictions[i]])
            
            writer.writerow(evaluate_test(words, predictions))

evaluate()

