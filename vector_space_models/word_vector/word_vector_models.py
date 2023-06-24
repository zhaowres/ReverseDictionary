import os
os.environ['GENSIM_DATA_DIR'] = '/vol/bitbucket/wz1620/gensim-data'

import gensim.downloader
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import json
import csv
import numpy as np

""""
models:    
    ['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 
    'word2vec-ruscorpora-300', 'word2vec-google-news-300', 
    'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100',
    'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 
    'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 
    'glove-twitter-200', '__testing_word2vec-matrix-synopsis']

"""
nltk.download('stopwords',download_dir='/vol/bitbucket/wz1620/nltk_data')

model_name = 'fasttext-wiki-news-subwords-300'

def baseline(inputs):
    wv = gensim.downloader.load(model_name)
    # wv = gensim.downloader.load('word2vec-google-news-300')

    vocab = list(wv.index_to_key)

    # create a filter for dictionary words (not names or places)
    dict_words = []
    f = open("../../data/dictionary_words.txt", "r")
    for line in f:
        dict_words.append(line.strip())    
    f.close()    
                                          

    def find_words(input):    
        positive_words = preprocess(input)
        predicted_words = [i[0] for i in wv.most_similar(positive=positive_words, topn=1000)]  
        words = []    

        for word in predicted_words:
             if (word in dict_words):
                 words.append(word)
        if (len(words) > 100):
             words = words[0:100]

        return words

    def preprocess(input):         

        #set stop words
        stop_words = set(stopwords.words('english'))

        #tokenize the input words
        word_tokens = word_tokenize(input)
       
        #filter for stop words
        filtered_sentence = []
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)

        for i in range(len(filtered_sentence) - 1, -1, -1):
            if filtered_sentence[i] not in vocab:                    
                del filtered_sentence[i]       

        # replace n-grams with phrases
        possible_expressions = []
        for w in [filtered_sentence[i:i+3] for i in range(len(filtered_sentence)-3+1)]:        
            possible_expressions.append('_'.join(w))            

        ex_to_remove = []

        for i in range(len(possible_expressions)):        
            if possible_expressions[i] in vocab:                    
                ex_to_remove.append(i)        

        words_to_remove = []    
        for i in ex_to_remove:
            words_to_remove += [i, i+1, i+2]        
        words_to_remove = sorted(set(words_to_remove))    

        words = [possible_expressions[i] for i in ex_to_remove]    
        for i in range(len(filtered_sentence)):
            if i not in words_to_remove:
                words.append(filtered_sentence[i])    

        return words

    predictions = []
    for defi in inputs:
        predictions.append(find_words(defi))

    return predictions

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
            
        predictions = baseline(inputs)
        with open(f'../../results/vector_space/{model_name}_{test_sets[i]}_results_.csv', 'w') as results:
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