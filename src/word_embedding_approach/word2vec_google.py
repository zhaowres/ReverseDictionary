import gensim.downloader

""""
models:    
    ['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 
    'word2vec-ruscorpora-300', 'word2vec-google-news-300', 
    'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100',
    'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 
    'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 
    'glove-twitter-200', '__testing_word2vec-matrix-synopsis']

"""

def word2vec_google(definitions):
    wv = gensim.downloader.load('word2vec-google-news-300')

    vocab = list(wv.index_to_key)

    # read dictionary words
    dict_words = []
    f = open("./words.txt", "r")
    for line in f:
        dict_words.append(line.strip())    
    f.close()    
                                          
    dict_words = dict_words[44:]

    def find_words(definition):    
        positive_words = determine_words(definition)
        similar_words = [i[0] for i in wv.most_similar(positive=positive_words, topn=100)]  
        words = []    

        for word in similar_words:
             if (word in dict_words):
                 words.append(word)
        if (len(words) > 10):
             words = words[0:10]

        return similar_words

    def determine_words(definition):         
        possible_words = definition.split()
        for i in range(len(possible_words) - 1, -1, -1):
            if possible_words[i] not in vocab:                    
                del possible_words[i]          

        possible_expressions = []
        for w in [possible_words[i:i+3] for i in range(len(possible_words)-3+1)]:        
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
        for i in range(len(possible_words)):
            if i not in words_to_remove:
                words.append(possible_words[i])    

        return words

    predictions = []
    for defi in definitions:
        predictions.append(find_words(defi))

    return predictions

print(word2vec_google(["occupy a place"]))
print(word2vec_google(["fit to be eaten"]))
print(word2vec_google(["inspector of buildings"]))
