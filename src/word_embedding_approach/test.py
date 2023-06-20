import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

print(list(api.info()['models'].keys()))