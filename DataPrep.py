import collections
from nltk.tokenize import word_tokenize
import re
import pickle

def loadData():
    textFile = 'data/practice.txt'
    vocab_size = 25000
    raw_text = open(textFile, encoding='utf-8').read()
    print('loaded')
    text = [w.lower() for w in word_tokenize(raw_text)]
    full_text = ['<n>' if re.findall(r'\d+',w) else w for w in text]
    print('tokenized')
    count = collections.Counter(full_text).most_common(vocab_size)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    processed_text = [w for w in full_text if w in dictionary.keys()]
    return dictionary, reverse_dictionary, processed_text

def writeData(d, rd, text):
    with open('data/dictionary', 'wb') as df:
        df.write(pickle.dumps(d))
        df.close()
    with open('data/reverse-dictionary', 'wb') as rdf:
        rdf.write(pickle.dumps(rd))
        rdf.close()
    with open('data/text', 'wb') as tf:
        tf.write(pickle.dumps(text))
        tf.close()
ld = loadData()
writeData(ld[0],ld[1], ld[2])
