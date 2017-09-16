import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from nltk.tokenize.texttiling import TextTilingTokenizer
# import texttiling1
# define vectorizer parameters

tfidf_vectorizer = TfidfVectorizer()
vectorizer = CountVectorizer()

corpusdir = '/home/abc/Desktop/adm/segment_sets'
corpusdir_p = '/home/abc/Desktop/adm/segment_sets_P'
token_dict = {}
newcorpus = PlaintextCorpusReader(corpusdir, '.*')
sortedall = sorted(newcorpus.fileids())
#print sortedall
documents = []
for filename in sortedall:
    fp = open(corpusdir + "/" + filename)
    print 'processing : ' + filename
    n = fp.read()
    # n = unicode(n, errors='ignore')
    lowers = n.lower()
    no_punctuation = lowers.translate(None, string.punctuation)
    s = unicode(no_punctuation, errors='ignore')

    ts = nltk.wordpunct_tokenize(s)
    tokens = [w.lower() for w in ts]
    ##printing tokens in all lowercase characters



    k = []
    stop = stopwords.words('english')
    k = [i for i in tokens if i not in stop]

    ## printing tokenized text after removing stop words




    porter = nltk.PorterStemmer()
    P = [porter.stem(i) for i in k if i.isalpha()]
    text = nltk.Text(P)
    # b = ' '.join(text)
    # c = b.split()
    d = ' '.join([i for i in text if not i.isdigit()])
    # print d

    documents.append(d)
#print documents
#print len(documents)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents).toarray()  # fit the vectorizer to synopses



# print tfidf_vectorizer.vocabulary_
print(tfidf_matrix.shape)
print len(tfidf_matrix)
terms1 = tfidf_vectorizer.get_feature_names()
#print terms1
Dict = {}
ind = {}
c = 0
terms1 = sorted(terms1)
for term in terms1:
    ind[term] = c
    c += 1

#print ind

for term in terms1:
    Dict[term] = 0
    t = ind[term]
    for i in range(len(tfidf_matrix)):
        if tfidf_matrix[i][t]:
            Dict[term] += 1

#print Dict
#print len(Dict)
Domain_specific_stopwords = []
for key in Dict:
    if len(key) <= 2:
        Domain_specific_stopwords.append(key)
for key in Dict:
    if Dict[key] >= len(tfidf_matrix) * 0.50:
        Domain_specific_stopwords.append(key)



print "stopjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj"
print Domain_specific_stopwords
print len(Domain_specific_stopwords)




documents_p = []
for i in documents:
    list_i = i.split()

    list1_i = [word for word in list_i if word not in Domain_specific_stopwords]
    str_i = ' '.join(list1_i)
    documents_p.append(str_i)





#print documents_p
#print "chi square from here"


#print "chi square lasts here"
wrt = 0
for filename in sortedall:
    f = open(corpusdir_p+"/"+filename, 'w')
    print 'processing : ' + filename
    f.write(documents_p[wrt])
    f.close()
    wrt += 1
