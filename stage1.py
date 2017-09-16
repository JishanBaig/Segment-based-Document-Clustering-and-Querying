__author__ = 'jishan baig'
import math
import os
import numpy
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

#from texttiling import TextTilingTokenizer
import modifiedtexttiling

corpusdir = '/home/abc/Desktop/adm/new_dataset'
corpusdir_p = '/home/abc/Desktop/adm/segments'
 	
newcorpus = PlaintextCorpusReader(corpusdir,'.*')
sortedall = sorted(newcorpus.fileids())
#print sortedall

for filename in sortedall:
    fp = open(corpusdir +"/" + filename)
    print 'processing : '+filename
    n = fp.read()
                                                                       ## printing whole file as a one text
    t = texttiling.TextTilingTokenizer()

    k = t.tokenize(n)
    #print len(k)
    #for i in range(len(k)):
        #print i
        #print k[i]

    for i in range(len(k)):
        name = filename[:-4]+'_'+str(i)+".txt"
        f = open(corpusdir_p+"/"+name, 'a')
        f.write(k[i])
        f.close()
    fp.close()
