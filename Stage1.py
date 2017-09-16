__author__ = 'JBaig'

import math
import os
import numpy
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

#from modifiedtexttiling import TextTilingTokenizer
import modifiedtexttiling

#input as all the documents with preprocessed text.
corpusdir = '/home/abc/Desktop/adm/new_dataset'

#output as all the segmented documents with thier corresponding document names as prefix. 
corpusdir_p = '/home/abc/Desktop/adm/segments'
 	
newcorpus = PlaintextCorpusReader(corpusdir,'.*')

#sort all the document names alphabetically.
sortedall = sorted(newcorpus.fileids())
#print sortedall

for filename in sortedall:
    #open each document.
    fp = open(corpusdir +"/" + filename)
    #print message.
    print 'processing : '+filename
    #save document text as string.
    n = fp.read()
    
    #Create TextTilingTokenizer() object
    t = modifiedtexttiling.TextTilingTokenizer()
    
    #get the segments as list of strings.
    k = t.tokenize(n)
    #print len(k)
    #for i in range(len(k)):
        #print i
        #print k[i]
    
    #Write every segment in a seperate text file.
    #Follow the convention : 
    #filename is : Doc0001.txt
    #got 4 segments with filenames : 0001_0.txt, 0001_1.txt,0001_2.txt,0001_3.txt   
    for i in range(len(k)):
        name = filename[:-4]+'_'+str(i)+".txt"
        f = open(corpusdir_p+"/"+name, 'a')
        f.write(k[i])
        f.close()
    fp.close()
    
