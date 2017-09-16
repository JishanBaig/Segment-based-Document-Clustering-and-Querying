from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import os
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import numpy as np
import operator
import os
import sys  
#corpus of segments
reload(sys)  
sys.setdefaultencoding('Cp1252')
corpusdir = '/home/abc/Desktop/adm/segments'
newcorpus = PlaintextCorpusReader(corpusdir,'.*')
sortedall = sorted(newcorpus.fileids())
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

doc_a = "/home/rashmi/Documents/adm_project/appended/0.txt"
doc_b = "/home/rashmi/Documents/adm_project/appended/1.txt"
doc_c = "/home/rashmi/Documents/adm_project/appended/2.txt"
doc_d = "/home/rashmi/Documents/adm_project/appended/3.txt"
doc_e = "/home/rashmi/Documents/adm_project/appended/4.txt" 

# compile sample documents into a list
doc_set = newcorpus.fileids()
doc_set = sorted(newcorpus.fileids())

# list for tokenized documents in loop
texts = []
t_d = 0
val=1000
try:
# loop through document list
	while t_d<len(doc_set):
		pointer = 'True'
		empty = []
	
		while pointer == 'True' and t_d<len(doc_set):
			print "inside"
			filename = doc_set[t_d]
			print filename + "filename"
			filename_sub = filename.split('_')
			if filename_sub[0] == str(val):
				print t_d
				t_d = t_d + 1
				empty.append("/home/abc/Desktop/adm/segments/" + filename)
			else:
			     pointer = 'false'
			     print "fail"
		texts = []
		print "receved all file"
		print empty
		print len(doc_set)
		for i in empty:
			#i = "/home/rashmi/Documents/adm_project/trial/"+i
			raw = open(i).read()
			raw = raw.lower()
			tokens = tokenizer.tokenize(raw)

		    # remove stop words from tokens
			stopped_tokens = [i for i in tokens if not i in en_stop]
		    
		    # stem tokens
			stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
		    
		    # add tokens to list
			texts.append(stemmed_tokens)

		#print i
		# turn our tokenized documents into a id <-> term dictionary
		dictionary = corpora.Dictionary(texts)
		#print(dictionary.token2id)
		corpus = [dictionary.doc2bow(text) for text in texts]
		# generate LDA model
		ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=8, id2word = dictionary, passes=20)
		#for top in ldamodel.print_topics():
		  #print top
		#declaring number of topics
		num_topics = 8
		arr = [[] for _ in range(num_topics)]
		# Assigns the topics to the documents in corpus
		counter=0
		print "Elements in array"
		print empty		
		while counter < len(empty):
			lda_corpus = ldamodel[corpus[counter]]
			num_topics1=len(lda_corpus)
			i=0
			print "Corpussssssssss Containsss"
			print lda_corpus
			topics = []
			while i < num_topics1:
				topics.append(lda_corpus[i][1])
				#print lda_corpus[i][1]
				i=i+1
			index, value = max(enumerate(topics), key=operator.itemgetter(1))
			#print ldamodel.top_topics(corpus[4],2)
			indexf = lda_corpus[index][0]
			print index
			print value
			topics[index] = 0
			flag=0	
			if len(lda_corpus) > 1: 
				index1, value1 = max(enumerate(topics), key=operator.itemgetter(1))
				indexff = lda_corpus[index1][0]
				print index1
				print value1
				flag=1
			if flag==1:
				 arr[int(indexff)].append(empty[counter])
				 flag=0
			arr[int(indexf)].append(empty[counter])
			counter=counter+1
		print "+++++++++++++++++++++++++++++++++++++++++++++++++"	
		print arr
		#print len(arr[0])
		counter=0
		#val = 1000
		while counter < len(arr):
			if len(arr[counter])>0:
				corpusdir_p = '/home/abc/Desktop/adm/segment_sets/'
				nameoffile = str(val)+ "_" +str(counter) + ".txt"
				newfile = open(corpusdir_p + nameoffile ,'w')
				innerc=0
				strfile=""
				while innerc < len(arr[counter]):
					#strfile = strfile + arr[counter][innerc] + "\n"
					os.system("cat "  +arr[counter][innerc] + " >> "+corpusdir_p + nameoffile)
					#os.system("echo \ >> " + corpusdir_p + nameoffile)
					innerc=innerc+1	
				#newfile.writelines(strfile)
				#newfile.write(" "+"\n")
				#newfile.write("{}\n".format("s"))
				newfile.close()
			
			counter=counter+1
			#print strfile
		val=val+1
		


		#print newcorpus.fileids()
except:
	print str(val) + "errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"
