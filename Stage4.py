from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import os
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import numpy as np
import operator
import re
import subprocess
from PyDictionary import PyDictionary
from collections import OrderedDict
import unicodedata
import xml.etree.ElementTree as ET
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer
import copy
import itertools
import nltk
from tabulate import tabulate
import sys
#Inverted index code building full text index


#removing all the files inside appended folder


import os
import glob
reload(sys)
sys.setdefaultencoding('UTF-8')
files = glob.glob('/home/abc/Desktop/adm/appended_files/*')
for f in files:
    os.remove(f)

os.remove("/home/abc/Desktop/adm/QueryInfo.txt")


noofcluster = len(os.listdir("/home/abc/Desktop/adm/segment_sets_P"))
noofcluster = noofcluster/100

# List Of English Stop Words
# http://armandbrahaj.blog.al/2009/04/14/list-of-english-stop-words/
_WORD_MIN_LENGTH = 2
_STOP_WORDS = frozenset([
'a', 'about', 'above', 'above', 'across', 'after', 'afterwards', 'again', 
'against', 'all', 'almost', 'alone', 'along', 'already', 'also','although',
'always','am','among', 'amongst', 'amoungst', 'amount',  'an', 'and', 'another',
'any','anyhow','anyone','anything','anyway', 'anywhere', 'are', 'around', 'as',
'at', 'back','be','became', 'because','become','becomes', 'becoming', 'been', 
'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 
'between', 'beyond', 'bill', 'both', 'bottom','but', 'by', 'call', 'can', 
'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 
'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 
'either', 'eleven','else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 
'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 
'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 
'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get',
'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 
'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 
'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'in', 'inc', 
'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 
'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 
'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 
'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 
'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 
'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only',
'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out',
'over', 'own','part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same',
'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 
'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 
'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 
'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 
'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 
'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third',
'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 
'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 
'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 
'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter',
'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 
'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 
'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself',
'yourselves', 'the'])

def word_split(text):
    """
    Split a text in words. Returns a list of tuple that contains
    (word, location) location is the starting byte position of the word.
    """
    word_list = []
    wcurrent = []
    windex = None

    for i, c in enumerate(text):
        if c.isalnum():
            wcurrent.append(c)
            windex = i
        elif wcurrent:
            word = u''.join(wcurrent)
            word_list.append((windex - len(word) + 1, word))
            wcurrent = []

    if wcurrent:
        word = u''.join(wcurrent)
        word_list.append((windex - len(word) + 1, word))

    return word_list

def words_cleanup(words):
    """
    Remove words with length less then a minimum and stopwords.
    """
    cleaned_words = []
    for index, word in words:
        if len(word) < _WORD_MIN_LENGTH or word in _STOP_WORDS:
            continue
        cleaned_words.append((index, word))
    return cleaned_words

def words_normalize(words):
    """
    Do a normalization precess on words. In this case is just a tolower(),
    but you can add accents stripping, convert to singular and so on...
    """
    normalized_words = []
    for index, word in words:
        wnormalized = word.lower()
        normalized_words.append((index, wnormalized))
    return normalized_words

def word_index(text):
    """
    Just a helper method to process a text.
    It calls word split, normalize and cleanup.
    """
    words = word_split(text)
    words = words_normalize(words)
    words = words_cleanup(words)
    return words

def inverted_index(text):
    """
    Create an Inverted-Index of the specified text document.
        {word:[locations]}
    """
    inverted = {}

    for index, word in word_index(text):
        locations = inverted.setdefault(word, [])
        locations.append(index)

    return inverted

def inverted_index_add(inverted, doc_id, doc_index):
    """
    Add Invertd-Index doc_index of the document doc_id to the 
    Multi-Document Inverted-Index (inverted), 
    using doc_id as document identifier.
        {word:{doc_id:[locations]}}
    """
    for word, locations in doc_index.iteritems():
        indices = inverted.setdefault(word, {})
        indices[doc_id] = locations
    return inverted

def search(inverted, query):
    """
    Returns a set of documents_full id that contains all the words in your query.
    """
    words = [word for _, word in word_index(query) if word in inverted]
    results = [set(inverted[word].keys()) for word in words]
    return reduce(lambda x, y: x & y, results) if results else []




#ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff


_WORD_MIN_LENGTH = 2
_STOP_WORDS = frozenset([
'a', 'about', 'above', 'above', 'across', 'after', 'afterwards', 'again', 
'against', 'all', 'almost', 'alone', 'along', 'already', 'also','although',
'always','am','among', 'amongst', 'amoungst', 'amount',  'an', 'and', 'another',
'any','anyhow','anyone','anything','anyway', 'anywhere', 'are', 'around', 'as',
'at', 'back','be','became', 'because','become','becomes', 'becoming', 'been', 
'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 
'between', 'beyond', 'bill', 'both', 'bottom','but', 'by', 'call', 'can', 
'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 
'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 
'either', 'eleven','else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 
'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 
'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 
'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get',
'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 
'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 
'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'in', 'inc', 
'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 
'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 
'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 
'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 
'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 
'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only',
'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out',
'over', 'own','part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same',
'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 
'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 
'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 
'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 
'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 
'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third',
'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 
'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 
'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 
'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter',
'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 
'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 
'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself',
'yourselves', 'the','author','conclusion'])

def word_split(text):
    """
    Split a text in words. Returns a list of tuple that contains
    (word, location) location is the starting byte position of the word.
    """
    word_list = []
    wcurrent = []
    windex = None

    for i, c in enumerate(text):
        if c.isalnum():
            wcurrent.append(c)
            windex = i
        elif wcurrent:
            word = u''.join(wcurrent)
            word_list.append((windex - len(word) + 1, word))
            wcurrent = []

    if wcurrent:
        word = u''.join(wcurrent)
        word_list.append((windex - len(word) + 1, word))

    return word_list

def words_cleanup(words):
    """
    Remove words with length less then a minimum and stopwords.
    """
    cleaned_words = []
    for index, word in words:
        if len(word) < _WORD_MIN_LENGTH or word in _STOP_WORDS:
            continue
        cleaned_words.append((index, word))
    return cleaned_words

def words_normalize(words):
    """
    Do a normalization precess on words. In this case is just a tolower(),
    but you can add accents stripping, convert to singular and so on...
    """
    normalized_words = []
    for index, word in words:
        wnormalized = word.lower()
        normalized_words.append((index, wnormalized))
    return normalized_words

def word_index(text):
    """
    Just a helper method to process a text.
    It calls word split, normalize and cleanup.
    """
    words = word_split(text)
    words = words_normalize(words)
    words = words_cleanup(words)
    return words

def inverted_index1(text):
    """
    Create an Inverted-Index of the specified text document.
        {word:[locations]}
    """
    inverted1 = {}

    for index, word in word_index(text):
        locations = inverted1.setdefault(word, [])
        locations.append(index)

    return inverted1

def inverted_index_add1(inverted1, doc_id, doc_index):
    """
    Add Invertd-Index doc_index of the document doc_id to the 
    Multi-Document Inverted-Index (inverted), 
    using doc_id as document identifier.
        {word:{doc_id:[locations]}}
    """
    for word, locations in doc_index.iteritems():
        indices = inverted1.setdefault(word, {})
        indices[doc_id] = locations
    return inverted1

def search(inverted1, query):
    """
    Returns a set of documents_full id that contains all the words in your query.
    """
    words = [word for _, word in word_index(query) if word in inverted1]
    results = [set(inverted1[word].keys()) for word in words]
    return reduce(lambda x, y: x & y, results) if results else []










if __name__ == '__main__':
    corpusdir = '/home/abc/Desktop/adm/new_dataset_Inv/'
    newcorpus = PlaintextCorpusReader(corpusdir,'.*')
    sortedall = sorted(newcorpus.fileids())
    tokenizer = RegexpTokenizer(r'\w+')
# compile sample documents_full into a list
    doc_set = newcorpus.fileids()
    doc_set = sorted(newcorpus.fileids())
    file_full = []
    for i in doc_set:
	#print i
	raw = open(corpusdir + i).read()
	
	file_full.append(raw)
	
	
   


    doc1 = """Sixth Applied Natural Language Processing Conference"""

    doc2 = """Sixth Applied Natural Language Processing Conference"""
    str1=""
    i=0
    f_n = 1000
    documents_full = { }
#default_data.update({'item3': 3})
    while i < len(file_full):
	#p =str(f_n)+".txt"
	documents_full.update({doc_set[i]: file_full[i]})
	i=i+1
	f_n=f_n+1
   

    # Build Inverted-Index for documents_full
    inverted = {}
    #documents_full = {'0.txt':doc1, '1.txt':doc2,'2.txt':doc1, '3.txt':doc2,'4.txt':doc1, '5.txt':doc2}
    	
    for doc_id, text in documents_full.iteritems():
        doc_index = inverted_index(text)
        inverted_index_add(inverted, doc_id, doc_index)
    huge = open("/home/abc/Desktop/adm/QueryInfo.txt",'w')
    huge.write("INVERTED INDEX FOR DOCUMENTS" + '\n')
   
    for word, doc_locations in inverted.iteritems():
        #print word, doc_locations
	huge.write(str(word) + "   " +str(doc_locations) + '\n')
    huge.close()


#changes for inverted index of titles




#creating index for titles

    tree = ET.parse('/home/abc/Desktop/adm/xml/A00.xml')
    root = tree.getroot()
    title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)
	#
    tree = ET.parse('/home/abc/Desktop/adm/xml/A83.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)

    tree = ET.parse('/home/abc/Desktop/adm/xml/A88.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)

    tree = ET.parse('/home/abc/Desktop/adm/xml/A94.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)

    tree = ET.parse('/home/abc/Desktop/adm/xml/A97.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)

    tree = ET.parse('/home/abc/Desktop/adm/xml/C02.xml')
    root = tree.getroot()    
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)

    tree = ET.parse('/home/abc/Desktop/adm/xml/C82.xml')
    root = tree.getroot()    
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)

    tree = ET.parse('/home/abc/Desktop/adm/xml/C90.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)

    tree = ET.parse('/home/abc/Desktop/adm/xml/E83.xml')
    root = tree.getroot()    
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)

    tree = ET.parse('/home/abc/Desktop/adm/xml/N03.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)
	
	
    
    tree = ET.parse('/home/abc/Desktop/adm/xml/P02.xml')
    root = tree.getroot()    
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)

    tree = ET.parse('/home/abc/Desktop/adm/xml/P03.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)

    
    tree = ET.parse('/home/abc/Desktop/adm/xml/P06.xml')
    root = tree.getroot()    
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)

    tree = ET.parse('/home/abc/Desktop/adm/xml/M91.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)
    tree = ET.parse('/home/abc/Desktop/adm/xml/M92.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)
    tree = ET.parse('/home/abc/Desktop/adm/xml/M93.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)
    tree = ET.parse('/home/abc/Desktop/adm/xml/M95.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)



    tree = ET.parse('/home/abc/Desktop/adm/xml/M98.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)
    tree = ET.parse('/home/abc/Desktop/adm/xml/W03.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)



    tree = ET.parse('/home/abc/Desktop/adm/xml/T75.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)





    tree = ET.parse('/home/abc/Desktop/adm/xml/T78.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)



    tree = ET.parse('/home/abc/Desktop/adm/xml/T87.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)
    tree = ET.parse('/home/abc/Desktop/adm/xml/N04.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)



    tree = ET.parse('/home/abc/Desktop/adm/xml/W03_1.xml')
    root = tree.getroot()
    #title = []
    for elem in root.findall('paper/title'):
	title.append(elem.text)











    f = open("/home/abc/Desktop/adm/title/title_data1",'w')
    stm=0
    while stm < len(title):
	raw = title[stm]
	
	
	f.writelines(raw)
	f.write("{}\n".format(" "))
	stm=stm+1
    f.close()







	
    f = open("/home/abc/Desktop/adm/title/title_data",'w')
    stm=0
    while stm < len(title):
	raw = title[stm]
	tokens = nltk.wordpunct_tokenize(raw)
	#tokens
	porter = nltk.PorterStemmer()
	P=[porter.stem(t) for t in tokens if t.isalpha()]
	text = nltk.Text(P)
	words = [w.lower()+" " for w in text]
	
	f.writelines(words)
	f.write("{}\n".format(" "))
	stm=stm+1
    f.close()

  
    title = []
    with open("/home/abc/Desktop/adm/title/title_data", "r") as ins:
    #array = []
    	for line in ins:
        	title.append(line)
  

    doc1 = """Sixth Applied Natural Language Processing Conference"""

    doc2 = """Sixth Applied Natural Language Processing Conference"""
    str1=""
    i=0
    f_n = 1000
    documents = { }
#default_data.update({'item3': 3})
    while i < len(title):
	p =str(f_n)+".txt"
	documents.update({p: title[i]})
	i=i+1
	f_n=f_n+1
   

    # Build Inverted-Index for documents
    inverted1 = {}
    #documents = {'0.txt':doc1, '1.txt':doc2,'2.txt':doc1, '3.txt':doc2,'4.txt':doc1, '5.txt':doc2}
    	
    for doc_id, text in documents.iteritems():
        doc_index = inverted_index1(text)
        inverted_index_add1(inverted1, doc_id, doc_index)
    huge = open("/home/abc/Desktop/adm/QueryInfo.txt",'ab')
    huge.write("INVERTED INDEX FOR TITLE" + '\n')
    for word, doc_locations in inverted1.iteritems():
        #print word, doc_locations
	huge.write(str(word) + "   " +str(doc_locations) + '\n')
    huge.close()























#Creating clusters for all files

corpusdir = '/home/abc/Desktop/adm/segment_sets_P'
newcorpus = PlaintextCorpusReader(corpusdir,'.*')
sortedall = sorted(newcorpus.fileids())
tokenizer = RegexpTokenizer(r'\w+')
doc_set = sorted(newcorpus.fileids())
oho = open("/home/abc/Desktop/adm/appended_files/detail_all.txt" ,'w')
oho.close()
#print doc_set
for i in doc_set:
	os.system("cat "  +corpusdir+"/"+i + " >> "+"/home/abc/Desktop/adm/appended_files/detail_all.txt")
	#os.system("echo \ >> " + corpusdir_p + nameoffile)
	newfile = open("/home/abc/Desktop/adm/appended_files/detail_all.txt" ,'ab')
	newfile.write("{}\n".format(" "))
	newfile.close()
			#os.system("echo \ >> " + corpusdir_p + nameoffile)
	#os.system("echo \ >> " + "/home/rashmi/Documents/adm_project/trial/try.txt")
f = open("/home/abc/Desktop/adm/appended_files/detail_all.mat","w")
f.close()
os.system("perl /home/abc/Desktop/adm/program_file/doc2mat-1.0/doc2mat /home/abc/Desktop/adm/appended_files/detail_all.txt /home/abc/Desktop/adm/appended_files/detail_all.mat")

#integration with cluto

output = subprocess.check_output("/home/abc/Desktop/adm/program_file/cluto-2.1.2/Linux-x86_64/vcluster  -showfeatures -nfeatures=10 /home/abc/Desktop/adm/appended_files/detail_all.mat "+str(noofcluster), shell=True)
##print output
f = open("/home/abc/Desktop/adm/appended_files/clusAll.txt","w")
f.writelines(output)
f.close() 
#stage 4
s=0
#removing .txt
'''
for i in doc_set:
	a = i.split("_")
	doc_set[s] = a[0] + ".txt"
	s=s+1
'''
print "doc sett after removing"
#print doc_set	
arr = [[] for _ in range(noofcluster)]
i = 0
with open("/home/abc/Desktop/adm/appended_files/detail_all.mat.clustering."+str(noofcluster)) as ins:
    for line in ins:
        line = line.rstrip('\n')
	if int(line) > -1:
        	arr[int(line)].append(doc_set[i])
        i=i+1
p=0
while p < len(arr):
	arr[p] = list(OrderedDict.fromkeys(arr[p]))
	p = p + 1
#for clu1 in len(arr):
#	print clu1	
#arr[clu1] = list(OrderedDict.fromkeys(arr[clu1]))

print "clllllllllllllllllllllllllllllclearend"
#print arr


#reading fetures from file clus.txt

fp = open("/home/abc/Desktop/adm/appended_files/clusAll.txt")
c = 0
empty = []
fp.seek(0)
for i in fp:
    c += 1
    s = i.rstrip()
    matches = s.find("way clustering solution")
    if matches > 0:
        #print c
        break

fp.seek(0)
d = 0
for i in fp:
    d = d + 1
    if d == c:
        j = i.rstrip()
        k = re.findall("[-+]?\d+[\.]?\d*", j)
        #print k[0]

N = int(k[0])
fp.seek(0)
clusters = []
d = 0
l = (c + 3) % 4
#print l
m = (c+3)+4*(N-1)
#print m

for i in fp:
    d += 1
    if d >= c+3 and (d % 4 is l):
        a = i.rstrip()
        b = a.split()
        e = []
        for k in range(len(b)):
            if k % 2 is 1:
                e.append(b[k])
        #print 'cluster : '
        #print e
        clusters.append(e)
    if d == m:
        fp.seek(0)
        break

	
#working on my improvemnt


doc_IDs = []
id=1000
cnt = 0
doc_set =[]

doc_set.append("/home/abc/Desktop/adm/xml/A00.xml")
doc_set.append("/home/abc/Desktop/adm/xml/A83.xml")
doc_set.append("/home/abc/Desktop/adm/xml/A88.xml")
doc_set.append("/home/abc/Desktop/adm/xml/A94.xml")
doc_set.append("/home/abc/Desktop/adm/xml/A97.xml")
doc_set.append("/home/abc/Desktop/adm/xml/C02.xml")
doc_set.append("/home/abc/Desktop/adm/xml/C82.xml")
doc_set.append("/home/abc/Desktop/adm/xml/C90.xml")
doc_set.append("/home/abc/Desktop/adm/xml/E83.xml")
doc_set.append("/home/abc/Desktop/adm/xml/N03.xml")
doc_set.append("/home/abc/Desktop/adm/xml/P02.xml")
doc_set.append("/home/abc/Desktop/adm/xml/P03.xml")
doc_set.append("/home/abc/Desktop/adm/xml/P06.xml")
doc_set.append("/home/abc/Desktop/adm/xml/M91.xml")

doc_set.append("/home/abc/Desktop/adm/xml/M92.xml")
doc_set.append("/home/abc/Desktop/adm/xml/M93.xml")
doc_set.append("/home/abc/Desktop/adm/xml/M95.xml")
doc_set.append("/home/abc/Desktop/adm/xml/M98.xml")


doc_set.append("/home/abc/Desktop/adm/xml/W03.xml")
doc_set.append("/home/abc/Desktop/adm/xml/T75.xml")
doc_set.append("/home/abc/Desktop/adm/xml/T78.xml")
doc_set.append("/home/abc/Desktop/adm/xml/T87.xml")

doc_set.append("/home/abc/Desktop/adm/xml/N04.xml")
doc_set.append("/home/abc/Desktop/adm/xml/W03_1.xml")




while cnt < len(doc_set):
	#tree = ET.parse('/home/rashmi/Documents/adm_project/titles/' + doc_set[cnt])
	tree = ET.parse(doc_set[cnt])
	
	#tree = ET.parse('/home/rashmi/Documents/adm_project/titles/A00.xml')
        
        root = tree.getroot()
    	#len(os.listdir("/home/abc/Desktop/adm/new_dataset"))
	for elem in root.findall('paper/title'):
		if id < (len(os.listdir("/home/abc/Desktop/adm/new_dataset"))+1000) :
			#title.append(elem)
			doc_IDs.append(str(id)+".txt")
			print "id"
			print  id
			print elem.text
			newfile = open("/home/abc/Desktop/adm/appended_files/all_title.txt" ,'ab')
			newfile.write("{}\n".format(elem.text))
			#newfile.write(elem.text)
			newfile.close()
		id=id+1
	cnt = cnt+1
f = open("/home/abc/Desktop/adm/appended_files/all_title.mat","w")
f.close()
os.system("perl /home/abc/Desktop/adm/program_file/doc2mat-1.0/doc2mat /home/abc/Desktop/adm/appended_files/all_title.txt /home/abc/Desktop/adm/appended_files/all_title.mat")

#writing output to cluto files
noofclustertitle = int(noofcluster*1.5)
output = subprocess.check_output("/home/abc/Desktop/adm/program_file/cluto-2.1.2/Linux-x86_64/vcluster -showfeatures  -nfeatures=20 /home/abc/Desktop/adm/appended_files/all_title.mat "+str(noofclustertitle), shell=True)
##print output
f = open("/home/abc/Desktop/adm/appended_files/cluster_title.txt","w")
f.writelines(output)
f.close() 




# Reading the cluto file for tiltes cluster


fp = open("/home/abc/Desktop/adm/appended_files/cluster_title.txt")
c = 0
empty = []
fp.seek(0)
for i in fp:
    c += 1
    s = i.rstrip()
    matches = s.find("way clustering solution")
    if matches > 0:
        #print c
        break

fp.seek(0)
d = 0
for i in fp:
    d = d + 1
    if d == c:
        j = i.rstrip()
        k = re.findall("[-+]?\d+[\.]?\d*", j)
        #print k[0]

N = int(k[0])
fp.seek(0)
clusters_ti = []
d = 0
l = (c + 3) % 4
#print l
m = (c+3)+4*(N-1)
#print m

for i in fp:
    d += 1
    if d >= c+3 and (d % 4 is l):
        a = i.rstrip()
        b = a.split()
        e = []
        for k in range(len(b)):
            if k % 2 is 1:
                e.append(b[k])
        #print 'cluster : '
        #print e
        clusters_ti.append(e)
    if d == m:
        fp.seek(0)
        break





















#getting the documents assigned to a cluster
title_clus = [[] for _ in range(noofclustertitle)]
i = 0
with open("/home/abc/Desktop/adm/appended_files/all_title.mat.clustering."+str(noofclustertitle)) as ins:
    for line in ins:
        line = line.rstrip('\n')
        title_clus[int(line)].append(doc_IDs[i])
        i=i+1


new_list_C = copy.deepcopy(arr)
new_list_Changed = copy.deepcopy(arr)
ch = 0
while ch < len(new_list_Changed):
	po=0
	while po < len(new_list_Changed[ch]):
		a = new_list_Changed[ch][po]
	        a = a.split("_")
	        new_list_Changed[ch][po] = a[0] + ".txt"
	       	po=po+1
	ch=ch+1

print "paper clusters"
#print arr
print "copied clustersssssssssssssssssssssssssss"
#print new_list_C


#improvemnt main logic

print "title original  clusters"
#print title_clus
title_clus_c=0
print " original  clusters features b"
#print clusters
print "title original  features b"
#print clusters_ti
Improve_file = open("/home/abc/Desktop/adm/appended_files/Improve_file.txt",'w')
while title_clus_c < len(title_clus):
	if len(title_clus[title_clus_c]) > 0 : 
		#print title_clus[title_clus_c]
		full_set = set(title_clus[title_clus_c])
		#full_set = set(['1080.txt'])
		#print "fullll settttttttttt"
		#print full_set
		title_clus_a = 0
		flag = 0
		while flag == 0 and title_clus_a < len(new_list_C):
			if full_set.issubset(new_list_Changed[title_clus_a]):
				# append in feature list cluster[].append(cluster_t[])
				f_clu = 0
				while f_clu < len(clusters_ti[title_clus_c]):
					clusters[title_clus_a].append(clusters_ti[title_clus_c][f_clu])
					f_clu = f_clu+1
				print "true"
				flag=1
			title_clus_a =title_clus_a+1
			if flag == 0 :
				print "false"
					#clusters[].append() 
				#you have a list of keywords append in them
				if flag == 0: 
					life = 0
					#lenght = len(title_clus[title_clus_c])
					orig_l = len(title_clus[title_clus_c])
										#while life == 0:
					buffe = (len(title_clus[title_clus_c]))*0.5
					exp = orig_l -buffe

					appendedd="false"
					flagp=0
					title_clus_ap = 0
					#while flagp == 0 title_clus_ap < len(new_list_Changed) and life==0:
					while title_clus_ap < len(new_list_Changed):
								print "subset"
								'''print "subsettt"
								print part_set
								print "cluster set"
								print arr[title_clus_ap]'''
								size = set(title_clus[title_clus_c]).intersection(new_list_Changed[title_clus_ap])
								size = list(size)
								if len(size) >= buffe:
									print "true got in subset"
								#	print title_clus'''
									flagp=1 
									life=1
									t_c = 0
									if appendedd == "false":
										while t_c < len(title_clus[title_clus_c]):
											#print len(title_clus[title_clus_c])
											#print "elemts"
											#print arr[title_clus_c][t_c]
											#print title_clus_ap
											#print t_c
											arr[title_clus_ap].append(title_clus[title_clus_c][t_c])
											t_c = t_c + 1	
										
										appendedd="true"
										f_clu = 0
						                                while f_clu < len(clusters_ti[title_clus_c]):
							                        	clusters[title_clus_ap].append(clusters_ti[title_clus_c][f_clu])
							                        	f_clu = f_clu+1
				
								title_clus_ap=title_clus_ap+1
							
								       #you have a list o	f keywords append in them
									#REdefine that cluster
					
			title_clus_a=title_clus_a+1

	title_clus_c=title_clus_c+1
	




crr = 0
while crr <len(title_clus):
	crr1 =0
	while crr1 < len(arr):
		if set(title_clus[crr]).issubset(arr[crr1]):
			Improve_file.write("files and features added to  Cluster id " + str(crr1) + '\n')
			Improve_file.write("====================================================================="+'\n')
			Improve_file.write("files added"+'\n')
			Improve_file.writelines(str(title_clus[crr]) + '\n')
			Improve_file.write("features added"+'\n')
			Improve_file.writelines(str(clusters_ti[crr]) + '\n')
			Improve_file.write("====================================================================="+'\n')
		
		crr1 = crr1+1
	
	crr = crr+1



Improve_file.close()

print "changes original clusterrrr"
#print arr
print "untouched cluster"
#print new_list_Changed
print "title original  clusters"
#print title_clus

'''
print "new_list_Changed"
print new_list_Changed
print " original  clusters features"
print clusters
print "title original  features"
print clusters_ti
'''




print "DDDDDDDDDDDDDDDDDDDDDD set"	
#print clusters
print "finding a word in a cluster"
a = ["product"]
a = set(a)
sr = 0
flag = 0
while sr < len(clusters):
	if a.issubset(clusters[sr]):
		print "ELements of cluster"
		print arr[sr]
		print sr
		flag=1
	sr=sr+1

'''
dictionary=PyDictionary("product")
syn = (dictionary.getSynonyms())
if len(syn) > 0 : 
	syn=syn[0]
	l = syn.values()[0]
	p = [x.encode('UTF8') for x in l]
	cct=0
	while cct < len(p):
		newlist = []
		newlist.append(p[cct])
		newlist = set(newlist)

		sr = 0
		while sr < len(clusters):
			if newlist.issubset(clusters[sr]):
				print "got the syn"
				print arr[sr]
				print sr
				flag = 1
			else:
				print "did not get syn"
			sr=sr+1
		cct=cct+1

	#print "call inverted"
#new_list = copy.copy(arr)
print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@title clusters"
#print title_clus
print "###################################file clusters"
#print arr

'''

huge = open("/home/abc/Desktop/adm/QueryInfo.txt",'ab')
huge.write("CLUSTER AND FEATURES" + '\n')
   
i = 0
while i < len(clusters):
	j = arr[i]
	p= set(j)
    #for word, doc_locations in inverted.iteritems():
        #print word, doc_locations
	p = list(p)
	huge.write("Cluster "+ str(i) + " " + "Documents : ")
	huge.write(str(p) + '\n')
	p = clusters[i]
	p = set(p)
	p = list(p)
	huge.write("Cluster "+ str(i) + " " + "Features : ")
	huge.write(str(p) + '\n')
	
	i=i+1
huge.close()
print "changes original clusterrrr"
#print arr







#files for query evaluation my changes file

huge = open("/home/abc/Desktop/adm/results/Cluster_Omega_impr.txt",'w')
huge.write(str(len(clusters)) + '\n')
#No of unique document 
#noofclustertitle = int(noofcluster*1.5)
huge.write(str(noofclustertitle) + '\n')
new_list_im = copy.deepcopy(arr)
i = 0
while i < len(clusters):
	j = new_list_im[i]
	s = 0 
	while s <  len(j):
		bb = j[s].split("_")
		k=bb[0]
		bb = k.split(".")
		j[s]=bb[0]
		s=s+1
	
	p= set(j)
    #for word, doc_locations in inverted.iteritems():
        #print word, doc_locations
	p = list(p)
	p = sorted(p)
	s = 0 
	while s <  len(p)-1:
		bb = p[s].split("_")
		huge.write(str(bb[0]) + " ")
		
		s=s+1
	#i = i+1
	bb = p[s].split("_")
	huge.write(str(bb[0])+"\n")
        i=i+1
huge.close()
print "changes original clusterrrr"
#print arr


#file for omega index unimproved
huge = open("/home/abc/Desktop/adm/results/Cluster_Omega_NoImp.txt",'w')
huge.write(str(len(clusters)) + '\n')
#No of unique document 
huge.write(str(noofcluster) + '\n')
 
i = 0
while i < len(clusters):
	j = new_list_Changed[i]
	p= set(j)
    #for word, doc_locations in inverted.iteritems():
        #print word, doc_locations
	p = list(p)
	p = sorted(p)
	s = 0 
	while s <  len(p)-1:
		bb = p[s].split(".")
		huge.write(str(bb[0]) + " ")
		s=s+1
	#i = i+1
	bb = p[s].split(".")
	huge.write(str(bb[0])+"\n")
        i=i+1
huge.close()
#print "Cluster output of program without impprovemnts"
#print tabulate({"Cluster ID": [[0],[1],[2],[3]], "files": title_details,"Top Features":clusters_ti}, headers="keys")


#print "valooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo"
#print new_list_C
#print clusters




#file of cluster of title
huge = open("/home/abc/Desktop/adm/results/Title.txt",'w')
huge.write(str(len(clusters_ti)) + '\n')
#No of unique document 
huge.write(str(15) + '\n')

title_details = []   
i = 0
print "Lenght of clustersss"
print clusters_ti
print len(clusters_ti)
print "title_clus"
print title_clus
print "title_details"
print title_details
while i < len(clusters_ti):
	try:
		j = title_clus[i]
		print "j"
		print j
		p= set(j)
	    #for word, doc_locations in inverted.iteritems():
		#print word, doc_locations
		p = list(p)
		p = sorted(p)
		s = 0 
		title_details.append(p)
		while s <  len(p)-1:
			bb = p[s].split(".")
			print "ps inside " + p[s]
			huge.write(str(bb[0]) + " ")
			s=s+1
		#i = i+1
		bb = p[s].split(".")
		huge.write(str(bb[0])+"\n")
		i=i+1
	except:
		print p[s]
huge.close()

#print tabulate({"Cluster ID": [[0],[1],[2],[3]], "files": title_details}, headers="keys")


'''
#applying LDA










'''

'''
ferw = open("/home/abc/Desktop/adm/Clusters/checkin.txt",'w')
ferw.write("cluster==============================================")
ferw.write(str(clusters))
ferw.close()
'''

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()


# list for tokenized documents in loop
texts = []
t_d = 0
val=1000
# loop through document list
#no of clusters
#as = []
lda_All = []
while t_d < len(new_list_C):
	pointer = 'True'
	empty = []
	lp_clus=0
	if len(new_list_C[t_d])>0 :
		while lp_clus < len(new_list_C[t_d]):
			empty.append("/home/abc/Desktop/adm/segment_sets_P/"+new_list_C[t_d][lp_clus])
			lp_clus = lp_clus+1
	
		texts = []
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
		ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word = dictionary, passes=20)
		#for top in ldamodel.print_topics():
		  #print top
		#declaring number of topics
		num_topics = 4
		arr_lda = [[] for _ in range(num_topics)]
		# Assigns the topics to the documents in corpus
		counter=0
		print "Elements in arr_ldaay"
#		#print empty		
		while counter < len(empty):
			lda_corpus = ldamodel[corpus[counter]]
			num_topics1=len(lda_corpus)
			print "num_topics"
#			#print num_topics1
		
			i=0
			print "Corpussssssssss Containsss"
#			print lda_corpus
			topics = []
			while i < num_topics1:
				topics.append(lda_corpus[i][1])
				#print lda_corpus[i][1]
				print "topics"
				#print topics
				i=i+1
			index, value = max(enumerate(topics), key=operator.itemgetter(1))
			#print ldamodel.top_topics(corpus[4],2)
			indexf = lda_corpus[index][0]
			#print index
			#print value
			#print indexf
			topics[index] = 0
			flag=0	
			if len(lda_corpus) > 1: 
				index1, value1 = max(enumerate(topics), key=operator.itemgetter(1))
				indexff = lda_corpus[index1][0]
		#		print index1
		#		print value1
		#		print indexff
				flag=1
			if flag==1:
				 arr_lda[int(indexff)].append(empty[counter])
				 flag=0
			arr_lda[int(indexf)].append(empty[counter])
			counter=counter+1
		print "+++++++++++++++++++++++++++++++++++++++++++++++++"	
		#print arr_lda
		va=0
		namef = str(t_d)+".txt"
		f = open("/home/abc/Desktop/adm/Clusters/"+namef,'w')
		counter1=0
		lda_specific = []
		while counter1 < len(arr_lda):
			if len(arr_lda[counter1]) > 0 : 

				wr = ldamodel.print_topic(counter1,topn=20)
				print "topic vallllllllllllllllllllllllll"
		#		print wr
				f.write("Topic " + str(counter1) + " : " )
		#------------------
				f.write(wr + '\n')
				raw = nltk.wordpunct_tokenize(wr)
				tokens = [w for w in raw if w.isalpha()]
				#clusters add tokens in this clusters[namef].append(tokens)
				lda_specific.append(tokens)
				tkens = 0
				while tkens < len(tokens):
					pasc = tokens[tkens].encode('UTF8') 
					clusters[t_d].append(pasc)
					tkens = tkens + 1
				f.write("files " + str(counter1) + " : " )
				f.writelines(str(arr_lda[counter1]) + '\n')
		
				#va=va+1
			counter1=counter1+1
		f.close()

	t_d=t_d+1
	#-----------------------
	lda_All.append(lda_specific)
	#print len(arr_lda[0])
	#val=val+1


print lda_All
#print arr_lda

'''
ferw = open("/home/abc/Desktop/adm/Clusters/checkin.txt",'ab')
ferw.write("lda_all==============================================================")
ferw.write(str(lda_All))
ferw.write("afte appendlda_all==============================================================")
ferw.write(str(clusters))
ferw.close()

'''
'''
#Query processing 
'''

#searching in a cluster

print "DDDDDDDDDDDDDDDDDDDDDD set"	
#print clusters
print "finding a word in a cluster"
raw = "semant corpu"
tokens = nltk.wordpunct_tokenize(raw)
	#tokens
porter = nltk.PorterStemmer()
P=[porter.stem(t) for t in tokens]
text = nltk.Text(P)
#queries = words
words = [w.lower() for w in text]


#occurence of value in cluter
qu_occ = [[] for _ in range(len(clusters))]
sr =0 
while sr < len(clusters):
	qu_occ[sr]=0
	sr=sr+1
nu = 0
while nu < len(words):  
	#a = set([])
	#a.add(words[nu])
	a = words[nu]
	a = a.encode('utf8')
	b = []
	b.append(a)
	sr = 0
	flag = 0
	while sr < len(clusters):
		print "cluster element"
		print clusters[sr]
		print "querryyyy ele"
		print b
		if set(b).issubset(clusters[sr]):
			print "ELements of cluster"
			qu_occ[sr] =qu_occ[sr]+1
			#print arr[sr]
			#print sr
			flag=1
		sr=sr+1
	nu = nu+1

print "Matrixx   dataaaaaaaaaaaaaAAAA"
print qu_occ
#check which cluster has the maximum query support
indexA, value = max(enumerate(qu_occ), key=operator.itemgetter(1))
print "higherst occ"
print indexA
#get all the files from that clusters
all = []
all.append(arr[indexA])
useless = []
tophighers = []
sy = 0
compared = set(arr[indexA])
print "Actually cluster ele"
print compared
result_docs_INV = []
result_docs1=[]
no_results = "false"
for query in words:
        result_docs1 = search(inverted, query)
	result_docs_INV.append(set(result_docs1))
	if len(result_docs1)>0:
		no_results = "true"
		
'''
if len(result_docs_INV)  == 10:
	size = set.intersection(compared,result_docs_INV[9],result_docs_INV[8],result_docs_INV[7],result_docs_INV[6],result_docs_INV[5],result_docs_INV[4],result_docs_INV[3],result_docs_INV[2],result_docs_INV[1],result_docs_INV[0])
if len(result_docs_INV)  == 9:
	size = set.intersection(compared,result_docs_INV[8],result_docs_INV[7],result_docs_INV[6],result_docs_INV[5],result_docs_INV[4],result_docs_INV[3],result_docs_INV[2],result_docs_INV[1],result_docs_INV[0])
if len(result_docs_INV)  == 8:
	size = set.intersection(compared,result_docs_INV[7],result_docs_INV[6],result_docs_INV[5],result_docs_INV[4],result_docs_INV[3],result_docs_INV[2],result_docs_INV[1],result_docs_INV[0])
if len(result_docs_INV)  == 7:
	size = set.intersection(result_docs_INV[6],result_docs_INV[5],result_docs_INV[4],result_docs_INV[3],result_docs_INV[2],result_docs_INV[1],result_docs_INV[0])
if len(result_docs_INV)  == 6:
	size = set.intersection(result_docs_INV[5],result_docs_INV[4],result_docs_INV[3],result_docs_INV[2],result_docs_INV[1],result_docs_INV[0])
if len(result_docs_INV)  == 5:
	size = set.intersection(result_docs_INV[4],result_docs_INV[3],result_docs_INV[2],result_docs_INV[1],result_docs_INV[0])
if len(result_docs_INV)  == 4:
	size = set.intersection(result_docs_INV[3],result_docs_INV[2],result_docs_INV[1],result_docs_INV[0])
if len(result_docs_INV)  == 3:
	size = set.intersection(result_docs_INV[3],result_docs_INV[2],result_docs_INV[1],result_docs_INV[0])
if len(result_docs_INV)  == 2:
	size = set.intersection(result_docs_INV[1],result_docs_INV[0])
	print "results from inverted index"
	print result_docs_INV[1]
	print result_docs_INV[0]
if len(result_docs_INV)  == 1:
	size = set.intersection(result_docs_INV[0])

inters = list(size)
print "ELement in alll"
print inters

tophighers.append(inters)
'''


if no_results=="true":
	if value > 0:
		print "tophigherrrr"
		print tophighers
		cdd = 0
		interns = []
		temp = []
		arr_slash = []
		inttt = 0
		while inttt < len(arr[indexA]):
			zx = arr[indexA][inttt].split('_')
	
	
			arr_slash.append(zx[0] + ".txt")
			inttt=inttt+1


		ferw = open("/home/abc/Desktop/adm/Clusters/checkin.txt",'w')
		ferw.write("arr[indexA][inttt]==============================================================")
		ferw.write(str(arr[indexA]))
		ferw.write("arr_slash==============================================================")
		ferw.write(str(arr_slash))
		#ferw.close()

		while cdd < len(result_docs_INV):
			size = set(arr_slash).intersection(list(result_docs_INV[cdd]))
			temp = list(size)
			pww = 0
			while pww < len(temp):
				interns.append(temp[pww])
				pww=pww+1
			cdd = cdd+1

		ferw.write("result_docs_INV==============================================================")
		ferw.write(str(result_docs_INV))
		ferw.write("interns==============================================================")
		ferw.write(str(interns))
		ferw.close()

		'''
		internss = []
		inttt = 0
		while inttt < len(interns):
			zx = interns[inttt].split('_')
	
	
			internss.append(zx[0] + ".txt")
			inttt=inttt+1
	
		interns = internss
		'''


		yettoorder = [[x,interns.count(x)] for x in set(interns)]
		print yettoorder
		findingmax = []
		inttt = 0
		while inttt < len(yettoorder):
			findingmax.append(yettoorder[inttt][1])
			inttt=inttt+1


		lne = 0
		#this is related to lda..no of parts of doc
		pri = findingmax[np.argmax(findingmax)]
		while lne < len(words):
			bb = 0
			while bb < len(yettoorder):
				if yettoorder[bb][1] == pri:
					tophighers.append(yettoorder[bb][0])
				bb=bb+1
			pri = pri-1
			lne = lne+1



		ferw = open("/home/abc/Desktop/adm/Clusters/QueryClusterOutput.txt",'w')
		ferw.write("All files of clusters ==============================================================")
		ferw.write(str(arr[indexA]))
		ferw.write("All files of clusters after striping ==============================================================")
		ferw.write(str(arr_slash))
		ferw.write("files in cluster having atleast 1 matching keywords as that of cluster rank==============================================================")
		ferw.write(str(tophighers))
		ferw.write("cluster feature==============================================================")
		ferw.write(str(clusters))





		#adding remaing element of clusters to the file
		size = list(set(arr_slash) - set(tophighers))
		temp = list(size)
		pww = 0
		while pww < len(temp):
			tophighers.append(temp[pww])
			pww=pww+1
	
	
		ferw.write("update tophighers==============================================================")
		ferw.write(str(temp))
		ferw.close()


		print "Appended"
		print tophighers

		QDetils = open("/home/abc/Desktop/adm/QueryOutput/QOutput.txt",'w')
		#now use the inverted index to get the location
		rt = 0
		srch = 0
		QDetils.write("Total no of Searched Results : " + str(len(tophighers)) + '\n')
		while srch < len(tophighers):
			flat = 0
			tt = 0
			print "file indexed"
			print tophighers[srch]
			while tt < len(words):
				a = words[tt]
				a = a.encode('utf8')
				#print "wordddddd"
				print a
				try:
					ing = inverted[a][tophighers[srch]]
					QDetils.write("Cluster id " + str(indexA) + '\n')
					QDetils.write("====================================================================="+'\n')
					fname = tophighers[srch].split(".")
					print tophighers[srch]
					print "file name"
					print tophighers[srch]
					print "title"
					#print title[fname[0]-1000]
					ty = int(fname[0])-1000
					QDetils.write("Paper Title " + title[ty] + '\n')
					#title.appen
					QDetils.write("File name " + tophighers[srch] + '\n')
			
					QDetils.write(documents_full[tophighers[srch]][ing[0]:ing[0]+100].replace('\n', ' '))
					QDetils.write('\n'+"====================================================================="+'\n')
			
					#print "Index "
					#print ing
					tt = len(words)
					break
				except:
					tt=tt+1
					rt = 1
					print "mmeeet an exception"	

				#tt=tt+1
			srch=srch+1

	

		QDetils.close()
		print "tophighers lenght"
		print len(tophighers)
		print len(set(tophighers))
		print title
		print len(title)
		print title[0]

		#appended results in a file

		#if there are fewer then 5 result then checks syn

		#if no syns list then go to inverted index



		'''
			print "Resuuuuuuuuuuuuuuuuuuuuuuuuuuuullllt"
			print result_docs1
			print "Search for '%s': %r" % (query, result_docs1)
			result_docs_INV.append(result_docs1)
			for _, word in word_index(query):
			    print "wordddddddddddd"
			    print word
			    def extract_text(doc, index): 
				return documents_full[doc][index:index+100].replace('\n', ' ')

			    for doc in result_docs1:
				print "doc of inverted"
				print doc
				print "word doc"
				print inverted[word][doc]
			       # for index in inverted[word][doc]:
				   # print '   - %s...' % extract_text(doc, index)
				  # F.write(extract_text(doc, index))


		#print "checking the nvalid"
		#print inverted["Rashmi"]["1075_4.txt"]
		print "Conatin of arr[index]"
		print arr[0]
		'''
		while sy < len(arr[indexA]):
	
			sy=sy+1







		'''




		'''

		'''



		#operation of inverted index

		flag=0
		if flag == 0:
		    F = open("try.txt","w")
		    # Search something and print results
		    #queries = ['Sixth', 'expands', 'West-coast Week']
		    raw = "asian gameseee entropi happy"
		    tokens = nltk.wordpunct_tokenize(raw)
			#tokens
		    porter = nltk.PorterStemmer()
		    P=[porter.stem(t) for t in tokens if t.isalpha()]
		    text = nltk.Text(P)
		    words = [w.lower()+" " for w in text]
		    queries = words
		    for query in queries:
			result_docs = search(inverted, query)
			print "Resuuuuuuuuuuuuuuuuuuuuuuuuuuuullllt"
			print result_docs
			print "Search for '%s': %r" % (query, result_docs)
			for _, word in word_index(query):
			    def extract_text(doc, index): 
				return documents_full[doc][index:index+100].replace('\n', ' ')

			    for doc in result_docs:
				for index in inverted[word][doc]:
				    print '   - %s...' % extract_text(doc, index)
				    F.write(extract_text(doc, index))
			print
		    F.close()


		#inverted index for title


		flag=0
		if flag == 0:
		    F = open("try.txt","w")
		    # Search something and print results
		    #queries = ['Sixth', 'expands', 'West-coast Week']
		   # queries = ['entropi']
		    for query in queries:
			result_docs = search(inverted1, query)
			print "Resuuuuuuuuuuuuuuuuuuuuuuuuuuuullllt from titleee"
			print result_docs
			print "Search for '%s': %r" % (query, result_docs)
			for _, word in word_index(query):
			    def extract_text(doc, index): 
				return documents[doc][index:index+100].replace('\n', ' ')



			    for doc in result_docs:
				for index in inverted1[word][doc]:
				    print '   - %s...' % extract_text(doc, index)
				    F.write(extract_text(doc, index))
			print
		    F.close()






		'''
		'''
		print tabulate({"Name": ["Alice", "Bob"],"Age": [24, 19]}, headers="keys")
		print tabulate({"Cluster ID": [[0],[1],[2],[3]], "files": title_details,"Top Features":clusters_ti}, headers="keys")
		'''



		print lda_All
	else:
		print "No match in Clusters"
else:
	print "Sorry no exact match element found"



#Code for synn






#---------------------



try:

	allwordssyn = []
	similar_occ = [[] for _ in range(len(clusters))]
	sr = 0
	while sr < len(clusters):
		similar_occ[sr] = 0
		sr=sr+1
	#####################printiing syn
	uu = 0
	raw = nltk.wordpunct_tokenize(raw)
	for query in raw:
		
		dictionary=PyDictionary(query)
		

		syn = (dictionary.getSynonyms())
		if len(syn) > 0 : 
			syn=syn[0]
			l = syn.values()[0]
			#tokens = nltk.wordpunct_tokenize(l)
		#tokens
			porter = nltk.PorterStemmer()
			P=[porter.stem(t) for t in l]
			text = nltk.Text(P)
	#queries = words
			wordssy = [w.lower() for w in text]
			ut =0
			while ut < len(wordssy):
				allwordssyn.append(wordssy[ut])
				uu=uu+1
				ut=ut+1

			p = [x.encode('UTF8') for x in wordssy]
			cct=0
			while cct < len(p):
				newlist = []
				newlist.append(p[cct])
				newlist = set(newlist)

				sr = 0
				while sr < len(clusters):
					if newlist.issubset(clusters[sr]):
						similar_occ[sr] = similar_occ[sr]+1
						print "got the syn"
						#print arr_slash[sr]
						print sr
						flag = 1
					else:
						print "did not get syn"
					sr=sr+1
				cct=cct+1

			#print "call inverted"
		#new_list = copy.copy(arr)
		print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@title clusters"
		#print title_clus
		print "###################################file clusters"
		#print arr
	ind = np.argmax(similar_occ)


	QDetils = open("/home/abc/Desktop/adm/QueryOutput/synOutput.txt",'w')
	#now use the inverted index to get the location
	rt = 0
	srch = 0
	arr_slash = []
	inttt = 0
	while inttt < len(arr[ind]):
		zx = arr[ind][inttt].split('_')
		arr_slash.append(zx[0] + ".txt")
		inttt=inttt+1
	arr_slash = list(set(arr_slash))
	QDetils.write("Total no of Searched Results for synonyms: " + str(len(arr_slash)) + '\n')
	while srch < len(arr_slash):
		flat = 0
		tt = 0
		print "file indexed"
		print arr_slash[srch]
		while tt < len(allwordssyn):
			a = allwordssyn[tt]
			a = a.encode('utf8')
			#print "wordddddd"
			print a
			try:
				ing = inverted[a][arr_slash[srch]]
				QDetils.write("Cluster id " + str(ind) + '\n')
				QDetils.write("====================================================================="+'\n')
				fname = arr_slash[srch].split(".")
				print arr_slash[srch]
				print "file name"
				print arr_slash[srch]
				print "title"
				#print title[fname[0]-1000]
				ty = int(fname[0])-1000
				QDetils.write("Paper Title " + title[ty] + '\n')
				#title.appen
				QDetils.write("File name " + arr_slash[srch] + '\n')
			
				QDetils.write(documents_full[arr_slash[srch]][ing[0]:ing[0]+100].replace('\n', ' '))
				QDetils.write('\n'+"====================================================================="+'\n')
			
				#print "Index "
				#print ing
				tt = tt + 1
				break
			except:
				tt=tt+1
				rt = 1
				print "mmeeet an exception"	

			#tt=tt+1
		srch=srch+1

	

	QDetils.close()

except:
	print "No synn present for words"


