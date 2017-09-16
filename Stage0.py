import re, collections
import nltk
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

NWORDS = train(words(file('big.txt').read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)



#get all the document which are appended and then proces sthem up
corpusdir = '/home/abc/Desktop/adm/new_dataset'
corpusdir1 = '/home/abc/Desktop/adm/new_dataset_Inv'
newcorpus = PlaintextCorpusReader(corpusdir, '.*')
sortedall = sorted(newcorpus.fileids())
#print sortedall
documents = []
for filename in sortedall:
	print "started with processing of file " + filename
	name = corpusdir + "/" + filename
	namenew = corpusdir1 + "/" + filename
	raw = open(name,'r').read()
	rawnew = open(namenew,'w')
	#raw = raw.replace("\n","")

	raw = raw.replace("-\r\n","")
	raw = raw.replace("-\n","")
	raw1 = open(name,'w')
	raw1.writelines(raw)
	raw1.close()

	raw = open(name,'r').read()
	#raw = raw.replace("\n","")



	ts = nltk.wordpunct_tokenize(raw)
	tokens = [correct(w) for w in ts]
	porter = nltk.PorterStemmer()
	P=[porter.stem(t) for t in tokens if t.isalpha()]
	text = nltk.Text(P)
	words = [w.lower()+" " for w in text]
	rawnew.writelines(words)

	tokens = [w+" " for w in ts]
	raw1 = open(name,'w')
	raw1.writelines(tokens)
	raw1.close()
	print "done with processing of file " + filename
	rawnew.close()




#print tokens
