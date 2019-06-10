# -*- coding: utf-8 -*-

"""
 A baseline authorship attribution method 
 based on a character n-gram representation
 and a linear SVM classifier.
 It has a reject option to leave documents unattributed
 (when the probabilities of the two most likely training classes are too close)
 
 Questions/comments: stamatatos@aegean.gr

 It can be applied to datasets of PAN-19 cross-domain authorship attribution task
 See details here: http://pan.webis.de/clef19/pan19-web/author-identification.html
 Dependencies:
 - Python 2.7 or 3.6 (we recommend the Anaconda Python distribution)
 - scikit-learn

 Usage from command line: 
    > python pan19-cdaa-baseline.py -i EVALUATION-DIRECTORY -o OUTPUT-DIRECTORY [-n N-GRAM-ORDER] [-ft FREQUENCY-THRESHOLD] [-pt PROBABILITY-THRESHOLD]
 EVALUATION-DIRECTORY (str) is the main folder of a PAN-19 collection of attribution problems
 OUTPUT-DIRECTORY (str) is an existing folder where the predictions are saved in the PAN-19 format
 Optional parameters of the model:
   N-GRAM-ORDER (int) is the length of character n-grams (default=3)
   FREQUENCY-THRESHOLD (int) is the cutoff threshold used to filter out rare n-grams (default=5)
   PROBABILITY-THRESHOLD (float) is the threshold for the reject option assigning test documents to the <UNK> class (default=0.1)
                                 Let P1 and P2 be the two maximum probabilities of training classes for a test document. If P1-P2<pt then the test document is assigned to the <UNK> class.
   
 Example:
     > python pan19-cdaa-baseline.py -i "mydata/pan19-cdaa-development-corpus" -o "mydata/pan19-answers"
"""
#python logreg.py -i "2018_data" -o "output_logreg"
from __future__ import print_function
import os
import glob
import json
import argparse
import time
import codecs
from collections import defaultdict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from pan19_evaluator import evaluate_all
from sklearn.linear_model import LogisticRegression
import numpy as np
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


#FEATURES - change in code is necessary

#lemmatizer all words - implemented in train docs and test docs
def lemmatizer(texts):
    lemmatizer = WordNetLemmatizer()
    texts_lemma = []
    for (text, label) in texts:
        text = [lemmatizer.lemmatize(word) for word in text.split()]
        texts_lemma.append((str(text), label))
    return texts_lemma
    
#stemmer all words - implemented in train docs and test docs
def stemmer(texts):
    stemmer = PorterStemmer()
    texts_stemmer = []
    for (text, label) in texts:
        text = [stemmer.stem(word) for word in text.split()]
        texts_stemmer.append((str(text), label))
    return texts_stemmer
        
#pos-tagging - should be implemented in train_text and test_text
def pos_tagging(docs):
    function=[',', '.', "'", '"', '!']
    new_text=''
    new_list=[]
    for text in docs:
        new_text=''
        text = nltk.word_tokenize(text)
        tagged_text = nltk.pos_tag(text)
        for (word, pos) in tagged_text:
            if word != ',':
                new_text += word+'_'+pos+' '
            if word in function:
                new_text += word+' '
        new_list.append(new_text)
    return(new_list)

#delete words below 3 characters - should be inplemented in train_docs and test_docs
def delete_words(texts):
    texts_new = []
    for (text, label) in texts:
        text = [word for word in text.split() if len(word) < 5]
        texts_new.append((str(text), label))
    return texts_new

#FEATURES - add to union
    
#function to get lenght of a text
def get_text_length(x):
    return np.array([len(t) for t in x]).reshape(-1, 1)

#function to get mean length of words for each text
def get_word_length(x):
    means = []
    for text in x:
        count = 0
        text = text.split()
        for word in text:
            count += len(word)
        means.append(([(count/len(text))]))
    return means

#function for sentence length
def get_sentence_length(x):
    means = []
    for text in x:
        length = []
        text_split = text.split('.')
        for t in text_split:
            length.append(len(t))
        means.append([np.mean(length)])
    return means

#get lexical diversity
def lex_div(x):
    lexical_div = []
    for text in x:
        unique_words = []
        text = text.lower()
        for word in text:
            if word not in unique_words:
                unique_words.append(word)
        lexical_div.append([len(text)/len(unique_words)])
    return lexical_div
    
#get percentage of digits
def get_digits(x):
    total = []
    for text in x:
        t = sum(c.isdigit() for c in text)
        total.append([t/len(text)])
    return total

#get percentage of spaces
def get_space(x):
    total = []
    for text in x:
        t = sum(c.isspace() for c in text)
        total.append([t/len(text)])
    return total
            
 #function to get mean use of punctuation for each text
def get_punctuation(x):
    means = []
    for text in x:
        count = 0
        for character in text:
            if character in string.punctuation:
                count += 1
        means.append(([(count/len(text))]))
    return means

 #function to get mean use of "-" for each text
def get_dash(x):
    means = []
    for text in x:
        count = 0
        for character in text:
            if character == '-':
                count += 1
        means.append(([(count/len(text))]))
    return means

 #function to get mean use of punctuation for each text
def get_apo(x):
    means = []
    for text in x:
        count = 0
        for character in text:
            if character == "'":
                count += 1
        means.append(([(count/len(text))]))
    return means

 #function to get mean use of " in a text
def get_quote(x):
    means = []
    for text in x:
        count = 0
        for character in text:
            if character == '"':
                count += 1
        means.append(([(count/len(text))]))
    return means

 #function to get mean use of parenteses for each text
def get_paren(x):
    means = []
    for text in x:
        count = 0
        for character in text:
            if character == '(' or ')':
                count += 1
        means.append(([(count/len(text))]))
    return means

#get percentage of the usage of upper-case letters
def get_uppercase(x):
    means = []
    for text in x:
        count = 0
        for word in text:
            if word.isupper():
                count += 1
        means.append(([(count/len(text))]))
    return means

#get percentage of the use of adjectives in a text
def adjectives(x):
    counts = []
    for text in x:
        adj = [word for word, pos in nltk.pos_tag(nltk.word_tokenize(text)) if pos.startswith('JJ')]
        count = len(adj)
        counts.append([(count/len(text))*100])
    return counts

#get percemtage of the use of personal pronoouns in a text
def personal_pronouns(x):
    counts = []
    for text in x:
        adj = [word for word, pos in nltk.pos_tag(nltk.word_tokenize(text)) if pos.startswith('PRP')]
        count = len(adj)
        #print(count)
        counts.append([(count/len(text))*100])
    return counts

#get percentage of the use of foreing words
def foreignword(x):
    counts = []
    for text in x:
        fw = [word for word, pos in nltk.pos_tag(nltk.word_tokenize(text)) if pos.startswith('FW')]
        count = len(fw)
        #print(count)
        counts.append([(count/len(text))*100])
    return counts

def represent_text(text,n):
    # Extracts all character 'n'-grams from  a 'text'
    if n>0:
        tokens = [text[i:i+n] for i in range(len(text)-n+1)]
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    return frequency

def read_files(path,label):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(path+os.sep+label+os.sep+'*.txt')
    texts=[]
    for i,v in enumerate(files):
        f=codecs.open(v,'r',encoding='utf-8')
        texts.append((f.read(),label))
        f.close()
    return texts

def extract_vocabulary(texts,n,ft):
    # Extracts all characer 'n'-grams occurring at least 'ft' times in a set of 'texts'
    occurrences=defaultdict(int)
    for (text,label) in texts:
        text_occurrences=represent_text(text,n)
        for ngram in text_occurrences:
            if ngram in occurrences:
                occurrences[ngram]+=text_occurrences[ngram]
            else:
                occurrences[ngram]=text_occurrences[ngram]
    vocabulary=[]
    for i in occurrences.keys():
        if occurrences[i]>=ft:
            vocabulary.append(i)
    return vocabulary

def baseline(path,outpath,n=3,ft=5,pt=0.1):
    start_time = time.time()
    # Reading information about the collection
    infocollection = path+os.sep+'collection-info.json'
    problems = []
    language = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])
    for index,problem in enumerate(problems):
        print(problem)
        # Reading information about the problem
        infoproblem = path+os.sep+problem+os.sep+'problem-info.json'
        candidates = []
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])
        # Building training set
        train_docs=[]
        for candidate in candidates:
            train_docs.extend(read_files(path+os.sep+problem,candidate))
       
        #train_docs = delete_words(train_docs)
        train_texts = [text for i,(text,label) in enumerate(train_docs)]


        train_labels = [label for i,(text,label) in enumerate(train_docs)]
        vocabulary = []
        vocabulary += extract_vocabulary(train_docs,n,ft) 
        if n >= 2:
            vocabulary += extract_vocabulary(train_docs,n-1,ft)
        if n == 3:
            vocabulary += extract_vocabulary(train_docs,n-2,ft)
            
      #features
        vectorizer_count_char = CountVectorizer(analyzer = 'char', ngram_range=(1,n), min_df = 3, max_df = 10, lowercase = True, vocabulary = vocabulary) #does not make a difference to add tokenizer to char
        vectorizer_count_word = CountVectorizer(analyzer = 'word', ngram_range=(1,n), min_df = 3, max_df = 10,  lowercase=True) #tokenizer=nltk.word_tokenize, 
        vectorizer_tfidf_char = TfidfVectorizer(analyzer = 'char', ngram_range = (1,n), use_idf=False, lowercase = False, vocabulary = vocabulary)
        vectorizer_tfidf_word = TfidfVectorizer(analyzer = 'word', ngram_range = (1,n), use_idf=False, lowercase = False)
        #lex = FunctionTransformer(lex_div, validate = False)
        #text_length = FunctionTransformer(get_text_length, validate=False) 
        #sentence_length = FunctionTransformer(get_sentence_length, validate = False)
        #word_length = FunctionTransformer(get_word_length, validate=False) 
        #digits = FunctionTransformer(get_digits, validate = False)
        #space = FunctionTransformer(get_space, validate = False)
        #punctuation = FunctionTransformer(get_punctuation, validate=False) 
        #uppercase = FunctionTransformer(get_uppercase, validate = False)
        apo = FunctionTransformer(get_apo, validate = False)
        #quote = FunctionTransformer(get_quote, validate = False)
        #dash = FunctionTransformer(get_dash, validate = False)
        paren = FunctionTransformer(get_paren, validate = False)
        #adjective = FunctionTransformer(adjectives, validate = False)
        #foreign_word = FunctionTransformer(foreignword, validate = False)
        #personal = FunctionTransformer(personal_pronouns, validate = False)


       # vectorizer_all_features = FeatureUnion([('count_char', vectorizer_count_char), ('count_word', vectorizer_count_word),
        #                           ('text_length', text_length), ('sentence_length', sentence_length), ('word_length', word_length),
         #                          ("lex", lex), ("digits", digits), ('space', space), ("punctuation", punctuation),
          #                         ("uppercase", uppercase), ('apo', apo), ('quote', quote), ('paren', paren),
           #                        ("personal", personal),("foreign word", foreign_word)])
           
        #collected featues in vectorizer
        vectorizer = FeatureUnion([('count_char', vectorizer_count_char), ('count_word', vectorizer_count_word),
                               ('tfidf_char', vectorizer_tfidf_char), ('tfidf_word', vectorizer_tfidf_word), 
                            ('apo', apo),('paren', paren)])

    
        
        train_data = vectorizer.fit_transform(train_texts) 
        train_data = train_data.astype(float)
        for i,v in enumerate(train_texts):
            train_data[i]=train_data[i]/len(train_texts[i])
        print('\t', 'language: ', language[index])
        print('\t', len(candidates), 'candidate authors')
        print('\t', len(train_texts), 'known texts')
        print('\t', 'vocabulary size:', len(vocabulary))
        # Building test set
        test_docs=read_files(path+os.sep+problem,unk_folder)
     
        #test_docs = delete_words(test_docs)
        test_texts = [text for i,(text,label) in enumerate(test_docs)]

        test_data = vectorizer.transform(test_texts)
        test_data = test_data.astype(float)
        for i,v in enumerate(test_texts):
            test_data[i]=test_data[i]/len(test_texts[i])
        print('\t', len(test_texts), 'unknown texts')
        # Applying SVM
        max_abs_scaler = preprocessing.MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)
        #Skift den her ud med en anden classifier
        clf=CalibratedClassifierCV(OneVsRestClassifier(LogisticRegression(C=1)))
        clf.fit(scaled_train_data, train_labels)
        predictions=clf.predict(scaled_test_data)
        proba=clf.predict_proba(scaled_test_data)
        # Reject option (used in open-set cases)
        count=0
        for i,p in enumerate(predictions):
            sproba=sorted(proba[i],reverse=True)
            if sproba[0]-sproba[1]<pt:
                predictions[i]=u'<UNK>'
                count=count+1
        print('\t',count,'texts left unattributed')
        # Saving output data
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open(outpath+os.sep+'answers-'+problem+'.json', 'w') as f:
            json.dump(out_data, f, indent=4)
        print('\t', 'answers saved to file','answers-'+problem+'.json')
    print('elapsed time:', time.time() - start_time)

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='PAN-19 Baseline Authorship Attribution Method')
    parser.add_argument('-i', type=str, help='Path to the main folder of a collection of attribution problems')
    parser.add_argument('-o', type=str, help='Path to an output folder')
    parser.add_argument('-n', type=int, default=3, help='n-gram order (default=3)')
    parser.add_argument('-ft', type=int, default=5, help='frequency threshold (default=5)')
    parser.add_argument('-pt', type=float, default=0.1, help='probability threshold for the reject option (default=0.1')
    args = parser.parse_args()
    if not args.i:
        print('ERROR: The input folder is required')
        parser.exit(1)
    if not args.o:
        print('ERROR: The output folder is required')
        parser.exit(1)
    
    baseline(args.i, args.o, args.n, args.ft, args.pt)

    evaluate_all('2018_data', 'output_logreg', 'evaluation_logreg')

if __name__ == '__main__':
    main()