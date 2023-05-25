# -*- encoding: utf-8 -*-

import torch
import nltk #Natural Language Toolkit: Zusammenstellung von Bibliotheken zur Sprachverarbeitung in Python
import numpy as np
from nltk.stem.porter import PorterStemmer  #Stemmer-Funktionalität einbinden

stemmer = PorterStemmer()

#print(torch.__version__)
#nltk.download('punkt') #Beim erstmaligen Ausführen Raute entfernen

def tokenize(sentence):                 #Funktion für Tokenization
    return nltk.word_tokenize(sentence)

def stem(word):                         #Funktion für Stemming
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):    #Sammlung von Wörtern zur Darstellung eines Satzes mit Wortanzahl ohne Berücksichtigung der Reihenfolge
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

def beispielTokenize():
    a = "Wie lange dauert die Technikerweiterbildung?"
    a = tokenize(a)
    print(a)

def beispielStemming():
    a = ["Organize","organizes","Organizing"]
    stemmed_words = [stem(w) for w in a]
    print(stemmed_words)

def beispielBagOfWords():
    satz = ["hallo", "wie", "gehts"] #Frage des Benutzers
    woerter = ["wie", "geht", "es", "dir", "gehts", "alles", "gut", "hallo"] #pattern für tag "gesundheit"
    bog = bag_of_words(satz, woerter) #3 Wörter stimmen überein, passt vermutlich zu diesem Pattern
    print(bog)




