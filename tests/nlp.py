import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from collections import defaultdict

def textToCategory(sentence,categoryList):
    tokens = nltk.word_tokenize(sentence.lower())
    stop_words = stopwords.words('english')
   
    words = [word for word in tokens if word not in string.punctuation and word not in stop_words]

    print(words)

    pos_tags = nltk.pos_tag(words)

    lemmatized_words = []
    lemmatizer = WordNetLemmatizer()

    for word, pos in pos_tags:
        if pos.startswith('J'):
            word_class = 'a'  
        elif pos.startswith('V'):
            word_class = 'v'  
        elif pos.startswith('N'):
            word_class = 'n'  
        elif pos.startswith('R'):
            word_class = 'r'  
        else:
            word_class = 'n'  

        lemmatized_word = lemmatizer.lemmatize(word, pos=word_class)
        lemmatized_words.append(lemmatized_word)

    wordFrequency = defaultdict(int)
    for word in lemmatized_words:
        wordFrequency[word]+=1

    print(wordFrequency)

def categoryExpand(word):
    categoryExpanded = []
    synsets = wordnet.synsets(word)
    
    for synset in synsets:
        for lemma in synset.lemmas():
            categoryExpanded.append(lemma.name())
            
    return sorted(list(set(categoryExpanded)))

def main():
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('floresta')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    
    sentence = "I would like to report an issue regarding the excessive noise coming from a construction site located on Elm Street. The construction work starts early in the morning and continues throughout the day, causing a significant disturbance to the surrounding residents. The noise levels are unbearable, making it challenging to concentrate, work from home, or even have a peaceful environment for our families. We kindly request the relevant authorities to address this issue and enforce stricter regulations on noise control for construction activities in residential areas."
    labels = ["Public Safety", "Infrastructure Issues", "Environmental Concerns", "Traffic and Transportation", "Noise and Nuisance", "Public Health", "Parks and Recreation", "Zoning and Land Use", "Community Events and Programs", "Civic Services"]
    textToCategory(sentence,labels)
   

    print(categoryExpand('safety'))
main()