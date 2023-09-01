import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
import nltk
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import concurrent.futures
import pandas as pd
from multiprocessing import Pool, freeze_support
from functools import partial

def tokenizer(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(sentence.lower())
    stop_words = stopwords.words('english')
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens if word not in string.punctuation]
    sentence = ' '.join(words)
    return sentence

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def text_summarization(input_text, max_length=150):
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    input_text = "summarize: " + input_text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    summary_ids = model.generate(input_ids, max_length=max_length, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def process_sentence(sentence):
    summarized = text_summarization(tokenizer(sentence), 180)
    return summarized

def main():
    n = 100
    df = pd.read_csv('C:/Users/hugor/OneDrive/Ambiente de Trabalho/Universidade/Estagio Verao Ubiwhere/UWEstagio/tests/datasets/US_stocks.csv')
    labels = df['Sector'].unique()
    df = df.head(n=n)
    sentences = df['Description']

    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')

    num_processes = 4
    with Pool(num_processes) as pool:
        process_sentence_partial = partial(process_sentence)
        processed_sentences = pool.map(process_sentence_partial, sentences)

    classifier = transformers.pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=0) 
    prediction = [classifier(sentence, labels, multi_label=True) for sentence in processed_sentences]
  
    predicted_label=[prediction[i]['labels'][0] for i in range(df.shape[0])]
    df['Predicted Label'] = predicted_label

    accuracy = accuracy_score(df['Sector'], df['Predicted Label'])
    precision = precision_score(df['Sector'], df['Predicted Label'],average='weighted')
    recall = recall_score(df['Sector'], df['Predicted Label'],average='weighted')
    f1 = f1_score(df['Sector'], df['Predicted Label'],average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

if __name__ == "__main__":
    freeze_support()  
    main()