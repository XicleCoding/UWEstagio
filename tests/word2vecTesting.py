from gensim.models import Word2Vec
import nltk



def word2vec():

    # Corpus of Portuguese text
    corpus = [
        "O gato é preto",
        "O cachorro é marrom",
        "O céu está azul",
        "A grama é verde",
        "Eu gosto de sorvete"
    ]

    # Preprocessing the text and tokenizing
    tokenized_corpus = [nltk.word_tokenize(sentence.lower()) for sentence in corpus]

    # Training the Word2Vec model
    model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

    # Accessing word vectors
    word_vectors = model.wv

    # Example usage: Finding most similar words
    similar_words = word_vectors.most_similar("gato", topn=3)
    print(similar_words)
    return


def word2vecsg():
        
    # Corpus of Portuguese text
    corpus = [
        "O gato é preto",
        "O cachorro é marrom",
        "O céu está azul",
        "A grama é verde",
        "Eu gosto de sorvete"
    ]

    # Preprocessing the text and tokenizing
    tokenized_corpus = [nltk.word_tokenize(sentence.lower()) for sentence in corpus]

    # Training the Word2Vec model with skip-gram architecture
    model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, sg=1, workers=4)

    # Accessing word vectors
    word_vectors = model.wv

    # Example usage: Finding most similar words
    similar_words = word_vectors.most_similar("gato", topn=3)
    print(similar_words)

    return



def word2veccbow():

    # Corpus of Portuguese text
    corpus = [
        "O gato é preto",
        "O cachorro é marrom",
        "O céu está azul",
        "A grama é verde",
        "Eu gosto de sorvete"
    ]

    # Preprocessing the text and tokenizing
    tokenized_corpus = [nltk.word_tokenize(sentence.lower()) for sentence in corpus]

    # Training the Word2Vec model with CBOW architecture
    model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, sg=0, workers=4)

    # Accessing word vectors
    word_vectors = model.wv

    # Example usage: Finding most similar words
    similar_words = word_vectors.most_similar("gato", topn=3)
    print(similar_words)
    return
