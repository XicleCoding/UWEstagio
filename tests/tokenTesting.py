from nltk.tokenize import word_tokenize



def token():
    text = "Olá, tudo bem? Como vai você?"
    tokens = word_tokenize(text, language='portuguese')
    print(tokens)
    return
