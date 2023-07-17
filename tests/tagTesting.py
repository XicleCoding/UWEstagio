import spacy
import pt_core_news_sm



def tag():
    # Load the Portuguese language model
    nlp = spacy.load('pt_core_news_sm')

    # Text to analyze
    text = "Eu gosto de comer pizza."

    # Process the text with SpaCy
    doc = nlp(text)

    # Iterate over the tokens and print their part-of-speech tags
    for token in doc:
        print(token.text, token.pos_)

        return
    
def tag2():
    nlp = spacy.load("pt_core_news_sm")
    nlp = pt_core_news_sm.load()
    doc = nlp("A floresta onde me encontro está a arder.")
    print([(w.text, w.pos_) for w in doc])