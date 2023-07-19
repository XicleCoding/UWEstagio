import BERT_data_load 
from nlp_utils import *
import nltk



if __name__ == '__main__':
    #Downloads
    ##nltk.download('stopwords')
    ##nltk.download('wordnet')

    dtf = BERT_data_load.load()

    #Testing stop words generation
    lst_stopwords = nltk.corpus.stopwords.words("english")
    ##print('----------------STOP WORDS----------------\n')
    ##print(lst_stopwords)
    
    #Text preprocessing
    dtf["text_clean"] = dtf["text"].apply(lambda x:utils_preprocess_text(
                                                                            x,
                                                                            lst_regex=None,
                                                                            punkt=True,
                                                                            lower=True,
                                                                            slang=False,
                                                                            lst_stopwords=lst_stopwords,
                                                                            stemm=False, 
                                                                            lemm=True, 
                                                                        )) #txt, lst_regex=None, punkt=True, lower=True, slang=True, lst_stopwords=None, stemm=False, lemm=True
    ##print('----------------PREPROCESSED TEXT EXAMPLE----------------\n')
    ##print(dtf.head())

    #Target Clusters Creation
    ## Create Dictionary {category:[keywords]}
    nlp = gensim_api.load("glove-wiki-gigaword-300")
    dic_clusters = {}
    dic_clusters["ENTERTAINMENT"] = get_similar_words(['celebrity','cinema','movie','music'], top=30, nlp=nlp)
    dic_clusters["POLITICS"] = get_similar_words(['gop','clinton','president','obama','republican'], top=30, nlp=nlp)
    dic_clusters["TECH"] = get_similar_words(['amazon','android','app','apple','facebook','google','tech'], top=30, nlp=nlp)
    
    ## print some examples
    ##for k,v in dic_clusters.items():
    ##    print(f'{k} : {v[0:5]} ... {len(v)}')

    #plot_w2v_cluster(dic_clusters, nlp, plot_type="2d", annotate=True, figsize=(20,10))

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    nlp = transformers.TFBertModel.from_pretrained('bert-base-uncased')

    txt = "river bank"
    ## tokenize
    idx = tokenizer.encode(txt)
    print("tokens:", tokenizer.convert_ids_to_tokens(idx))
    print("ids   :", tokenizer.encode(txt))
    ## word embedding
    idx = np.array(idx)[None,:]
    embedding = nlp(idx)
    #print("shape:", embedding[0][0].shape)
    ## vector of the second input word
    #print(embedding[0][0][2])


    ## create list of news vector
    lst_mean_vecs = [utils_bert_embedding(txt, tokenizer, nlp).mean(0) 
                    for txt in dtf["text_clean"]]
    ## create the feature matrix (n news x 768)
    X = np.array(lst_mean_vecs)

