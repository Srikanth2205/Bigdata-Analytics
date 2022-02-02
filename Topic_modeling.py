# Importing necessary libraries
if __name__ == '__main__':
    import nltk
    import gensim
    from nltk.corpus import reuters
    from nltk import sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from gensim.models import LdaModel
    from gensim.corpora import Dictionary
    from pprint import pprint
    from gensim.models import CoherenceModel
    import numpy as np
    import pyLDAvis.gensim_models
    import pandas as pd

    # Reuters-21578 corpus download along with necessary downloads

    # nltk.download('reuters')
    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download('averaged_perceptron_tagger')

    reuters.readme()  # readme file about the corpus

    '''Documents inside the corpus required for topic modeling'''
    data = reuters.raw()

    '''total number of documents in the corpus'''
    documents = reuters.fileids()
    print(len(documents))

    '''Breakdown of raw documents into sentences '''
    tokenized_data = nltk.tokenize.sent_tokenize(data)

    '''Process to remove words whose length is less than 4 characters'''
    text_data = []
    for i in tokenized_data:
        text_data.append(gensim.utils.simple_preprocess(i, deacc=True, min_len=4))

    '''Use english stop words'''
    stops = set(stopwords.words('english'))

    '''Add words 'said' to stop_words list'''
    stops.add("said")

    '''Process of removing stop words from each sentence'''
    stop_data = []
    for line in text_data:
        list_word = []
        for word in line:
            if word not in stops:
                list_word.append(word)
        stop_data.append(list_word)

    '''import Wordnet lemmatizer to convert words into one unique parts of speech'''
    lemmatizer = WordNetLemmatizer()

    '''Process of lemmatizing each words to its noun form'''
    final_data = []

    for line in stop_data:
        lem_data = []
        for word in line:
            lem_data.append(lemmatizer.lemmatize(word, pos='n'))
        final_data.append(lem_data)

    '''Tagging each word with a unique id'''
    id2word = Dictionary(final_data)

    '''Converting english words to its vector format'''
    texts = final_data
    corpus = [id2word.doc2bow(text) for text in texts]

    '''Random way of performing LDA'''
    num_topics = 5

    '''Build LDA model'''
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics, passes=2)

    '''Topics with their word distribution'''
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    '''Tuning LDA parameters based on coherence value'''
    coherence_model_lda = CoherenceModel(model=lda_model, texts=final_data, dictionary=id2word, coherence='c_v')
    lda_coherence = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', lda_coherence)

    '''Grid search for best hyper parameters'''

    '''Topics range definition'''
    topics_range = range(5, 11, 1)

    '''Alpha parameter range definition'''
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')

    '''Eta parameter range definition'''
    eta = list(np.arange(0.01, 1, 0.3))
    eta.append('symmetric')

    model_results = {'Topics': [],
                     'Alpha': [],
                     'Eta': [],
                     'Coherence': []
                     }

    '''Performing grid search'''
    for k in topics_range:
        for i in alpha:
            for j in eta:
                lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=k, random_state=50,
                                                       chunksize=50, passes=2, alpha=i, eta=j)
                coherence_model_lda = CoherenceModel(model=lda_model, texts=final_data, dictionary=id2word,
                                                     coherence='c_v')

                cv = coherence_model_lda.get_coherence()

                model_results['Topics'].append(k)
                model_results['Alpha'].append(i)
                model_results['Eta'].append(j)
                model_results['Coherence'].append(cv)

    print("Complete")

    '''Storing the grid search results to a file'''
    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)

    '''Pick parameters with high coherence value'''
    data = pd.read_csv('lda_tuning_results.csv')
    sorted_data = data.sort_values(by=['Coherence'], ascending=False).iloc[:1, :]
    num_topics = sorted_data["Topics"].values
    alpha_val = sorted_data["Alpha"].values
    eta_val = sorted_data["Eta"].values
    coherence_val = sorted_data["Coherence"].values

    print(" Number of Topics:", num_topics[0], "\n",
          "Best Alpha Value:", alpha_val[0], "\n",
          "Best Eta Value:", eta_val[0], "\n",
          "Coherence Score:", coherence_val[0])

    '''Performing LDA on found parameters'''
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=int(num_topics[0]),
                                           random_state=50,
                                           chunksize=50,
                                           passes=2,
                                           alpha=float(alpha_val[0]),
                                           eta=float(eta_val[0]))

    lda_coherence = CoherenceModel(model=lda_model, texts=final_data, dictionary=id2word, coherence='c_v')

    print(lda_coherence.get_coherence())

    '''Topics with their word distribution'''
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    '''Graphical representation of topics with their word distribution'''
    LDAvis_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(LDAvis_vis, 'task3_output.html')
