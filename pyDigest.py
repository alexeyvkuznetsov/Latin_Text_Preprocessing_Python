def similar(id, corpus, size=10):
    '''The function returns the most similar documents to the one passed into based on cosine similarity
    calculated on the Tfidf matrix of a given corpus
    id: the index of the "document" in the corpus queried for its most similar documents
    corpus: a list of plain word strings ("documents"), the position of the "document" in the list is the
    id where indexing runs from 0 until len(corpus)-1
    size: the number of documents returned, default value is set to 10.'''
    # Handle errors
    valid_id = range(len(corpus))
    if id not in valid_id:
        raise ValueError("id must be in the range of len(corpus) which is between 0 and %r." % len(corpus))
    if type(corpus) != list:
        raise TypeError("corpus must be a plain list of word strings.")
    if type(size) != int:
        raise TypeError("size must be an integer")
    valid_size = range(1, (len(corpus)-1))
    if size not in valid_size:
        raise ValueError("size must be between 1 and %r." % (len(corpus)-1))
    # Import modules and initilaize models
    from sklearn.metrics.pairwise import linear_kernel              
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    # Calculate Tfidf matrix (X) and cosine similarity matrix (cosine_sim)
    X = vectorizer.fit_transform(corpus)
    cosine_sim = linear_kernel(X, X)
    # Calculate most similar documents
    sim_scores = list(enumerate(cosine_sim[id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(size + 1)]
    return sim_scores

def similar_sections(id, size=10):
    '''Returns a dataframe with the most similar thematic sections
    id: thematic section's id
    size: number of similar thematic sections returned
    '''
    import pandas as pd
    path_sID = 'https://raw.githubusercontent.com/mribary/pyDigest/master/input/Ddf_Section_IDs_v001.csv'
    path_doc = 'https://raw.githubusercontent.com/mribary/pyDigest/master/input/D_doc_sections_001.csv'
    path_df = 'https://raw.githubusercontent.com/mribary/pyDigest/master/input/Ddf_v105.csv'
    sID = pd.read_csv(path_sID, index_col=0)        # sections with section IDs (432)
    doc_df = pd.read_csv(path_doc, index_col=0)
    df = pd.read_csv(path_df, index_col=0)          # text units (21055)
    corpus = list(doc_df.doc)
    similar_to_id = similar(id, corpus, size)
    similar_dict_id = {'Section_id':[], 'Book_no':[], 'Section_no':[], 'Section_title':[], 'Similarity_score':[]}
    for i in range(size):
        section_id = similar_to_id[i][0]
        text_unit_id = sID.loc[sID.Section_id == section_id].index[0]
        book_no = df.loc[df.index == text_unit_id,'Book_no'].values[0]
        section_no = df.loc[df.index == text_unit_id,'Section_no'].values[0]
        section_title = df.loc[df.index == text_unit_id,'Section_title'].values[0]
        similarity_score = similar_to_id[i][1]
        similar_dict_id['Section_id'].append(section_id)
        similar_dict_id['Book_no'].append(book_no)
        similar_dict_id['Section_no'].append(section_no)
        similar_dict_id['Section_title'].append(section_title.lower())
        similar_dict_id['Similarity_score'].append(similarity_score)
    similar_df_id = pd.DataFrame(similar_dict_id)
    title = doc_df.Title[id]
    print("Thematic sections most similar to thematic section %r:" %id)
    print("%r" %title)
    return similar_df_id

def linkage_for_clustering(X, threshold=0.0):
    ''' The function takes a matrix X with observations stored in rows and features stored in columns.
    It returns a dataframe with linkage combinations of method and metric used for hierarchical
    clustering sorted by reverse order based on the absolute value of the cophenetic correlation
    coefficient (CCC). The CCC score ranges between -1 and 1 and measures how how faithfully a
    dendrogram preserves the pairwise distances between the original unmodeled data points.
    The cophenetic correlation is expected to be positive if the original distances are compared
    to cophenetic distances (or similarities to similarities) and negative if distances are
    compared to similarities.
    It needs to be noted that CCC is calculated for the whole dendrogram. Ideally, one should
    calculate CCC at the specific cut point where the dendrogram's output is used to identify the
    clusters. It is recommended to calculate CCC at the specific cut level yielding k clusters to
    confirm that the correct method-metric combination has been used for hierarchical clustering.
    The 'average' method generally produces the best CCC score especially with matrices with high
    dimensionality. Instead of relying exclusively on the CCC score, one also needs to consider 
    what method-metric combination suits the particular dataset on which hierarchical clustering 
    is performed by scipy's linkage function.
    '''
    import numpy as np
    # Handle errors
    if isinstance(X, np.ndarray) is not True:
        raise TypeError("X must be a matrix with samples in rows and observations in columns")
    if type(threshold) != float:
        raise TypeError("threshold must be a float")
    if abs(threshold) > 1:
        raise ValueError("threshold must be between -1 and 1")
    # Import basic packages
    import pandas as pd
    from scipy.cluster.hierarchy import linkage
    # List of 7 methods for the linkage function
    methods = ['ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median']
    # List of 22 metrics for the linkage function
    metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', \
        'dice',  'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', \
        'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', \
        'sokalsneath', 'sqeuclidean', 'yule']
    # Create list of dictioanries for the 154 method-metric combinations
    dicts = []
    for x in methods:
        for y in metrics:
            d = {'method':x, 'metric':y}
            dicts.append(d)
    # Load combinations into a dataframe
    linkages = {'method':[], 'metric': []}
    for i in range(len(dicts)):
        linkages['method'].append(dicts[i]['method'])
        linkages['metric'].append(dicts[i]['metric'])
    l = pd.DataFrame(linkages, columns=['method', 'metric'])
    # Calculate linkage matrices (Z) from X
    Z_matrices = []
    valid_mms = []
    for i in range(len(dicts)):
        try:
            Z = linkage(X, method=dicts[i]['method'], metric=dicts[i]['metric'])
            Z_matrices.append(Z)
            valid_mms.append(True)
            print('|', end= '')
        except:
            valid_mms.append(False)
            pass
    # Drop invald combinations and reindex
    l = l.loc[valid_mms]
    l.reset_index(drop=True)
    # Calculate Cophenetic Correlation Coefficient for valid linkage combinations
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist
    valid_scores = []
    CCC_scores = []
    for Z in Z_matrices:
        try:
            c, coph_dists = cophenet(Z, pdist(X))
            if np.isnan(c):
                valid_scores.append(False)
                CCC_scores.append(None)
            else:
                valid_scores.append(True)
                CCC_scores.append(c)
                print('|', end= '')
        except RuntimeWarning:
            valid_scores.append(False)
            pass
    # Insert scores, drop no values and reset index
    l['CCC_score'] = CCC_scores
    l['CCC_abs_score'] = [abs(number) if number is not None else number for number in CCC_scores]
    l = l.loc[valid_scores]
    l.reset_index(drop=True)
    # Sort method-metric pairs according to CCC score
    l.sort_values(by=['CCC_score', 'method', 'metric'], ascending=False, inplace=True)
    return l[l.CCC_score > threshold]

def latin_lemma_text(list_of_texts, stopwords=None):
    '''
    Create a list of continuous lemma texts for Latin with cltk (prerequisite).
       
    list_of_texts: raw text items stored in a list object
    stopwords: list of stopwords to be removed, default is None where nothing is removed
    
    Latin lemmatizer is cltk's BackoffLatinLemmatizer. Install, import and load before using the function
    '''

    # Import packages and models from cltk and initialize tools
    from cltk.corpus.utils.importer import CorpusImporter
    from cltk.lemmatize.latin.backoff import BackoffLatinLemmatizer
    corpus_importer = CorpusImporter('latin')                           # Initialize cltk's CorpusImporter
    corpus_importer.import_corpus('latin_models_cltk')                  # Import the latin_models_cltk corpus
    lemmatizer = BackoffLatinLemmatizer()                               # Initialize Latin lemmatizer
        
    import re
    punctuation = r"[\"#$%&\'()*+,-/:;<=>@[\]^_`{|}~.?!«»]"             # Punctuation pattern
    a = []
    for i in range(len(list_of_texts)):
        text = str(list_of_texts[i])
        new_text = ''.join(["" if ord(i) < 32 or ord(i) > 126 else i for i in text])                                                               # Remove Greek (non-ASCII) characters
        text_no_punct = re.sub(punctuation, '', new_text)               # Remove punctuation
        text_one_white_space = re.sub(r"\s{2,}", ' ', text_no_punct)    # Leave only one white space b/w words
        text_no_trailing_space = text_one_white_space.strip()           # Remove trailing white space
        text_lower = text_no_trailing_space.lower()                     # Transform to all lower case
        text_split = text_lower.split(' ')                              # Split to a list of tokens
        lemmas = lemmatizer.lemmatize(text_split)                       # Lemmatize
        textunit = ''                                                   # Empyt string for textunti
        for y in range(len(lemmas)):
            if stopwords is not None:
                if lemmas[y][1] not in stopwords:
                    textunit = textunit + str(lemmas[y][1] + ' ')
            else:
                textunit = textunit + str(lemmas[y][1] + ' ')
        textunit = textunit.strip()
        a.append(textunit)                                           # Add the "document" to a list
    return a

def tmp_download(url):
    """
    Create a temporary path for the download in /tmp.
    The file is cleared from /tmp at the next reboot.
    The default behaviour depends on system settings.
    """
    import os
    import requests
    import sys
    import random
    import string

    baseFile = os.path.basename(url)

    uuid_path = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(10)])

    temp_path = "/tmp"
    temp_path_uniq = os.path.join(temp_path, uuid_path)
    os.mkdir(temp_path_uniq)

    download_path = os.path.join(temp_path_uniq, baseFile)

    total_size = int(requests.get(url, stream=True).headers['Content-length'])

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(download_path, 'wb') as f:
            count = total_size
            for chunk in r.iter_content(chunk_size=8192):
                count = count - sys.getsizeof(chunk)
                print(str(baseFile) + ' : ' + \
                    str("%.2f" % round(((count / total_size)*100), 2)) \
                    + '%' + ' remaining ', end='\r')
                f.write(chunk)

    print('\nDownload is complete\nThe file is available at:\n' + str(download_path))
    print('\nMove the file to a permanent location, if you wish to keep it.\n')
    return download_path

def eval(model):
    '''
    The function takes a Latin gensim-FastText object and prints the TOEFL-style
    synonym evaluation based on LiLa's benchmark.
    '''
    # Load benchmark TOEFL synonyms
    import pandas as pd
    benchmark_path = 'https://embeddings.lila-erc.eu/samples/syn/syn-selection-benchmark-Latin.tsv' 
    benchmark = pd.read_csv(benchmark_path, sep='\t', header=None)

    from math import isnan
    true = 0
    total = 0
    for j in range(len(benchmark)):
        check = model.wv.most_similar(benchmark.iloc[j][0])
        if isnan(check[0][1]):
            next
        else:
            total += 1
            scores = []
            for i in range(1, 5):
                source = benchmark.iloc[j][0]
                target = benchmark.iloc[j][i]
                score = model.wv.similarity(w1=source, w2=target)
                scores.append(score)
            if scores[0] == max(scores):
                true += 1
    print('Number of term(s) missing from the model and removed from evaluation: ' + str(len(benchmark) - total))
    print(str(round((true/total)*100, 2)) + "% matches LiLa's synonyms")