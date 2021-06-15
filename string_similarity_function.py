def string_similarity_scoring(gunnison, gsa):
    import numpy as np
    import pandas as pd
    import spacy
    import textdistance
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import re
    
    # read in data
    gsa = pd.read_csv(gsa)
    gunnison = pd.read_csv(gunnison)
    
    datasets = [gsa, gunnison]
    
    # Loop through each dataset to perform preprocessing
    counter = 1
    for df in datasets:
        df = df[df.role.notnull()]
        
    
   #'''-----------------------------------------Preprocess Strings-----------------------------------------'''

        # Take out role in functional responsibility
        without_role = []
        
        for r,fr in zip(df.role, df['functional_responsibility']):
            without_role.append(fr.replace(str(r),''))
        
        
        # Preprocess data for normalization
        preprocessed = []
        
        for row in without_role:
            phrase = re.sub('[\n]+',' ', row) # Substitute new line characters for spaces
            phrase = phrase.lower() # Lowercase all values
            phrase = re.sub('[^A-Za-z0-9]+', ' ', phrase) # remove special characters
            phrase = re.sub('ing', ' ', phrase) # remove ing
            phrase = re.sub(r'[0-9]+', ' ', phrase) # remove numbers
            phrase = re.sub(' +', ' ', phrase) # remove double spaces
            preprocessed.append(phrase)
            
            
        # Assign new data to a column and strip spaces
        df['responsibility'] = preprocessed
        df['responsibility'] = df.responsibility.str.strip()
        
        
   #'''-----------------------------------------Decontract Words-----------------------------------------'''

        # Try and think of ways to expand contractions
        def decontracted(phrase):
            # specific
            phrase = re.sub(r"won\'t", "will not", phrase)
            phrase = re.sub(r"can\'t", "can not", phrase)
        
            # general
            phrase = re.sub(r"n\'t", " not", phrase)
            phrase = re.sub(r"\'re", " are", phrase)
            phrase = re.sub(r"\'s", " is", phrase)
            phrase = re.sub(r"\'d", " would", phrase)
            phrase = re.sub(r"\'ll", " will", phrase)
            phrase = re.sub(r"\'t", " not", phrase)
            phrase = re.sub(r"\'ve", " have", phrase)
            phrase = re.sub(r"\'m", " am", phrase)
            return phrase
        
        # Apply contraction
        contracted = []
        for row in df.responsibility:
            contracted.append(decontracted(row))
            
        # reassign contraction
        df['responsibility'] = contracted
        
        
   #'''-----------------------------------------Remove Stop Words-----------------------------------------'''

        # Place stop words in list
        my_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                        'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
                        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                        "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
                        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
                        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                        'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                        'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                        'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
                        'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
                        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
                        'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
                        "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 
                        've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', 
                        "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
                        'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
                        "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
                        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
        
        # Create stopword removal function
        def remove_mystopwords(sentence):
            text_tokens = sentence.split(" ")
            tokens_filtered= [word for word in text_tokens if not word in my_stopwords]
            return (" ").join(tokens_filtered)
        
        # remove stopwords
        df['responsibility'] = df.responsibility.apply(remove_mystopwords)
        df['responsibility'] = df.responsibility.str.strip()

        
   #'''-----------------------------------------Apply Lemmatization-----------------------------------------'''

        # Apply lemmatization in order to get the base form of a word
        lemmatized = []
        nlp = spacy.load('en_core_web_sm') # load spacy model, can be "en_core_web_sm"
        
        for text in df['responsibility']:
            doc = nlp(text)
            # Lemmatizing each token
            mytokens = [word.lemma_ if word.lemma_ != "-PrON-" else word.lower_ for word in doc]
            join_list = (' ').join(mytokens) # Join list of words into a sentence
            lemmatized.append(join_list) # Append to a list
            
        df['responsibility'] = lemmatized
        
        # Assign dataframes to variables
        if counter == 1:
            gsa = df
        else:
            gunnison = df
        
        counter += 1
        
        
   #'''-----------------------------------------Create Table of Confidence Scores-----------------------------------------'''

    role_list = []
    c_role_list = []
    resp_list = []
    f_resp_list = []
    c_resp_list = []
    cf_resp_list = []
    c_company_list = []
    jaccard_index_list = []
    sorensen_coefficient_list = []
    cosine_similarity_list = []
    ratcliff_obershelp_similarity_list = []
    
    for role, resp, f_resp in zip(gunnison.role, gunnison.responsibility, gunnison.functional_responsibility):
        for c_role, c_resp, cf_resp, c_company in zip(gsa.role, gsa.responsibility, gsa.functional_responsibility, gsa.company):
            
            # Token Based Algorithms - comparing token/word differences
            token_1 = resp.split()
            token_2 = c_resp.split()
            
            jaccard_index = textdistance.jaccard(token_1, token_2)
            sorensen_coefficient = textdistance.sorensen(token_1, token_2)
            cosine_similarity = textdistance.cosine(token_1, token_2)
            
            # Sequence Based Algorithm - comparing order words are in
            string_1 = resp
            string_2 = c_resp
            
            ratcliff_obershelp_similarity = textdistance.ratcliff_obershelp(string_1, string_2)
            
            # Extract from comparing responsibility: YOE, education, rates, company
            
            # Append all to list
            role_list.append(role)
            c_role_list.append(c_role)
            resp_list.append(resp)
            f_resp_list.append(f_resp)
            cf_resp_list.append(cf_resp)
            c_resp_list.append(c_resp)
            c_company_list.append(c_company)
            jaccard_index_list.append(jaccard_index)
            sorensen_coefficient_list.append(sorensen_coefficient)
            cosine_similarity_list.append(cosine_similarity)
            ratcliff_obershelp_similarity_list.append(ratcliff_obershelp_similarity)
            
            
   #'''-----------------------------------------Create Dataframe-----------------------------------------'''

    # Create columns using dictionary
    df_dict = {'gunnison_role': role_list, 'comparing_role': c_role_list,  
               'comparing_company': c_company_list,
               'gunnison_func_resp': f_resp_list,
               'comparing_func_resp': cf_resp_list,
               'gunnison_processed_resp':resp_list, 
               'comparing_processed_resp': c_resp_list, 
               'tb_jaccard_index': jaccard_index_list, 
               'tb_sorensen_coefficient': sorensen_coefficient_list, 
               'tb_cosine_similarity': cosine_similarity_list, 
               'sb_ratcliff_obershelp_similarity': ratcliff_obershelp_similarity_list}
    
    # Convert to dataframe
    string_similarity = pd.DataFrame(df_dict)
    string_similarity['average'] = string_similarity.iloc[:,7:].transpose().mean().transpose()
    
    string_similarity.to_csv('GSA_only/string_similarity_score.csv')
    
    return string_similarity