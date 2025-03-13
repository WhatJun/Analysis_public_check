'''
卒業研究意味距離算出用（変更あり）
'''

import itertools
from gensim import models
import pandas as pd


RAWDATA_PATH = './rawdata.csv'
SCOREDATA_PATH = './data_sum.csv'

# load model
model_path = './model/glove-retrofitting.bin'
model =  models.KeyedVectors.load_word2vec_format(model_path, binary=True)

# extract and process rawdata of dat task
df = pd.read_csv(RAWDATA_PATH)
dat_rawdata = df.iloc[1:, 75:85]
dat_data = dat_rawdata.applymap(lambda x: x.strip())

def calculator(words):
    
    # judgement of valid words
    vword_list = []
    for word in words:
        if word in model and word not in vword_list:     # check the word if both in-corpus and singular
            vword_list.append(word)
    # print(vword_list)

    # calculation of semantic distance
    minimum = 7 # at least and only use [minimum] words to calculate
    results = []
    nvw = len(vword_list)
    if nvw >= minimum:
        combinations = itertools.combinations(vword_list[:minimum],2)
        for word1, word2 in combinations:
                result = model.distance(word1, word2)
                results.append(result)
    
    # calculation of dat score
    if results:
        average_result = sum(results)/len(results)
        score = average_result*100
        return score
    else:
        return 0 # valid words do not match the minimum criterion


# calculation
scores = []
for row in dat_data.itertuples(index=False):
    score = calculator(row)
    scores.append(score)
    # print(score)


# ouput scores to datasum file
data_file = pd.read_csv(SCOREDATA_PATH)
data_file["DAT_score"] = scores
data_file.to_csv(SCOREDATA_PATH, index=False)
