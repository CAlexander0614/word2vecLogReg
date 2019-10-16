import gensim.models.keyedvectors as word2vec
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
import re


model = word2vec.KeyedVectors.load_word2vec_format\
    ("GoogleNews-vectors-negative300.bin", binary=True)



def getVector(w):
    if w in model:
        return model[w]
    else:
        return np.zeros((300))


#removes punctuation
def removePunc(x):
    temp = r'[^a-zA-z0-9\s]'
    text = re.sub(temp, '', x)
    return text

#changes numbers to ##
def changeNums(x):
    if re.search(r'\d',x):
        x = re.sub('[0-9]{1,}', '#', x)
    return x

#complete preprocessing function
def textPreprocess(x):


    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE)              #removes hyperlinks
    text = removePunc(text)
    text = changeNums(text)
    text = re.sub(r'(\n)', " ", text)
    text = re.sub('\bu\b',"you",text)
    text = re.sub('\bur\b', "your", text)
    text = re.sub('\blol\b', "I am laughing", text)
    text = re.sub('\blmao\b', "I am hysterically laughing", text)
    text = re.sub('\bjk\b', "just kidding", text)
    text = re.sub('\bsmh\b', "shake my head",text)
    text = re.sub('\bnvm\b', "never mind", text)
    text = re.sub('\bofc\b', "of course", text)
    text = re.sub('\bikr\b', "I know right", text)
    text = re.sub('\btldr\b', "too long did not read", text)

    return text


#sums up all the word vectors for each comment
def sumComVec(text):

    sumVec = np.zeros(300)
    splits = text.split()
    for words in splits:
        if words in model:
            sumVec+=(getVector(words))
        else:
            continue

    return sumVec



def classifySubreddit_train(file):

    global logistic_reg_model

    logistic_reg_model = LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100000, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.00001,
          verbose=0, warm_start=False)


    redditComments = []

    all_comments = []

    subredditsY = []

    file = open(file,"r")

    #takes data from json file
    for lines in file:
        redditComments.append(json.loads(lines))


    #seperates comments and labels(subreddits)
    for comments in redditComments:
        all_comments.append(textPreprocess(comments["body"]))
        subredditsY.append(comments["subreddit"])


    featuresX = []

    #sums up all word vectors in a comment, adds vector to features matrix
    for comments in all_comments:
        x = sumComVec(comments)
        featuresX.append(x)

    featuresX = np.array(featuresX)

    logistic_reg_model.fit(featuresX,subredditsY)



def classifySubreddit_test(text):

    preText = textPreprocess(text)
    vector = [sumComVec(preText)]

    vector = np.array(vector)
    vector = vector.reshape(1,-1)
    return(logistic_reg_model.predict(vector)[-1])




