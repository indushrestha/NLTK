
# Name:Indu Shrestha


## Import all of the libraries and data that we will need.
import nltk
import textblob
from nltk.corpus import names  # see the note on installing corpora, above
from nltk.corpus import opinion_lexicon
from nltk.corpus import movie_reviews

import random
import math

from sklearn.feature_extraction import DictVectorizer
import sklearn
import sklearn.tree
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
try: # different imports for different versions of scikit-learn
    from sklearn.model_selection import cross_val_score   # simpler cv this week
except ImportError:
    try:
        from sklearn.cross_validation import cross_val_score
    except:
        print("No cross_val_score!")





#####################
#
## Problem 4: Movie Review Sentiment starter code...
#
#####################

# a boolean to turn on/off the movie-review-sentiment portion of the code...
RUN_MOVIEREVIEW_CLASSIFIER = True
if RUN_MOVIEREVIEW_CLASSIFIER == True:

    ## Read all of the opinion words in from the nltk corpus.
    #
    pos=list(opinion_lexicon.words('positive-words.txt'))
    neg=list(opinion_lexicon.words('negative-words.txt'))

    ## Store them as a set (it'll make our feature extractor faster).
    # 
    pos_set = set(pos)
    neg_set = set(neg)



    ## Read all of the fileids in from the nltk corpus and shuffle them.
    #
    pos_ids = [(fileid, "pos") for fileid in movie_reviews.fileids('pos')]
    neg_ids = [(fileid, "neg") for fileid in movie_reviews.fileids('neg')]
    labeled_fileids = pos_ids + neg_ids

    ## Here, we "seed" the random number generator with 0 so that we'll all 
    ## get the same split, which will make it easier to compare results.
    random.seed(0)   # we'll use the seed for reproduceability... 
    random.shuffle(labeled_fileids)



    ## Define the feature function
    #  Problem 4's central challenge is to modify this to improve your classifier's performance...
    #
    def opinion_features(fileid):
        """ starter feature engineering for movie reviews... """
       
        rawtext = movie_reviews.raw(fileid)
        TB = textblob.TextBlob( rawtext )

        total_words=len(TB.words)
        total_sentence= len(TB.sentences)
        positive_count=0
        negative_count=0
        for i in range(len(TB.words)):
            if TB.words[i] in pos_set :
                if TB.words[i-1] in ['not','less','few',"isn't","hasn't","wasn't"] or TB.words[i-2] =='not':
                     negative_count += 1
                else:
                    positive_count += 1

            elif TB.words[i] in neg_set:
                if TB.words[i-1] in ['not','less','few',"isn't","hasn't","wasn't"] or TB.words[i-2] =='not':
                     positive_count += 1
                else:
                    negative_count += 1
        

        # Note:  movie_reviews.raw(fileid) is the whole review!
        # create a TextBlob with
        rawtext = movie_reviews.raw(fileid)
        TB = textblob.TextBlob( rawtext )
        # now, you can use TB.words and TB.sentences...

        # here is the dictionary of features...
        features = {}   # could also use a default dictionary!
        
        features['positive'] = positive_count
        features['negative'] = negative_count
        features['total words']= total_words
        features['total sentence']= total_sentence
        features['sentimence']=TB.sentiment.subjectivity
        features['polarity']=TB.sentiment.polarity
        #features['negative_r'] = negative_count//total_words
        #features['positive_r'] = positive_count//total_words
        return features


    #
    ## Ideas for improving this!
    #
    # count both positive and negative words...
    # is the ABSOUTE count what matters?
    # 
    # other ideas:
    #
    # feature ideas from the TextBlob library:
    #   * part-of-speech, average sentence length, sentiment score, subjectivity...
    # feature ideas from TextBlob or NLTK (or just Python):
    # average word length
    # number of parentheses in review
    # number of certain punctuation marks in review
    # number of words in review
    # words near or next-to positive or negative words: "not excellent" ?
    # uniqueness
    #
    # many others are possible...


    ## Extract features for all of the movie reviews
    # 
    print("Creating features for all reviews...", end="", flush=True)
    features = [opinion_features(fileid) for (fileid, opinion) in labeled_fileids]
    labels = [opinion for (fileid, opinion) in labeled_fileids]
    fileids = [fileid for (fileid, opinion) in labeled_fileids]
    print(" ... feature-creation done.", flush=True)


    ## Change the dictionary of features into an array
    #
    print("Transforming from dictionaries of features to vectors...", end="", flush=True)
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(features)
    print(" ... vectors completed.", flush=True)

    ## Split the data into train, devtest, and test

    X_test = X[:100,:]
    Y_test = labels[:100]
    fileids_test = fileids[:100]

    X_devtest = X[100:200,:]
    Y_devtest = labels[100:200]
    fileids_devtest = fileids[100:200]

    X_train = X[200:,:]
    Y_train = labels[200:]
    fileids_train = fileids[200:]

    ## Train the decision tree classifier - perhaps try others or add parameters
    #
    dt = sklearn.tree.DecisionTreeClassifier()
    dt.fit(X_train,Y_train)
    
    

    ## Evaluate on the devtest set; report the accuracy and also
    ## show the confusion matrix.
    #
    print("Score on devtest set: ", dt.score(X_devtest, Y_devtest))
    Y_guess = dt.predict(X_devtest)
    CM = confusion_matrix(Y_guess, Y_devtest)
    print("Confusion Matrix:\n", CM)

    ## Get a list of errors to examine more closely.
    #
    errors = []

    for i in range(len(fileids_devtest)):
        this_fileid = fileids_devtest[i]
        this_features = X_devtest[i:i+1,:]
        this_label = Y_devtest[i]
        guess = dt.predict(this_features)[0]
        if guess != this_label:
            errors.append((this_label, guess, this_fileid))

    PRINT_ERRORS = True 
    if PRINT_ERRORS == True:
        num_to_print = 15    # #15 is L.A. Confidential
        count = 0

        for actual, predicted, fileid in errors:
            print("Actual: ", actual, "Predicted: ", predicted, "fileid:", fileid)
            count += 1
            if count > num_to_print: break

    PRINT_REVIEW = True
    if PRINT_REVIEW == True:
        print("Printing the review with fileid", fileid)
        text = movie_reviews.raw(fileid)
        print(text)

    ## Finally, score on the test set:
    print("Score on test set: ", dt.score(X_test, Y_test))
    L=[]
    for max_depth in range(2,4):
        for n_estimators in range(100,150,20):
            rforest = ensemble.RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators)

            # an example call to run 5x cross-validation on the labeled data
            scores = cross_val_score(rforest, X_train, Y_train, cv=5)
            #print(max_depth,n_estimators,"CV scores:", scores)
            #print(max_depth,n_estimators,"CV scores' average:", scores.mean())
            L+=[[scores.mean(),max_depth,n_estimators]]
    print(max(L))




    

