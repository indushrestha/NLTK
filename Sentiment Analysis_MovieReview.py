
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



    # 
    # ## Reflections/Analysis
    #
    # Include a short summary of
    #   (a) how well your final set of features did!
    """
        I used many features but later I deciced to be more precise 
        with total of six features. It gives the accuracy of 60 percent in 
        devtest and above 71 percent on test data which i think is good since 
        I tried so many times with different features and this gives the good result. 
        I have posted one of the output below
    """

    """
        run hw6pr3movies.py
        Creating features for all reviews... ... feature-creation done.
        Transforming from dictionaries of features to vectors... ... vectors completed.
        Score on devtest set:  0.6
        Confusion Matrix:
        [[22 23]
        [17 38]]
        Actual:  pos Predicted:  neg fileid: pos/cv498_8832.txt
        Actual:  neg Predicted:  pos fileid: neg/cv952_26375.txt
        Actual:  pos Predicted:  neg fileid: pos/cv719_5713.txt
        Actual:  neg Predicted:  pos fileid: neg/cv314_16095.txt
        Actual:  pos Predicted:  neg fileid: pos/cv269_21732.txt
        Actual:  pos Predicted:  neg fileid: pos/cv638_2953.txt
        Actual:  neg Predicted:  pos fileid: neg/cv219_19874.txt
        Actual:  neg Predicted:  pos fileid: neg/cv097_26081.txt
        Actual:  neg Predicted:  pos fileid: neg/cv144_5010.txt
        Actual:  neg Predicted:  pos fileid: neg/cv398_17047.txt
        Actual:  pos Predicted:  neg fileid: pos/cv925_8969.txt
        Actual:  neg Predicted:  pos fileid: neg/cv056_14663.txt
        Actual:  pos Predicted:  neg fileid: pos/cv607_7717.txt
        Actual:  pos Predicted:  neg fileid: pos/cv439_15970.txt
        Actual:  pos Predicted:  neg fileid: pos/cv885_12318.txt
        Actual:  pos Predicted:  neg fileid: pos/cv887_5126.txt
        Printing the review with fileid pos/cv887_5126.txt
        the laserman : somehow the title of writer-director-producer peter wang's film conjures up images of superheroes , like ultraman and spiderman .
        you kind of expect an adventure flick about a crime fighter who can shoot laser beams from his fingertips .
        as it turns out , the laserman _is_ about crime and about laser beams , but there aren't any superheroes .
        instead , wang's film is populated by a group of refreshingly off-beat characters living in the ultimate cultural melting pot : new york city .
        the laserman is a comic brew which celebrates ethnicity , eccentricity , and electricity .
        the film tells the bizarre story of arthur weiss ( marc hayashi ) , a chinese-american laser scientist whose life becomes incredibly hectic after he accidentally kills his lab assistant in an experiment .
        he loses his job but finds work with a mysterious company which secretly plans to use laser technology to commit dastardly deeds .
        arthur's professional life is cluttered with moral dilemmas .
        his personal life , on the other hand , is cluttered with colorful friends and quirky relatives .
        in fact , arthur is by far the blandest character in the film , despite a charismatic performance by hayashi ( the san francisco-based actor whose films include chan is missing and the karate kid ii ) .
        it's the auxiliary characters who give the laserman its unique spark .
        arthur's not-so-typical jewish mother , ruth , for example , is convinced that a chinese soul is trapped in her jewish body .
        she has dyed her red hair black , she takes herbal medicine daily , and she is perpetually cooking up strange delicacies , such as matzo balls in soy sauce--the ultimate fusion of jewish and chinese cuisine .
        veteran stage actress joan copeland takes the part and runs with it , almost stealing the movie in the process .
        she plays ruth as a driven woman , determined to overcome her genetic heritage by immersing herself in chinese culture .
        arthur's girlfriend janet ( maryann urbano ) is a kooky free-spirit who would rather meditate than copulate ; her ultimate goal is orgasm through zen meditation .
        arthur's best friend , joey ( tony leung ) , is a small time thief who hustles everything from microwave ovens to machine guns .
        joey is married to arthur's jewish sister , but he is also having an affair with a chinese immigrant who works in a whore house .
        arthur's 11-year-old son , jimmy , played by the amazingly adorable david chan , is--horror of horrors--bad at math !
        he finds it impossible to meet his father's lofty expectations .
        the various people in arthur's life come together to form a rich tapestry of humanity .
        like wang's earlier film , a great wall ( about a san francisco family visiting relatives in china ) , the laserman revolves around cultural differences .
        every character in the film is , in some way or another , trying to find his identity--struggling to negotiate a balance between his native culture and the american way .
        the movie also offers a provocative look at technology .
        wang appears in the movie as lieutenant lu , a detective who is fed up with machines , even though he relies on them to do his job .
        the film views technology with a wary eye , acknowledging its necessity while at the same time realizing its potential dangers .
        wang raises the time-honored question of whether scientists should be held responsible for their inventions .
        was einstein responsible for the a-bomb ?
        is arthur weiss responsible for his lasers ?
        the movie pits spirituality against technology , man against machine , and the result is a draw .
        according to the film , technology has its place , but we must employ it with great forethought and caution .
        ironically , by its very nature , the laserman is a triumph of technology--the technology of filmmaking .
        wang's direction is exquisite , especially during the tense finale in which the director frantically cross-cuts between the various subplots , perhaps in homage to d . w .
        griffith .
        cinematographer ernest dickerson , who has worked on all of spike lee's films , gives the laserman a distinctive , artistic look .
        mason daring's score , which includes a send-up of bach , is right on target .
        the laserman is an ambitious endeavor , which is to be applauded , but it's sometimes ambitious to a fault .
        wang serves up so many slices of life in the film that it's hard to digest them all .
        for instance , one character ( arthur's sister ) has negligible screen time , and consequently we just don't care about her marital problems .
        in weaving his web , wang has included a few too many strands .
        overall , however , the laserman is a charmingly eclectic concoction .
        on the surface , the film is a light and bouncy comedy-thriller , overflowing with whimsical humor and visual style .
        the heavier issues emerge only when you take a deeper look at the film .
        you can ponder the moral questions or you can just sit back and enjoy the absurdity of life in china town .

        Score on test set:  0.71
    """

    #   (b) what other features you tried and which ones seemed to 
    #       help the most/least
    """
    As I mentioned earlier, I tried many features and also within the positive
     and negative feature I tried to consider so many things such as  negative word before 
     a positive word like not. The feature that gave me the best results are:
        features['positive'] = positive_count
        features['negative'] = negative_count
        features['total words']= total_words
        features['total sentence']= total_sentence
        features['sentimence']=TB.sentiment.subjectivity
        features['polarity']=TB.sentiment.polarity
    while some of the feature were giving me low result such as:
        features['negative_r'] = negative_count//total_words
        features['positive_r'] = positive_count//total_words
    The reason can be that these features were not best fit with other features.


    """
# Random Forest for EC

    """
    While using random forest, I got the accuracy higher which is around 76 percent
     with depth 3 and n_estimator of 120.
    [0.75719528014190784, 3, 100]


    """



    

