import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import string

# read data
trainingSet = pd.read_csv("./data/train.csv")
testingSet = pd.read_csv("./data/test.csv")

# calculate the helpfulness weight
trainingSet['Helpful%'] = np.where(trainingSet['HelpfulnessDenominator'] > 0,
                                   trainingSet['HelpfulnessNumerator'] / trainingSet['HelpfulnessDenominator'], -1)
# trainingSet['HelpfulnessRange'] = pd.cut(x=trainingSet['Helpful%'], bins=[-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0],
#                                          labels=['0', '1', '2', '3', '4', '5'], include_lowest=True)
trainingSet = trainingSet.drop(columns=['HelpfulnessDenominator', 'HelpfulnessNumerator'])
print(trainingSet.head())

# # test with a small training set
# trainingSet = trainingSet.head(10000)
# trainingSet.to_csv('./data/train_small.csv')


# create a text processor
def text_process(text):
    nopunc = [i.lower() for i in text if i not in string.punctuation]
    nopunc_text = ''.join(nopunc)
    return [i for i in nopunc_text.split() if i not in stopwords.words('english') and not i.isdigit()]


trainingSet = trainingSet.replace(np.nan, '', regex=True)
Tf_Idf = TfidfVectorizer(ngram_range=(1, 2), analyzer=text_process)
text_vector = Tf_Idf.fit_transform(trainingSet['Text'])
text_vector = pd.DataFrame(text_vector.toarray(), columns=Tf_Idf.get_feature_names())
sum_vector = Tf_Idf.fit_transform(trainingSet['Summary'])
sum_vector = pd.DataFrame(sum_vector.toarray(), columns=Tf_Idf.get_feature_names())
trainingSet = trainingSet.join(text_vector)
trainingSet = trainingSet.join(sum_vector, lsuffix='text', rsuffix='sum')
# # test with only summary review
# trainingSet = trainingSet.join(sum_vector)
trainingSet = trainingSet.drop(columns=['Summary', 'Text'])

# split test set out of the training set and generate a prediction set with no score but other used features
predictionSet = pd.merge(trainingSet, testingSet, left_on='Id', right_on='Id')
predictionSet = predictionSet.drop(columns=['Score_x'])
predictionSet = predictionSet.rename(columns={'Score_y': 'Score'})
predictionSet = predictionSet.drop(columns=['ProductId', 'UserId', 'Time'])
predictionSet.to_csv('./data/predictionSet.csv', index=False)
print(predictionSet.shape)
print(predictionSet.head())

# pre-process training set, drop columns of Id, UserId, Time
trainingSet = trainingSet.drop_duplicates(subset=['ProductId', 'UserId', 'Time'])
trainingSet = trainingSet.drop(columns=['Id', 'ProductId', 'UserId', 'Time'])
trainingSet = trainingSet.replace('', np.nan)
trainingSet = trainingSet.dropna()
trainingSet.to_csv('./data/trainingSet.csv', index=False)
print(trainingSet.shape)
print(trainingSet.head())







