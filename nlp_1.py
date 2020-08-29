'''
Author: Shamanth K M
Date: 29/08/2020
'''

import nltk
from nltk.corpus import stopwords

stopword = stopwords.words("english")
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# convert text to lower case
text = "A boy and a girl were playing together. The boy had a collection of marbles. The girl has some sweets with her. The boy told the girl that he would give her all his marbles in exchange for the sweets with her. The girl agreed.The boy kept the most beautiful and the biggest marbles with him and gave her the remaining marbles. The girl gave him all her sweets as she promised. That night the girl slept peacefully. But the boy could not sleep as he kept wondering if the girl has hidden some sweets from him the way he had hidden the best marbles from her."
text = text.lower()

# word tokenize
word_tokens = nltk.word_tokenize(text)
print(word_tokens)

# sent tokenize
sent_token = nltk.sent_tokenize(text)
print(sent_token)

# stop words removal
removing_stopwords = [word for word in word_tokens if word not in stopword]
print(removing_stopwords)

# lemmatize the text so as to get its root form eg: functions,funtionality as function
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords]
print(lemmatized_word)

# stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form
snowball_stemmer = SnowballStemmer("english")
stemmed_word = [snowball_stemmer.stem(word) for word in lemmatized_word]
print(stemmed_word)

# Remove punctuation marks
words_without_punctuation = []
for word in lemmatized_word:
    if word.isalpha():
        words_without_punctuation.append(word)
print(words_without_punctuation)

# POS tag helps us to know the tags of each word like whether a word is noun, adjective etc.
pos_tag = nltk.pos_tag(words_without_punctuation)
print(pos_tag)

# counting the word occurrence using FreqDist library
freq = FreqDist(words_without_punctuation)
print(freq.most_common(10))

#plot the frequency of words
freq.plot(10)

#plotting the wordcloud
wordcloud = WordCloud().generate(text)
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show
