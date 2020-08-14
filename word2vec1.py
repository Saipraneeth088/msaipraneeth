
import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = """The origin of India’s foreign trade can be traced back to the age of the Indus Valley
               civilization. But the growth of foreign trade gained momentum during the British rule.
               During that period, India was a supplier of food stuffs and raw materials to England
               and an importer of manufactured goods. However, organised attempts to promote foreign
               trade were made only after Independence, particularly with the onset of economic 
               planning. Indian economic planning completed five decades. During this period, the 
               value, composition, and direction of India’s foreign trade have undergone significant 
               changes. In early 1990s the Indian economy had witnessed dramatic policy changes. 
               The idea behind the new economic model known as Liberalization, Privatization, and 
               Globalization in India (LPG), was to make the Indian economy one of the fastest 
               growing economies in the world. An array of reforms was initiated with regard to 
               industrial, trade and social sector to make the economy more competitive. Earlier
               India was afraid of global companies and the government ensured high tariff barriers.
               But now the scenario is changing and the competition is looked at from the holistic 
               angle."""



# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)


words = model.wv.vocab

# Finding Word Vectors
vector = model.wv['model']

# Most similar words
similar = model.wv.most_similar('privatization')