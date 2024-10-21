# Step 1: Topic Modeling using LDA
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import nltk
import pandas as pd
df=pd.read_csv('./dataset/quora_questions_filtered.csv')
documents=list(df['Question'])
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess text
def preprocess(text):
    # Load spacy model
    nlp = spacy.load('en_core_web_sm')
    
    # Tokenize
    doc = nlp(text.lower())
    
    # Remove stopwords and lemmatize
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return tokens


# Preprocess documents
preprocessed_docs = [preprocess(doc) for doc in documents[:5000]]

# Create a dictionary and a corpus
dictionary = corpora.Dictionary(preprocessed_docs)
corpus = [dictionary.doc2bow(text) for text in preprocessed_docs]

# Train LDA model
lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, passes=15)

# Print the topics
topics = lda_model.print_topics(num_words=4)
for topic in topics:
    print(f"Topic {topic[0]}: {topic[1]}")

# Visualize the topics using pyLDAvis
lda_display = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)