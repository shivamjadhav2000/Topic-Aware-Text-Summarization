import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
df=pd.read_csv('./dataset/quora_questions.csv')
data=list(df['Question'])
cv=CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm=cv.fit_transform(data)
print(dtm)
