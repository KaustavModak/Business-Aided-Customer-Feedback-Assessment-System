# importing required libraries and modules
import streamlit as st
import re
import nltk
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words=stopwords.words('english') # words which have minimum importance in the text
pattern=re.compile('[^a-zA-Z]') # pattern storing the special characters and numbers
port_stem=PorterStemmer() # creating an object of PorterStemmer()

# stemming definition
def stemming(content):
    stemmed_content=re.sub(pattern,' ',content) # removing all the characters from the content other than a-z or A-Z
    stemmed_content=stemmed_content.lower() # lowering all the characters from the content
    stemmed_content=stemmed_content.split() # splitting all the words from the content and storing it in a list
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if word not in stop_words] # stemming each word presnt in the list and then putting back only those words which are not stopwords
    stemmed_content=' '.join(stemmed_content) # joining back all the words of the content together with a ' '
    return stemmed_content

model=pickle.load(open('trained_model.pkl','rb')) # loading the saved Logistic Regression Model
vectorizer=pickle.load(open('vectorized.pkl','rb')) # loading the saved TF-IDF vectorizer

st.title("Business-Aided Customer Feedback Assessment System") # adding the title to streamlit interface
st.write("Enter a product review below to analyze the sentiment")

user_input=st.text_area(label="Your Review",value="")

if st.button('Predict Sentiment'):
    if user_input.split()=="":
        st.warning("Please enter the review")
    else:
        stemmed=stemming(user_input)
        vectorized_input=vectorizer.transform([stemmed])
        prediction=model.predict(vectorized_input)
        st.success(f"Predicted Sentiment: **{prediction}**")



'''
Here are some sample customer reviews you can use as user inputs to test your sentiment prediction app:

---

### Positive Reviews

```text
Absolutely loved this product! The flavor was amazing and the packaging was neat.
```

```text
I’ve been using this brand for a while and it never disappoints. Highly recommended!
```

```text
Fast delivery and great quality. Will definitely order again.
```

---

### Neutral Reviews

```text
It was okay. Not too bad, not too great — just average.
```

```text
Taste was decent, but not something I’d go out of my way for.
```

---

###  Negative Reviews

```text
Terrible experience. The product arrived broken and the quality was awful.
```

```text
I wouldn’t buy this again. It tasted bad and smelled weird.
```

```text
Disappointed. It looked good online but was completely different in reality.
```

'''
