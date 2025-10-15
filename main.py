import nltk
# data required for the examples and exercises in the book
nltk.download("book")
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')

import spacy 
import pandas as pd


# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_csv("hf://datasets/franciellevargas/HateBR/HateBR.csv")