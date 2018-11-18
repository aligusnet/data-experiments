# Data Experiments Log

## 17/11/2018

### Spam Assassin Corpus

* We can use text similarity  (computing based on pre-learnt word similarity statistical models) to classify texts. Spacy includes some models: [SpaCy en models](https://spacy.io/models/en);

* removing unnecessary  pipeline components allows significantly improve documents processing time:

```python
import spacy
nlp = spacy.load('en_core_web_lg', disable=["tagger", "parser", "ner"])
```

* large statistical model gives better results rather than the median one with the same documents processing time (more time requires to load the model);

* I believe it is better to use vectors only statistical model if only similarity analysis required (`en_vectors_web_lg` vs. `en_core_web_lg`). However, I've got slightly worse results using vectors only model.


## 18/11/2018

### Quora Insincere Questions

* It took 33 hours to run a random search on the full set of training data of quora questions with no good results. It definitely makes sense to start searching on smaller subsets of data: Quora_Insincere_Questions/sklearn/token_counts.ipynb
