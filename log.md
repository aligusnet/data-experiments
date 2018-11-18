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

* **sklearn**: It took 33 hours to run a random search on the full set of training data of quora questions with no good results. It definitely makes sense to start searching on smaller subsets of data: Quora_Insincere_Questions/sklearn/token_counts.ipynb

* **Vowpal Wabbit**: Using spacy word vectors to train logistic model did not show as good results as ordinary Word Count or TF-IDF.

* **Vowpal Wabbit**: I do not understand how regularization works on VW. Small values for L1 and L2 around 1e-9 do not show any significant results, bigger values around 1e-6 demonstrated significant metrics degradation.

* **Vowpal Wabbit**: Squared and hinge loss function do not work as well as logistic. [VW Loss functions](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Loss-functions).
