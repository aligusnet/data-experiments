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

* **Vowpal Wabbit**: Cleaning text data does not show good results for logistic/squared/hinge loss functions (/Quora_Insincere_Questions/Vowpal_Wabbit/logistic_spacy_vectors_clean_data.ipynb). Unexpected.

### Spam Assassin Corpus

* **spacy** - cleaning text data (removing stop words and punctuation) allowed to achieve the best results (/Spam_Assassin_Corpus/sklearn/spacy_vectors_clean_data_random_forest.ipynb).


## 19/11/2018

### Quora Insincere Questions

* **tokens counts - logistic regression** - train on 20 000 samples.

Best results on test data:

| | |
| --- | ---
| accuracy | 0.9214
| precision | 0.3952991452991453
| recall | 0.6271186440677966
| f1 | 0.4849279161205766
| AUC | 0.8917624596984817

Best result got using Grid Search:

| | |
| --- | ---
| accuracy | 0.9226
| precision | 0.4012875536480687
| recall | 0.6338983050847458
| f1 | 0.49145860709592637
| AUC | 0.8916551090617626

* **TF-IDF - logistic regression** - train on 20 000 samples.

Best result got using Grid Search:

| | |
| --- | ---
| accuracy | 0.9214
| precision | 0.39705882352941174
| recall | 0.6406779661016949
| f1 | 0.490272373540856
| AUC | 0.903716565500099

* **cleaned tokens counts - logistic regression**

| | |
| --- | ---
| accuracy | 0.92
| precision | 0.389937106918239
| recall | 0.6305084745762712
| f1 | 0.4818652849740932
| AUC | 0.8956688701165367

* **spacy vectors - logistic regression**

| | |
| --- | ---
| accuracy | 0.9268
| precision | 0.41277641277641275
| recall | 0.5694915254237288
| f1 | 0.4786324786324786
| AUC | 0.9163407121886201
