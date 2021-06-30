from hnlp import Corpus

path = "/media/sf_kubuntu/lab/corpus/dialogTo20210601/dialog.json"
corpus = Corpus("custom", path)


for item in corpus:
    print(item)
