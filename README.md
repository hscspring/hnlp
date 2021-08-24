# hnlp
Humanly DeepLearning NLP, quickly demo and easily play.

- Quickly setup a production, from 0 to 90%+, gain the maximum marginal income.
- Quickly evaluate a new idea based on the existing datasets.

```python

train = Corpus() >> Preprocessor() >> Tokenizer() >> DataManager()
# x_train, y_train = train.load_data()

task = Pretrained() >> Model() 

task.fit(dm)
task.evaluate(test)


```



## Test

```bash
$ python -m pytest
```





- [x] 自动寻找 LearningRate
