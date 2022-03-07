# hnlp
Humanly DeepLearning NLP, quickly demo and easily play.

- Quickly setup a production, from 0 to 90%+, gain the maximum marginal income.
- Quickly evaluate a new idea based on the existing datasets and models.

```python

data = Corpus() >> Preprocessor() >> Tokenizer() >> DataManager()
# x_train, y_train = data.run("/path/to/file")

pipe = (
    Corpus() >> 
    Preprocessor() >> 
    Tokenizer() >> 
    DataManager() >> 
    Pretrained() >> 
    Model(config) >>
    Task(optimizer, loss_fn, metric)
)

pipe.train("/path/to/train_file")
pipe.test("/path/to/test_file")
```



## Test

```bash
$ python -m pytest
```





- [x] 自动寻找 LearningRate
