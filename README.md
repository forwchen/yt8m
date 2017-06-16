# Models from Xi Wang
There are three branches of code from Xi Wang:

### 1. diff

  This branch contains code that does feature transformation (take difference of adjacent features in the sequence).
  This is achieved by modifying the `readers.py` [code](https://github.com/forwchen/yt8m/blob/diff/readers.py#L197).
    
### 2. filter

  This branch contains code that does label filtering. The non-filtered classes are selected to be those with less training samples.
  This is achieved by discarding the labels of the filtered classes in `readers.py` [code](https://github.com/forwchen/yt8m/blob/master/yt8m-xiw/filter/readers.py#L129).
  The classes that are filtered are stored in `filter-*.npy` respectively.
  
### 3. xiw

  This branch contains code for normal models, such as LSTM, GRU, DBoF, etc.

The models in these branches are all trained with default arguments in video_level_models.py or frame_level_models.py. For example, the lstm_layers are all 2 for RNN models as defined in frame_level_models.py. By default, the models are trained with only training data. The learning rates for GRU, LSTM, LayerNorm LSTM models are 0.001. The learning rates for DBoF and MoE models are 0.1.

Particularly, this branch `diff` is used to train the following models:

|Model|Setting|
| --- | --- |
|GRU|Feature transformation|
|GRU|Feature transformation, learning rate = 0.0005|
|GRU|Feature transformation, using train+validate data|
