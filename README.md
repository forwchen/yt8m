# Models from Xi Wang
There are three branches of code from Xi Wang:

### 1. diff

  This branch contains code that does feature transformation (take difference of adjacent features in the sequence).
  This is achieved by modifying the `readers.py` [code](https://github.com/forwchen/yt8m/blob/diff/readers.py#L197).
    
### 2. filter

  This branch contains code that does label filtering. The non-filtered classes are selected to be those with less training samples.
  This is achieved by discarding the labels of the filtered classes in `readers.py` [code](https://github.com/forwchen/yt8m/blob/filter/readers.py#L129).
  The classes that are filtered are stored in `filter-*.npy` respectively.
  
### 3. xiw

  This branch contains code for normal models, such as LSTM, GRU, DBoF, etc.

The models in these branches are all trained with default arguments in `video_level_models.py` or 
`frame_level_models.py`. For example, the `lstm_layers` are all 2 for RNN models as defined in `frame_level_models.py`.
By default, the models are trained with only training data.
The learning rates for GRU, LSTM, LayerNorm LSTM models are 0.001.
The learning rates for DBoF and MoE models are 0.1.

Particularly, this branch `filter` is used to train the following models:

|Model|Setting|
| --- | --- |
|MoE |Train on 2534 classes with less positive samples|
|MoE |Train on 3571 classes with less positive samples|
