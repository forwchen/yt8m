# Models from Xi Wang
There are three branches of code from Xi Wang:

### 1. diff

  This branch contains code that does feature transformation (take difference of adjacent features in the sequence).
  This is achieved by modifying the `readers.py` [code](https://github.com/forwchen/yt8m/blob/master/yt8m-xiw/diff/readers.py#L197).
    
### 2. filter

  This branch contains code that does label filtering. The non-filtered classes are selected to be those with less training samples.
  This is achieved by discarding the labels of the filtered classes in `readers.py` [code](https://github.com/forwchen/yt8m/blob/master/yt8m-xiw/filter/readers.py#L129).
  The classes that are filtered are stored in `filter-*.npy` respectively.
  
### 3. xiw

  This branch contains code for normal models, such as LSTM, GRU, DBoF, etc.
