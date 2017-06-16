import pandas as pd
import numpy as np
import eval_util
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import time
from sklearn import linear_model
from tensorflow import gfile

def csv_to_array(csv_filedir):
    """
    save array to scv
    """
    csv_table = pd.read_csv(csv_filedir)
    sorted_table = csv_table.sort_values(by='VideoId')
    sorted_table = sorted_table.reset_index(drop=True)
    label_array = np.zeros((len(sorted_table), 4716), dtype=np.float16)
    label_pair = sorted_table['LabelConfidencePairs']
    videoid = sorted_table['VideoId']
    for i in range(len(label_pair)):
        example = label_pair[i].split(' ')
        for j in range(0, len(example), 2):
            label_array[i, int(example[j])] = float(example[j + 1])
    return label_array, videoid

def array_to_csv(label_array, videoid, output_file, top_k=20):
    with gfile.Open(output_file, 'w+') as out_file:
        out_file.write("VideoId,LabelConfidencePairs\n")
        for i in range(label_array.shape[0]):
            top_indices = np.argpartition(label_array[i], -top_k)[-top_k:]
            line = [(class_index, label_array[i][class_index])
                    for class_index in top_indices]
            line = sorted(line, key=lambda p: -p[1])
            out_line = str(videoid[i]) + "," + " ".join("%i %f" % pair for pair in line) + "\n"
            out_file.write(out_line)

def w_mul(l_w_pair):
    label, w = l_w_pair
    return label*w

def ensemble(pred_list, weight_list):
    sum_label = None
    videoid_out = None
    pool = ThreadPool(4)
    sum_label_list = pool.map(w_mul, zip(pred_list,weight_list))
    pool.close()
    pool.join()
    sum_label = np.stack(sum_label_list,-1)
    del sum_label_list
    sum_label = np.sum(sum_label, 2, keepdims=False)
    sum_label /= np.sum(np.asarray(weight_list))
    return sum_label

class val(object):
    def __init__(self, input_files, label_file, top_k=20):
        start_time = time.time()
        pool = ThreadPool(4)
        pool_out = pool.map(csv_to_array, input_files)
        pool.close()
        pool.join()
        self.pred_list = zip(*pool_out)[0]
        print time.time()-start_time
        self.labels_val,self.videoid = csv_to_array(label_file)
        self.top_k = top_k
        self.weights = None

    def cal_gap(self):
        self.predict()
        evl_metrics = eval_util.EvaluationMetrics(4716, self.top_k)
        predictions_val = ensemble(self.pred_list, self.weights)

        print predictions_val.shape
        iteration_info_dict = evl_metrics.accumulate(predictions_val,
                                                         self.labels_val, np.zeros(predictions_val.shape[0]))
        epoch_info_dict = evl_metrics.get()
        print(("GAP@%d:" %self.top_k) + str(epoch_info_dict['gap']))

    def employ(self, input_files, output_file,top_k=20):
        """
        Perform ensemble based on the learned weights.
        """
        pool = ThreadPool(4)
        pool_out = pool.map(csv_to_array, input_files)
        pool.close()
        pool.join()
        pred_list = zip(*pool_out)[0]
        videoid = zip(*pool_out)[1][0]
        predictions_val = ensemble(pred_list, self.weights)
        array_to_csv(predictions_val, videoid, output_file, top_k)

    def predict(self):
        """
        Train the regression model with predictions on validation set.
        Save the learned weights to apply to test set predictions.
        """
        pred_array = np.stack(self.pred_list, -1)
        reg = linear_model.Ridge(alpha=.5)
        pred = np.reshape(pred_array, [-1, len(self.pred_list)])
        y = np.reshape(self.labels_val, [-1,1])
        reg.fit(pred, y)

        self.weights = reg.coef_[0].tolist()



