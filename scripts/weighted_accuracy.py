import pandas as pd
import numpy as np
import math

def weighted_accuracy(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    df = pd.concat([y_true.rename(columns={'spot_id_delta': 'y_true'}), y_pred.rename(columns={'spot_id_delta': 'y_pred'})], axis=1)
    df['accuracy'] = df.apply(lambda row: (math.floor(abs((np.sign(row.y_true) + np.sign(row.y_pred)/2))))*(1 - abs((row.y_true - row.y_pred)/row.y_true)), axis=1)
    return df['accuracy'].mean()

if __name__ == '__main__':

    y_true = pd.DataFrame(data={'spot_id_delta': [1.4, 4, -2, -7.3]})
    y_pred = pd.DataFrame(data={'spot_id_delta': [1, -1, 1, -1]})
    result = weighted_accuracy(y_true, y_pred)
    assert round(result, 2) == 0.21