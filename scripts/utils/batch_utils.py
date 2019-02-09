"""
Batch utils for batch generator
"""

import time
import pandas as pd
import numpy as np
import scipy.stats as st


class Outlier(object):
    """ Outlier class to detect and remove outliers"""

    def __init__(self, data, start_indices, end_indices, fin_colidxs, stride):
        self._data = data
        self._start_indices = start_indices
        self._end_indices = end_indices
        self._fin_colidxs = fin_colidxs
        self._stride = stride

    def _get_outlier_idxs(self, method):
        """
        Returns indices of outliers specified via method.
        :param method: string, choose from 'normal','ransac','rolling'
        :return: List of indices of outliers
        """
        if method == 'normal':
            return self._normal_method(confidence_interval=0.975)
        else:
            raise ValueError("Invalid method name or other methods haven't been defined yet")

    def _remove_outlier(self, method):
        """
        Removes the outlier indices collected from _get_outlier_idx
        :param method: string, choose from 'normal','ransac','rolling'
        :return: start_indices, end_indices
        """
        outlier_idx = self._get_outlier_idxs(method=method)
        new_start_indices = []
        new_end_indices = []

        print("Indices before outlier: %i" % len(self._start_indices))

        for i in range(len(self._start_indices)):
            start_idx = self._start_indices[i]
            end_idx = self._end_indices[i]
            seq_outlier_status = False
            num_steps = (end_idx - start_idx) // self._stride + 1
            for cur_idx in [start_idx + step * self._stride for step in range(num_steps)]:
                seq_outlier_status = seq_outlier_status or cur_idx in outlier_idx

            if seq_outlier_status is False:
                new_start_indices.append(start_idx)
                new_end_indices.append(end_idx)

        self._start_indices = new_start_indices
        self._end_indices = new_end_indices
        print("Indices after outliers removed: %i" % len(self._start_indices))
        return self._start_indices, self._end_indices

    def _normal_method(self, confidence_interval=0.95):
        """
        Identifies the outlier based on normal distribution and confidence level
        :return: List of indices where feature vector is an outlier
        """

        outlier_df = pd.DataFrame(index=self._data.index)
        outlier_arr = [False]*len(self._data.index)
        fin_col_df = self._data[[self._data.columns[x] for x in self._fin_colidxs]]
        t = time.time()
        for i, gvkey in enumerate(self._data.gvkey.unique()):
            if i % 100 == 0:
                print(i, time.time() - t)
                t = time.time()

            # Slice a local copy of the gvkey dataframe and identify outliers based on growth rates of oiadpq
            df = fin_col_df[self._data['gvkey'] == gvkey]
            growth_rate = df['oiadpq_ttm'] / df['oiadpq_ttm'].shift(periods=1)
            growth_rate = growth_rate[~growth_rate.isin([np.nan, np.inf, -np.inf])]

            if growth_rate.size <= 1:
                continue

            # Calculate the outliers based on z-score
            z_score = st.norm.ppf(confidence_interval)
            std = np.std(growth_rate.values)
            mean = np.mean(growth_rate.values)
            growth_rt_outlier = [x < mean - z_score * std or x > mean + z_score * std for x in growth_rate.values]
            growth_rt_outlier = pd.Series(growth_rt_outlier, index=growth_rate.index)

            for ix in growth_rt_outlier.index:
                if growth_rt_outlier[ix]:
                    outlier_arr[self._data.index.get_loc(ix)] = True
            outlier_df['outlier'] = outlier_arr

        return outlier_df[outlier_df['outlier'] == True].index

    def _rolling(self):
        return

    def _ransac(self):
        return

    def get_indices(self, method):
        """
        Runs outlier removal method and returns start and end indices
        :param method: string, choose from 'normal','ransac','rolling'
        :return: start_indices, end_indices
        """
        return self._remove_outlier(method=method)

