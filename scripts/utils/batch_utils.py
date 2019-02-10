"""
Batch utils for batch generator
"""

import time
import pandas as pd
import numpy as np
import scipy.stats as st


class Outlier(object):
    """ Outlier class to detect and remove outliers"""

    def __init__(self, data, start_indices, end_indices, fin_colidxs, stride, confidence_level, window=None):
        self._data = data
        self._start_indices = start_indices
        self._end_indices = end_indices
        self._fin_colidxs = fin_colidxs
        self._stride = stride
        self.confidence_level = confidence_level
        self.window = window

        self.outlier_df = pd.DataFrame(index=self._data.index)
        self.outlier_arr = [False] * len(self._data.index)
        self.fin_col_df = self._data[[self._data.columns[x] for x in self._fin_colidxs]]

    def _get_outlier_idxs(self, method):
        """
        Returns indices of outliers specified via method.
        :param method: string, choose from 'normal',rolling_std,rolling_stationary
        :return: List of indices of outliers
        """
        if method == 'normal':
            return self._normal_method(confidence_level=self.confidence_level)
        elif method == 'rolling_stationary':
            return self._rolling_stationary(confidence_level=self.confidence_level, window=self.window)
        elif method == 'rolling_std':
            return self._rolling_std(confidence_level=self.confidence_level, window=self.window)
        else:
            raise ValueError("Invalid method name")

    def _remove_outlier(self, method):
        """
        Removes the outlier indices collected from _get_outlier_idx
        :param method: string, choose from 'normal',rolling_std,rolling_stationary
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

    def _get_growth_rate(self, gvkey):
        """
        Creates the growth rate series from oiadpq with a period of 1. Return growth rate series, mean and std.
        Mean and std are calculated excluding nan, inf values
        :param gvkey:
        :return: growth rate pandas series, mean, std
        """
        # Slice a local copy of the gvkey dataframe and identify outliers based on growth rates of oiadpq
        df = self.fin_col_df[self._data['gvkey'] == gvkey]
        growth_rate = df['oiadpq_ttm'] / df['oiadpq_ttm'].shift(periods=1)
        # Get mean excluding nan, inf
        reduced_gr = growth_rate[~growth_rate.isin([np.nan, np.inf, -np.inf])]
        mean = reduced_gr.mean()
        std = reduced_gr.std()
        growth_rate = growth_rate.fillna(mean)
        # Replace inf and -nf with large value as they are likely to be outliers. They should not be ignored
        growth_rate = growth_rate.replace(np.inf, np.nan)
        growth_rate = growth_rate.replace(-np.inf, np.nan)

        return growth_rate, mean, std

    def _normal_method(self, confidence_level=0.975):
        """
        Identifies the outlier based on normal distribution of growth rates and confidence level.
        Mean and std dev of the complete series for a gvkey is used (not the rolling mean)
        :return: List of indices where feature vector is an outlier
        """
        z_score = st.norm.ppf(confidence_level)
        t = time.time()

        for i, gvkey in enumerate(self._data.gvkey.unique()):
            if i % 100 == 0:
                print(i, time.time() - t)
                t = time.time()

            growth_rate, mean, std = self._get_growth_rate(gvkey=gvkey)
            growth_rate = growth_rate.fillna(1e8)

            if growth_rate.size <= 1:
                continue

            # Calculate the outliers based on z-score
            growth_rt_outlier = [x < mean - z_score * std or x > mean + z_score * std for x in growth_rate.values]
            growth_rt_outlier = pd.Series(growth_rt_outlier, index=growth_rate.index)

            for ix in growth_rt_outlier.index:
                if growth_rt_outlier[ix]:
                    self.outlier_arr[self._data.index.get_loc(ix)] = True
            self.outlier_df['outlier'] = self.outlier_arr

        return self.outlier_df[self.outlier_df['outlier'] == True].index

    def _rolling_stationary(self, confidence_level=0.975, window=3):
        """
        Identifies outliers based on normal distribution of residuals around rolling mean
        :param confidence_level: confidence level for classifying outliers
        :param window: window size for rolling mean in years. The metric is internally multiplied by stride
        :return: List of indices where feature vector is an outlier
        """
        window = int(self._stride * window)
        z_score = st.norm.ppf(confidence_level)
        t = time.time()

        for i, gvkey in enumerate(self._data.gvkey.unique()):
            if i % 100 == 0:
                print(i, time.time() - t)
                t = time.time()

            growth_rate, mean, std = self._get_growth_rate(gvkey=gvkey)
            # growth_rate has np.nan inplace of inf or -inf

            if growth_rate.size <= 1:
                continue

            # rolling computations
            growth_rate_rolling_mean = growth_rate.rolling(window, min_periods=1).mean()
            residuals = growth_rate - growth_rate_rolling_mean

            # calculate the outliers based on residuals
            # Here is the main difference from _normal_method. In this method, we take the residuals of the
            # rolling mean and use that for computing std dev. In _normal_method, std dev of growth_rate is used
            # which is for the whole series and not just the window size
            std = np.std(residuals.values)
            growth_rate = growth_rate.fillna(1e8)
            growth_rt_outlier = [y < mean - z_score * std or y > mean + z_score * std for y, mean in
                                 zip(growth_rate.values, growth_rate_rolling_mean.values)]
            growth_rt_outlier = pd.Series(growth_rt_outlier, index=growth_rate.index)

            for ix in growth_rt_outlier.index:
                if growth_rt_outlier[ix]:
                    self.outlier_arr[self._data.index.get_loc(ix)] = True
            self.outlier_df['outlier'] = self.outlier_arr

        return self.outlier_df[self.outlier_df['outlier'] == True].index

    def _rolling_std(self, confidence_level=0.975, window=3):
        """
        Identifies outliers based on normal distribution of residuals around rolling mean with rolling std
        :param confidence_level: confidence level for classifying outliers
        :param window: window size for rolling mean in years. The metric is internally multiplied by stride
        :return: List of indices where feature vector is an outlier
        """
        window = int(self._stride * window)
        z_score = st.norm.ppf(confidence_level)
        t = time.time()

        for i, gvkey in enumerate(self._data.gvkey.unique()):
            if i % 100 == 0:
                print(i, time.time() - t)
                t = time.time()

            growth_rate, mean, std = self._get_growth_rate(gvkey=gvkey)
            # growth_rate has np.nan inplace of inf or -inf

            if growth_rate.size <= 1:
                continue

            # rolling computations
            growth_rate_rolling_mean = growth_rate.rolling(window, min_periods=1).mean()
            residuals = growth_rate - growth_rate_rolling_mean

            # calculate the outliers based on residuals
            # Here is the main difference from _normal_method. In this method, we take the residuals of the
            # rolling mean and use that for computing std dev. In _normal_method, std dev of growth_rate is used
            # which is for the whole series and not just the window size

            # Rolling std
            std = residuals.rolling(window, min_periods=1).std()
            growth_rate = growth_rate.fillna(1e8)
            growth_rt_outlier = [y < mean - z_score * std or y > mean + z_score * std for y, mean, std in
                                 zip(growth_rate.values, growth_rate_rolling_mean.values, std.values)]
            growth_rt_outlier = pd.Series(growth_rt_outlier, index=growth_rate.index)

            for ix in growth_rt_outlier.index:
                if growth_rt_outlier[ix]:
                    self.outlier_arr[self._data.index.get_loc(ix)] = True
            self.outlier_df['outlier'] = self.outlier_arr

        return self.outlier_df[self.outlier_df['outlier'] == True].index

    def get_indices(self, method):
        """
        Runs outlier removal method and returns start and end indices
        :param method: string, choose from 'normal',rolling_std,rolling_stationary
        :return: start_indices, end_indices
        """
        return self._remove_outlier(method=method)
