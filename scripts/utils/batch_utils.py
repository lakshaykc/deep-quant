"""
Batch utils for batch generator
"""

import time
import pandas as pd
import numpy as np
import scipy.stats as st


class Outlier(object):
    """ Outlier class to detect and remove outliers"""

    def __init__(self, data, start_indices, end_indices, fin_colidxs, stride, confidence_level, window,
                 start_date, end_date, max_unrollings):
        self._data = data
        self._start_indices = start_indices
        self._end_indices = end_indices
        self._fin_colidxs = fin_colidxs
        self._stride = stride
        self.confidence_level = confidence_level
        self.window = window
        self._start_date = start_date
        self._end_date = end_date
        self._max_unrollings = max_unrollings

        self.outlier_df = pd.DataFrame(index=self._data.index)
        self.outlier_arr = [False] * len(self._data.index)
        self.fin_col_df = self._data[[self._data.columns[x] for x in self._fin_colidxs]]

        # Convert the datadate column to datetime object. 'datadate_obj' is the new column keeping the original
        # 'datadate' format intact
        self._data['datadate_obj'] = pd.to_datetime(self._data['date'], format="%Y%m")

    def _get_output_series(self, gvkey):
        """
        Returns pandas series for the given output (oiadpq_ttm) for the gvkey
        :param gvkey:
        :return: pandas series
        """
        df = self.fin_col_df[self._data['gvkey'] == gvkey]
        return df['oiadpq_ttm']

    def _get_outlier_idxs(self, confidence_level=0.999, window=2):
        """
        Identifies the outliers based on rolling mean and std dev and returns index of outliers
        :param confidence_level:
        :param window: window size for rolling computation
        :return: indices of outliers
        """
        window = int(self._stride * window)
        z_score = st.norm.ppf(confidence_level)
        t = time.time()

        for i, gvkey in enumerate(self._data.gvkey.unique()):
            if i % 100 == 0:
                print(i, time.time() - t)
                t = time.time()

                output_series = self._get_output_series(gvkey=gvkey)

                roll_mean = output_series.rolling(window, min_periods=1).mean()
                dev = z_score * output_series.rolling(window).std()

                # Identify outliers based on lb and ub
                lb_series = roll_mean - dev
                ub_series = roll_mean + dev

                # Fill NaNs with inf values so the output values are not rejected as outliers
                lb_series = lb_series.fillna(-np.inf)
                ub_series = ub_series.filla(np.inf)

                # boolean series to keep track of outliers
                outlier_flag = ~((lb_series <= output_series) * (output_series <= ub_series))

                for ix in outlier_flag.index:
                    if outlier_flag[ix]:
                        self.outlier_arr[self._data.index.get_loc(ix)] = True
                self.outlier_df['outlier'] = self.outlier_arr

        return self.outlier_df[self.outlier_df['outlier'] == True].index

    def _remove_outlier(self):
        """
        Removes the outlier indices collected from _get_outlier_idx
        :return: start_indices, end_indices
        """
        outlier_idx = self._get_outlier_idxs()
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

    def get_indices(self, train=True):
        """
        Runs outlier removal method and returns start and end indices
        :param train: True, if outlier removal is applicable to training data
        :return: start_indices, end_indices
        """
        assert train is True
        return self._remove_outlier()

    def get_outlier_bounds_for_preds(self, confidence_level=0.999, window=2):
        """
        Determines upper and lower bounds for outlier detection based on the history of target field. For every date
        and gvkey, output fields are used to create the bounds using rolling stationary method.
        The history goes as far as window_size * stride

         Returns dataframes for lower and and upper bounds with dates as index and gvkeys as column names (similar to
         output dataframes used during prediction)

        :return: [lower_bound_dataframe, upper_bound_dataframe]
        """

        # Trim the data outside the date ranges
        print("Before trim, data shape:")
        print(self._data.shape)
        self._data = self._trim_data(self._data, self._start_date, self._end_date, self._max_unrollings)
        print("After trim, data shape:")
        print(self._data.shape)

        # Get gvkeys
        unique_gvkeys = self._data.gvkey.unique()
        print('Unique gvkeys:')
        print(unique_gvkeys.shape)
        df_lb = pd.DataFrame()
        df_ub = pd.DataFrame()

        # Parameters for rolling calculations and outliers
        window = int(self._stride * window)
        z_score = st.norm.ppf(confidence_level)

        t = time.time()
        print("Running outlier analysis for predictions ...")
        for i, gvkey in enumerate(unique_gvkeys):
            if i % 100 == 0:
                print(i, time.time() - t)
                t = time.time()

            output_series = self._get_output_series(gvkey=gvkey)

            roll_mean = output_series.rolling(window, min_periods=1).mean()
            std = output_series.rolling(window).std()

            # Create the lb, ub output series
            lb_series = roll_mean - z_score * std
            ub_series = roll_mean + z_score * std

            # Replace NaNs with inf in std series. The NaNs are due to rolling window. These values should always
            # be kept and there is not enough data to say if they are outliers or not.
            lb_series = lb_series.fillna(-np.inf)
            ub_series = ub_series.fillna(np.inf)

            # Rename the series
            lb_series, ub_series = lb_series.rename(gvkey), ub_series.rename(gvkey)

            # Convert series to dataframe
            df_gvkey_lb, df_gvkey_ub = lb_series.to_frame(), ub_series.to_frame()

            # Concatenate to the main dataframe
            df_lb = pd.concat([df_lb, df_gvkey_lb], axis=1)
            df_ub = pd.concat([df_ub, df_gvkey_ub], axis=1)

        df_lb = df_lb.fillna(-np.inf)
        df_ub = df_ub.fillna(np.inf)

        return df_lb, df_ub

    @staticmethod
    def _trim_data(data, start_date, end_date, max_unrollings):
        """
        Trims the data to keep only the data within the start and end dates of training or prediction
        :param start_date:
        :param end_date:
        :param max_unrollings:
        :return:
        """

        start_date = pd.to_datetime(start_date, format="%Y%m")
        # Offset the start date to include the data corresponding to max_unrollings which is required as input data
        start_date_offset = start_date - pd.DateOffset(years=max_unrollings)
        end_date = pd.to_datetime(end_date, format="%Y%m")

        # Filter the data according to the dates
        data = data[data.datadate_obj >= start_date_offset]
        data = data[data.datadate_obj <= end_date]

        return data
