import os
from datetime import datetime

import numpy as np
from ggplot import *
import pandas as pd
from matplotlib import pyplot as plt

from strategies.pairs_trading.data.database.factory import InstrumentsDbFactory


class SimpleAlgorithm:
    def __init__(self):
        factory = InstrumentsDbFactory()
        self.prices_table = factory.get_table('prices')
        self.pair_prices = None
        self.pair_1 = None
        self.pair_2 = None
        self.start_date = None
        self.end_date = None

    def set_pair(self, pair_1, pair_2):
        self.pair_1 = pair_1
        self.pair_2 = pair_2

    def set_data(self):
        self.pair_prices = self.prices_table.get_pair_close(self.pair_1, self.pair_2)

    def set_date(self, start_date='2010-01-01', end_date='2020-01-03'):
        self.start_date = start_date
        self.end_date = end_date

    def __get_pair_prices(self):
        pair_prices = self.pair_prices.copy()
        pair_prices = pair_prices[((pair_prices['code'] == self.pair_1)
                                    | (pair_prices['code'] == self.pair_2))
                                   & ((pair_prices['date'] >= self.start_date) &
                                      (pair_prices['date'] <= self.end_date))]

        return pair_prices

    def __get_normalised_close(self):
        normalised_close = self.__get_pair_prices()
        first_close = normalised_close.groupby(['date', 'code']).first().reset_index()
        first_close_dict = {
            self.pair_1: first_close[first_close['code'] == self.pair_1]['close'].values[0],
            self.pair_2: first_close[first_close['code'] == self.pair_2]['close'].values[0]
        }
        normalised_close['normalised_close'] = normalised_close.apply((lambda row: row['close']/first_close_dict[row['code']]), axis=1)

        return normalised_close

    def get_spread(self):
        spreadprices = self.__get_normalised_close()
        spread_df = spreadprices[['date', 'code', 'normalised_close']].pivot(index='date', columns=['code'], values='normalised_close')
        spread_df = spread_df.assign(spread=(spread_df[self.pair_1] / spread_df[self.pair_2])).reset_index()
        spread_df = spread_df.melt(id_vars=['date'], value_vars=[self.pair_1, self.pair_2, 'spread'], var_name='code', value_name='value')

        return spread_df

    def plot_prices(self):
        pair_prices = self.__get_pair_prices()
        p = (ggplot(aes(x='date', y='close', color='code'), data=pair_prices) + geom_line())
        print(p)

    def plot_normalised_close(self):
        normalised_close = self.__get_normalised_close()
        p = (ggplot(aes(x='date', y='normalised_close', color='code'), data=normalised_close) + geom_line())
        print(p)

    def plot_spreadprices(self):
        spread_df = self.get_spread()

        p = (ggplot(aes(x='date', y='value', color='code'), data=spread_df) + geom_line())
        print(p)

    def plot_static_threshold(self):
        pair_prices = self.__get_normalised_close()
        soy_prices = pair_prices.copy()
        soy_prices = soy_prices.assign(soy=pd.to_datetime(soy_prices['date']).dt.to_period('Y').dt.to_timestamp())
        soy_prices = soy_prices.sort_values('soy', ascending=False).groupby(['code', 'soy'], as_index=False).apply(lambda x: x.iloc[0])
        soy_prices = soy_prices.assign(soy_close=soy_prices['close'])
        soy_prices = soy_prices[['code', 'soy', 'soy_close']]

        pair_prices = pair_prices.assign(soy=pd.to_datetime(pair_prices['date']).dt.to_period('Y').dt.to_timestamp())
        pair_prices = pd.merge(pair_prices, soy_prices, on=['code', 'soy'])
        pair_prices['normalised_close'] = pair_prices['close']/pair_prices['soy_close']
        pair_prices = pair_prices[['date', 'code', 'normalised_close']].pivot(index='date', columns=['code'], values='normalised_close')
        pair_prices = pair_prices.assign(spread=(pair_prices[self.pair_1] / pair_prices[self.pair_2])).reset_index()
        pair_prices = pair_prices.melt(id_vars=['date'], value_vars=[self.pair_1, self.pair_2, 'spread'], var_name='code', value_name='value')
        pair_prices = pair_prices.assign(year_date=pd.DatetimeIndex(pair_prices['date']).year).reset_index()

        p = (ggplot(aes(x='date', y='value', color='code'), data=pair_prices) + geom_line() + facet_wrap('year_date', scales='free'))
        print(p)

    def plot_dynamic_thresholds(self, start_date=None):
        pair_prices = self.get_spread()
        pair_prices = pair_prices.pivot(index='date', columns=['code'], values='value').reset_index()

        if start_date is not None:
            pair_prices = pair_prices[pair_prices['date'] >= start_date]

        # Boillinger band calculations
        sd = 2
        look_back = 20
        pair_prices['TP'] = (pair_prices['spread'] + pair_prices['spread'] + pair_prices['spread']) / 3
        pair_prices['std'] = pair_prices['TP'].rolling(look_back).std(ddof=0)
        pair_prices['MA-TP'] = pair_prices['TP'].rolling(look_back).mean()
        pair_prices['BOLU'] = pair_prices['MA-TP'] + sd * pair_prices['std']
        pair_prices['BOLD'] = pair_prices['MA-TP'] - sd * pair_prices['std']
        pair_prices = pair_prices.dropna()

        ax = pair_prices[['spread', 'BOLU', 'BOLD']].plot(color=['blue', 'orange', 'yellow'])
        ax.fill_between(pair_prices.index, pair_prices['BOLD'], pair_prices['BOLU'], facecolor='orange', alpha=0.1)
        plt.show()

if __name__ == '__main__':
    # Algorithm summary
    # 1. Calculate the ratio of stock prices (stock1 / stock2)
    # 2. Calculate SMA and standard deviation of the spread over an N day lookback (where N = 20, by default)
    # 2a. If ratio crosses SMA – x StdDev then buy stock 1 and sell stock 2 in equal $ amounts (where x = 2, by default)
    # 2b. If ratio crosses SMA + x StdDev then sell stock 1 and buy stock 2 in equal $ amounts (where x = 2, by default)
    # 3. Close all trades if ratio touches the SMA.
    simple_algorithm = SimpleAlgorithm()

    simple_algorithm.set_pair('FITB', 'MS')
    simple_algorithm.set_date(end_date=datetime.today().strftime('%Y-%m-%d'))
    simple_algorithm.set_data()
    # simple_algorithm.plot_prices()
    # simple_algorithm.plot_normalised_close()
    # simple_algorithm.plot_static_threshold()
    simple_algorithm.plot_dynamic_thresholds(start_date='2019-01-01')

