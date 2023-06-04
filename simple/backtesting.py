from datetime import datetime

from strategies.pairs_trading.expermients.simple.simple_algorithm import SimpleAlgorithm
from fastquant import backtest as backtest_fq

class BackTest:
    def __init__(self):
        self.simple_algorithm = SimpleAlgorithm()

    def __format_to_backtest(self, df):
        df = df[['date', 'spread']]
        df.columns = ['dt', 'close']

        return df

    def perform_backtest(self, pair_1='FITB', pair_2='MS', start_date='2012-01-01', end_date=datetime.today().strftime('%Y-%m-%d')):
        self.simple_algorithm.set_pair(pair_1, pair_2)
        self.simple_algorithm.set_date(start_date=start_date, end_date=end_date)
        self.simple_algorithm.set_data()
        spread_df = self.simple_algorithm.get_spread()
        spread_df = spread_df.pivot(index='date', columns=['code'], values='value').reset_index()
        spread_df = self.__format_to_backtest(spread_df)
        init_eq = 1000
        sd = 2
        n = 20

        res = backtest_fq("bbands", spread_df, init_cash=init_eq, period=n, devfactor=sd)
        print(res)


if __name__ == '__main__':
    backtest = BackTest()
    backtest.perform_backtest()
    backtest.perform_backtest('KO', 'PEP', start_date='2012-01-01')
    backtest.perform_backtest('UFS', 'ITT', start_date='2012-01-01')
    backtest.perform_backtest('EXP', 'HDS', start_date='2012-01-01')
    backtest.perform_backtest('CVLT', 'HRTX', start_date='2012-01-01')
    backtest.perform_backtest('WERN', 'GNRC', start_date='2012-01-01')

