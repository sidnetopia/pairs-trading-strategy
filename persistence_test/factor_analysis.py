import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class FactorAnalysis:
    def __init__(self):
        self.lsr_df = None
        self.deciles = None

    def set_data(self):
        self.lsr_df = pd.read_csv('lsr_df.csv')

    def set_deciles(self):
        self.deciles = self.lsr_df[self.lsr_df['startofyear'] < '2018-01-01']
        1

    def add_mean_monthly_return(self, deciles):
        mean_monthly_return = deciles.groupby(['decile'])['mean_return'].mean()
        deciles = deciles.set_index(['decile'])
        deciles['mean_monthly_return'] = mean_monthly_return
        deciles.reset_index(level=0, inplace=True)

        return deciles

    def plot_factor(self, deciles=self.deciles):
        """
        Unsurprisingly, it looks like our “perfect factor plot” from earlier.

        That’s unsurprising because we are plotting the mean of the very same thing we are using to sort into the deciles. So if it didn’t look like this – with monotonically increasing returns for each decile – then we’d know we’d have done something wrong!

        Of note is that the spread between the best 10% of pairs and the worst 10% is nearly 12% a month. That’s a big spread!
        """
        deciles = self.add_mean_monthly_return(deciles)
        
        plt.bar(deciles['decile'], deciles['mean_monthly_return'])
        plt.ylabel('mean_monthly_return')
        plt.xlabel('decile')
        plt.show()
        # p = (ggplot(aes(x='decile', y='mean_monthly_return'), data=self.deciles) + geom_bar(position='fill'))
        # print(p)

    def plot_forward_return_factor(self):
        """
        If backtest performance is predictive of future performance, then we would expect to see our factor plot to have a “similar” shape to that one above.
        Of course, it’s never going to look that good – but we expect/want it to have that shape.

        Output should have a clear monotonic increase in future pairs trading profitability for each of the deciles.
        """
        deciles = self.add_mean_monthly_return(self.deciles)
        forward_returns = self.deciles.copy()
        
        deciles = pd.merge(deciles, forward_returns, on=['stock1', 'stock2'])
        deciles = add_mean_monthly_return(deciles)

        self.plot_factor(deciles)

if __name__ == '__main__':
    factor_analysis = FactorAnalysis()
    factor_analysis.set_data()
    factor_analysis.set_deciles()
    factor_analysis.plot_factor()