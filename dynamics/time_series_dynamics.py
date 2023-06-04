import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ggplot import ggplot, geom_line, aes

class TimeSeriesDynamics:
    """
    Synthetic prices for experiments
    """

    def __init__(self):
        pass

    def rnorm(self, n=1, mean=0, sd=1):
        return np.random.normal(mean, sd, n)[0]

    def gen_random_walk(self, sz, P0, bias, sigma):
        rw = [P0]
        for i in range(1, sz + 1):
            rw.append(rw[i - 1] + self.rnorm(1, bias, sigma))

        return rw

    def plot_random_walk(self, rw):
        plt.plot(rw, color='blue')
        plt.xlabel('index')
        plt.ylabel('price')
        plt.title('Random Walk Model of Price')
        plt.show()

    def plot_random_walks(self, rws):
        def get_cmap(n, name='hsv'):
            """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
            RGB color; the keyword argument name must be a standard mpl colormap name."""
            return plt.cm.get_cmap(name, n)

        cmap = get_cmap(len(rws))
        for i, rw in enumerate(rws):
            plt.plot(rw, c=cmap(i))

        plt.xlabel('index')
        plt.ylabel('price')
        plt.title('Synthetic Prices with Equal Vol')
        plt.show()

    def gen_common_trend(self,
                         dx=1,
                         dy=1,
                         trend_common=0,
                         vol_common=1,
                         Dx=1,
                         Dy=1,
                         trend_x=0,
                         vol_x=1,
                         trend_y=0,
                         vol_y=1,
                         Cx=1000,
                         Cy=1000,
                         noise_vol_x=1,
                         noise_vol_y=1,
                         sz=500
                         ):
        # common trend component parameters
        n = self.gen_random_walk(sz, 0., trend_common, vol_common)

        # specific trend component parameters
        Nx = self.gen_random_walk(sz, 10, trend_x, vol_x)
        Ny = self.gen_random_walk(sz, 10, trend_y, vol_y)

        # specific stationary component
        theta_x = Cx + self.rnorm(sz, 0., noise_vol_x)
        theta_y = Cy + self.rnorm(sz, 0., noise_vol_y)

        # synthetic price series
        x = dx * n + theta_x + Dx * Nx
        y = dy * n + theta_y + Dy * Ny

        spread = x / y
        idx = range(len(x))
        df = pd.DataFrame(dict(idx=idx, x=x, y=y, spread=spread, n=n, Nx=Nx, Ny=Ny))
        df['theta_x'] = theta_x
        df['theta_y'] = theta_y

        return df

    def plot_mct(self,
                trend_common=0,
                vol_common=1,
                dx=1,
                dy=1,
                trend_x=0,
                vol_x=1,
                trend_y=0,
                vol_y=1,
                Dx=1,
                Dy=1,
                noise_vol_x=1,
                noise_vol_y=1
                ):

        df = self.gen_common_trend(dx=dx,
                            dy=dy,
                            trend_common=trend_common,
                            vol_common=vol_common,
                            Dx=Dx,
                            Dy=Dy,
                            trend_x=trend_x,
                            vol_x=vol_x,
                            trend_y=trend_y,
                            vol_y=vol_y,
                            noise_vol_x=noise_vol_x,
                            noise_vol_y=noise_vol_y,
                            sz=5000)

        # pivot longer
        df = df.melt(id_vars=['idx', 'spread', 'n', 'Nx', 'Ny', 'theta_x', 'theta_y'], value_vars=['x', 'y'], var_name='code', value_name='price')

        p = (ggplot(aes(x='idx', y='price', color='code'), data=df) + geom_line())
        print(p)

if __name__ == '__main__':
    time_series_dynamics = TimeSeriesDynamics()

    rw_sz = 100
    P0 = 100.
    trend_common = 0
    vol_common = 1
    # rw = time_series_dynamics.gen_random_walk(rw_sz, P0, trend_common, vol_common)
    # time_series_dynamics.plot_random_walk(rw)

    # rws = []
    # n = 20
    # for i in range(n):
    #     rw = time_series_dynamics.gen_random_walk(rw_sz, P0, trend_common, vol_common)
    #     rws.append(rw)
    #
    # time_series_dynamics.plot_random_walks(rws)
    time_series_dynamics.plot_mct()