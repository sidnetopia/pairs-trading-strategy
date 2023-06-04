import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skbio.stats.ordination import pcoa

from strategies.pairs_trading.data.database.factory import InstrumentsDbFactory
from scipy.spatial.distance import pdist


class Clustering:
    def __init__(self):
        factory = InstrumentsDbFactory()
        self.prices_table = factory.get_table('prices')
        self.prices = None
        self.corr_mat = None
        self.dis = None

    def set_data(self):
        self.prices = self.prices_table.get_stocks_prices()
        # self.prices = pd.read_csv('test.csv')

    def set_daily_returns(self):
        self.prices['daily_return'] = self.prices.groupby(['code']).apply(lambda x: (x['close']/x['close'].shift(1)) - 1).reset_index()['close']
        self.prices.fillna(0.0)

    def get_corr_mat(self):
        self.prices = self.prices.pivot(index='date', columns=['code'], values='daily_return').reset_index()
        self.prices = self.prices.fillna(0.0)
        self.prices = self.prices.drop('date', 1)
        self.corr_mat = self.prices.corr()

        return self.corr_mat

    def transform_corr_mat(self):
        self.corr_mat = (1-np.abs(self.corr_mat))

    def calculate_dis_mat(self):
        self.transform_corr_mat()
        self.dis = pdist(self.corr_mat)
        # self.dis = distance_matrix(self.corr_mat, self.corr_mat.T)

        return self.dis

    def check_symmetric(self, tol=0.1):
        is_symmetric = np.all(np.abs(self.corr_mat-self.corr_mat) < tol)

        return is_symmetric

    def plot_elbow(self):
        """Shows that elbow is in the cluster of 2 which is off in our universe since we trade off many stocks"""

        ss = []
        # ss = [(self.corr_mat.shape[0]-1) * np.sum(self.corr_mat.apply(lambda x: np.var(x)))]
        for i in range(1, 15+1):
            kmeans = KMeans(n_clusters=i, init='k-means++')
            kmeans.fit(self.dis.reshape(-1,1))
            ss.append(kmeans.inertia_)

        plt.plot(range(1, 15+1), ss, '--bo')
        plt.title('K vs Cluster Sum Squared Distance')
        plt.xlabel('Number of clusters')
        plt.ylabel('Within-cluster sum of squares')
        plt.show()

    def plot_clusters(self):
        dis = distance_matrix(self.corr_mat, self.corr_mat.T, p=1)
        reduced_data = PCA(n_components=2).fit_transform(dis)
        # reduced_data = pcoa(dis)
        # print(reduced_data.eigvals())
        kmeans = KMeans(init="k-means++", n_clusters=2, n_init=4)
        kmeans.fit(reduced_data)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation="nearest",
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired, aspect="auto", origin="lower")

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
                    color="w", zorder=10)
        plt.title("K-means clustering on the digits dataset (PCA-reduced data)\n"
                  "Centroids are marked with white cross")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()

if __name__ == '__main__':
    clustering = Clustering()
    clustering.set_data()
    clustering.set_daily_returns()
    clustering.get_corr_mat()
    clustering.transform_corr_mat()
    clustering.calculate_dis_mat()
    clustering.plot_elbow()
    clustering.plot_clusters()