
import pandas as pd
from matplotlib.pyplot import plot as plt
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import scale
from scipy.special import comb


class CommonDriverExplorer:
    """ 
    extracting useful information. 
    Guide: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html  
    
    Analyse stocks by uncovering the common drivers using PCA. 
    """

    def __init__(self):
        factory = InstrumentsDbFactory()
        self.prices_table = factory.get_table('prices')
        self.prices = None
        self.ret_wide = None
        self.dis = None
        self.reduced_data = None 
        self.pca = PCA()
        self.spca = SparsePCA()

    def set_data(self):
        self.prices = self.prices_table.get_stocks_prices()

    def set_daily_returns(self):
        self.prices['daily_return'] = self.prices.groupby(['code']).apply(lambda x: (x['close']/x['close'].shift(1)) - 1).reset_index()['close']
        self.prices.fillna(0.0)

    def set_wide_data(self):
        ret = self.prices.copy()
        self.ret_wide = ret.pivot(index='date', columns=['code'], values='daily_return').reset_index()['date']

    def get_pca(self):
        self.reduced_data = self.pca.fit_transform(self.ret_wide)

    def plot_eigenvectors(labels=None):
        """ Taken from  https://stackoverflow.com/a/50845697 """

        score = self.reduced_data[:,0:2]
        coeff = np.transpose(self.pca.components_)

        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        plt.scatter(xs * scalex,ys * scaley, c = y)
        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
            if labels is None:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
            else:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.grid()

        #Call the function. Use only the 2 PCs.
        myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
        plt.show()

    def plot_component(self):
        """ 
        Taken from https://vitalflux.com/pca-explained-variance-concept-python-example/ 
        see that the first Principal Component (PC) explained a little over 25% of the variance in the data. It’s probably safe to assume that this PC is a proxy for the general stock market, which we call “market beta
        """

        exp_var_pca = self.reduced_data.explained_variance_ratio_
        #
        # Cumulative sum of eigenvalues; This will be used to create step plot
        # for visualizing the variance explained by each principal component.
        #
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        #
        # Create the visualization plot
        #
        plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
        plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def display_eigenvalues(self):
        print(self.pca.components_)
        print(self.pca.explained_variance_)

    def explain_similar_exposures(self):
        """ Explain similar exposures using PCA """
        # which(pc$rotation['XOM', ] == max(pc$rotation['XOM', ]))
        # # PC4 
        # #  4 
        # which(pc$rotation['CVX', ] == max(pc$rotation['CVX', ]))
        # #PC4 
        # #  4 
        # which(pc$rotation['COP', ] == max(pc$rotation['COP', ]))
        # # PC4 
        # #  4

        # top_xom <- sort(pc$rotation['XOM', ], decreasing=TRUE)[1:20]
        # top_cvx <- sort(pc$rotation['CVX', ], decreasing=TRUE)[1:20]
        # top_cop <- sort(pc$rotation['COP', ], decreasing=TRUE)[1:20]

        # sum(names(top_xom) %in% names(top_cvx))
        # # 6
        # sum(names(top_xom) %in% names(top_cop))
        # # 3
        # sum(names(top_cvx) %in% names(top_cop))
        # # 4
        pass

    def get_spca(self):
        """
        Interpretation is simpler
        Clustering algorithms would likely find it easier to identify clusters, since we’d be zeroing noisy, fleeting loadings (to which regular PCA assigns a value)
        We have a large number of variables (stocks) in our universe, which tends to invite overfitting. Sparsity in the loadings matrix is something of a counter to this.
        It aligns with the principle of Occam’s Razor, where parsimonious models are preferred over needlessly complex ones
        """        
        self.spca.fit(self.prices)

    def display_sparse_pca(self):
        print(self.spca.components_)
        print(self.spca.explained_variance_)

    def get_dbscan(self, eps_to_try):
        db = []

        for e in eps_to_try:
            print("doing eps=", e)
            db.append(DBSCAN(eps=3, min_samples=2).fit(X))
            print(db)

        return db
    
    def get_tsne_results(self, scaled_spca_factor_df):
        """
        t-Distributed Stochastic Neighbor Embedding.
        Visualize cluster and cross-check with DBSCAN.
        Useful clusters should be identified in both algorithms
        """
        scaled_spca_factor_df = scaled_spca_factor_df.dropna()

        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(scaled_spca_factor_df)

        return tsne_results

    def plot_tsne_clusters(self, tsne_results, eps_count):
        clusters = int(eps)

        df_subset['tsne-2d-one'] = tsne_results[:,0]
        df_subset['tsne-2d-two'] = tsne_results[:,1]
        plt.figure(figsize=(16,10))

        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", clusters),
            data=df_subset,
            legend="full",
            alpha=0.3
        )

    def plot_tsne_dbscan_clusters(tsne_object, dbscan_object, title):
        tsne_coords = tsne_object.Y 
        tsne_coords.column_name = ['d1', 'd2']
        tsne_df = as.data.frame(tsne_coords)
        tsne_df['cluster'] = as.factor(dbscan_object.cluster)
        
        tsne_df['alpha'] = tsne_df['cluster'].apply(lambda x: 0.1 if x == 0 else 1)

        p = (ggplot(aes(x='d1', y='d2', color='code'), data=df) +  
            geom_point(aes(color='cluster', alpha='alpha')) +
            scale_alpha(guide='none') +
            ggtitle(title)
            )
        print(p)

    def cluster_to_text(dbscan_object, file_path, filename_prefix):
        """
        Creates cluster of stocks e.g. AEE, AEP, CMS, etc.
        This will serves a text file to import to watchlist in trading view
        """

        cl = as.data.frame(cbind(colnames(ret_wide), as.numeric(dbscan_object$cluster)), stringsAsFactors=FALSE)
        colnames(cl) <- c("ticker", "cluster")
        num_clusters <-length(unique(cl$cluster))
        
        for i in range(0, num_clusters-1):
            file_name = f"{file_path}{filename_prefix} {cluster} {i}{.txt}"
            cl[cl$cluster==i, "ticker"].to_csv(file_name)


    def cluster_instruments(self):
        for i in range(0, num_clusters-1):
            snapshot_df = cl[cl$cluster==i, 'ticker'] # crosscheck with ticker in snapshot_fd
            snapshot_df["ticker", "type", "sector", "industry", "location", "marketcap"]
            # print(n=Inf)infinity

    def extract_cluster_sizes(self, clusters):
        num_clusters = 

        cl_sizes = []
        for i in range(0, num_clusters-1):
            cl_sizes.append(cl_sizes, sum(clusters==i)) # combine

        return cl_sizes

    def pairwise_combinations_all_cluster(self, cluster_sizes):
        """
        Determine number of unique pairwise combinations across all clusters
        """
        combinations = 0
        for i in range(0, cluster_sizes):
            combinations = combinations + comb(i, 2, exact=False)

        return combinations

    def subset_returns(self, start_date, end_date, returns, max_nans=250):
        """
        spread into df with one observation per date for each ticker, select all columns except date
        """

        subset_rets = returns[returns['date'] >= start_date & returns['date'] < start_date]
        subset_rets = subset_rets.pivot(index='date', columns=['code'], values='daily_return').reset_index()
        subset_rets = subset_rets.drop('date', 1)

        stocks_to_drop = subset_rets.loc[:, stocks_to_drop.isnull().sum() >= max_nans]
        print("Stocks to drop shape ", stocks_to_drop.shape)
        subset_rets = subset_rets.loc[:, stocks_to_drop.isnull().sum() < max_nans]
        for i in range(0, subset_rets.shape[1]):
            subset_rets.iloc[i] = subset_rets.iloc[i].fillna((subset_rets.iloc[i].mean()), inplace=True)
        
        return subset_rets

    def tune_eps_auto(self, dat, eps_to_try):
        # best_cl_metric <- Inf
        # best_db <- NA
        # for(e in eps_to_try) {
        #     print(paste0("doing eps=", e))
        #     this_db <- dbscan::dbscan(dat, eps=e, minPts=5)
            
        #     cl <- as.data.frame(cbind(colnames(ret_wide), as.numeric(this_db$cluster)), stringsAsFactors=FALSE)
        #     colnames(cl) <- c("ticker", "cluster")
        #     num_clusters <- length(unique(cl$cluster))
            
        #     cl_sizes <- c()
        #     for(i in c(1:(num_clusters-1))) {
        #     cl_sizes <- c(cl_sizes, sum(cl$cluster==i))
        #     }
        #     # reward more smaller clusters and more total points in clusters
        #     # penalise large clusters
        #     if (max(cl_sizes) > 100)
        #     this_cl_metric <- Inf
        #     else
        #     this_cl_metric <- (max(cl_sizes))/((sum(cl_sizes)*num_clusters)**2)
            
        #     print(this_db)
        #     print(this_cl_metric)
        #     if(this_cl_metric < best_cl_metric) {
        #     print(paste0("new best metric: ", this_cl_metric))
        #     print(paste0("new best eps:", e))
        #     best_cl_metric <- this_cl_metric
        #     best_db <- this_db
        #     }
        # }
        # return(best_db)

    def analysis_pipeline(self):
        # ##### Pipeline

        # ### subset and process returns
        # subset1_rets <- subset_returns("1999-01-01", "2005-01-01", ret)
        # subset2_rets <- subset_returns("2005-01-01", "2010-01-01", ret)
        # subset3_rets <- subset_returns("2010-01-01", "2015-01-01", ret)
        # subset4_rets <- subset_returns("2015-01-01", "2020-01-01", ret)
        # return_subsets <- list(subset1_rets, 
        #                     subset2_rets, 
        #                     subset3_rets,
        #                     subset4_rets)

        # ### Cluster plotting and extraction
        # cluster_subsets <- list()
        # subset_num <- 0
        # for(return_subset in return_subsets) {
        # subset_num <- subset_num + 1
        # # PCA
        # pc <- prcomp(scale(return_subset), rank.=200)
        # summary(pc)
        
        # # Apply DBSCAN to PCA loadings
        # scaled_pca_factor_df <- scale(pc$rotation, center=TRUE, scale=TRUE)
        # db_pca <- tune_eps_auto(scaled_pca_factor_df, seq(5, 15, 0.1))
        
        # # Plot using t_SNE
        # tsne_pca <- Rtsne(na.omit(scaled_pca_factor_df), dims = 2, perplexity=30, verbose=TRUE, max_iter = 1000, normalize=FALSE, pca=FALSE)
        # print(plot_tsne_dbscan_clusters(tsne_pca, db_pca, paste0("t-SNE from PCA subset ", subset_num,  ", eps=", db_pca$eps)))
        
        # # create cluster dataframe
        # cl <- as.data.frame(cbind(colnames(return_subset), as.numeric(db_pca$cluster)), stringsAsFactors=FALSE)
        # colnames(cl) <- c("ticker", "cluster")
        # cluster_subsets <- c(cluster_subsets, list(cl))
        # }

    def count_cluster_sectors(self):
        ### unique sectors in each cluster
        # cluster_sectors <- list()
        # for(cluster_subset in cluster_subsets) {
        # these_clusters <- list()
        # num_clusters <-length(unique(cluster_subset$cluster))
        # for(i in 1:(num_clusters-1)) {  # skip zeroth cluster
        #     print(paste0("Cluster ", i, " unique sectors"))
        #     these_clusters[[i]] <- snapshot_df %>% 
        #     filter(ticker %in% cluster_subset[cluster_subset$cluster==i, 'ticker']) %>%
        #     group_by(sector) %>%
        #     summarise(count = n())
        # }
        # cluster_sectors <- c(cluster_sectors, list(these_clusters))
        # }

        # ### store list of dataframes of sector counts for each cluster
        # sector_counts <- list()
        # for(subset_sectors in cluster_sectors) {
        # # extract unique sectors from subset
        # num_clusters <- length(subset_sectors)
        # sectors <- c()
        # for(i in 1:num_clusters) {
        #     sectors <- c(sectors, subset_sectors[[i]][[1]])
        # }
        # sectors <- unique(sectors)
        # num_sectors <- length(sectors)
        
        # # initalise dataframe for count of sectors per cluster
        # df <- data.frame(matrix(0, ncol=num_sectors, nrow=num_clusters))
        # colnames(df) <- sectors
        
        # # populate sector count dataframe
        # for(i in 1:num_clusters) {
        #     these_sectors <- subset_sectors[[i]][[1]]
        #     these_counts <- subset_sectors[[i]][[2]]
        #     for(j in 1:length(these_sectors)) {
        #     df[i, these_sectors[j]] <- these_counts[j] 
        #     }
        # }
        # sector_counts <- c(sector_counts, list(df))
        # }

    def visualize_sectors(self):
        ### plot heatmaps of cluster sector uniqueness by subset
        # subset_num <- 0
        # for(subset_sector_counts in sector_counts) {
        # subset_num <- subset_num + 1
        
        # subset_sector_counts$cluster <- as.factor(rownames(subset_sector_counts))
        # subset_sector_counts_long <- pivot_longer(tbl_df(subset_sector_counts), -cluster, names_to = 'sector', values_to = 'count')
        # subset_sector_counts_long[subset_sector_counts_long$count==0, "count"] <- NA
        
        # p <- ggplot(subset_sector_counts_long, aes(x=sector, y=cluster)) +
        #     geom_tile(aes(fill=count)) +
        #     geom_text(aes(label=count)) +
        #     scale_fill_gradient(low="steelblue1", high="steelblue4", na.value = "grey90") + 
        #     theme(axis.text.x=element_text(angle=45, hjust=1)) +
        #     ggtitle(paste0("Sectors represented by cluster, subset ", subset_num))
        # print(p)
        # }

if __name__ == "__main__":
    common_driver_explorer = CommonDriverExplorer()
    common_driver_explorer.set_data()
    common_driver_explorer.set_daily_returns()
    common_driver_explorer.set_wide_data()
    common_driver_explorer.get_pca()
    common_driver_explorer.plot_component()
    common_driver_explorer.plot_eigenvectors()
    common_driver_explorer.display_eigenvalues()
    common_driver_explorer.explain_similar_exposures()

    # apply DBSCAN to SPCA loadings TODO: convert the lines below to python
    # scaled_spca_factor_df <- scale(spc$loadings, center=TRUE, scale=TRUE)
    # db_spca <- tune_eps(scaled_spca_factor_df, seq(1, 10, 0.5))
    # TODO: explore eps of 6.5 and fine tune i.e. db_spca <- tune_eps(scaled_spca_factor_df, seq(6, 7, 0.1)). This should have a reasonable value when eps is 6.4
    # scaled_spca_factor_df = scale(common_driver_explorer.spca$loadings)
    db = common_driver_explorer.get_dbscan()
    print(db)
    tsne_results = common_driver_explorer.get_tsne_results(scaled_spca_factor_df)
    # plot_tsne_dbscan_clusters(tsne, db_spca_final, paste0("t-SNE from SPCA all data, eps=", db_spca_final$eps))
    
    # cluster_to_text(db_spca_final, "C:/Users/Kris/Documents/rw-ml-bootcamp/unsupervised-learning/", "spca_all_data")


    # TODO: test functions of class. Haven't test because of memory constrain

    # TODO: cluster_sizes <- lapply(list(db_pca_final_100$cluster,
    #                          db_pca_final_50$cluster,
    #                          db_pca_final_20$cluster,
    #                          db_pca_final_10$cluster),
    #                     extract_cluster_sizes)
    # total_combns <- lapply(cluster_sizes, 
    #                     pairwise_combinations_all_clusters)
    # [[1]]
    # [1] 7667
    #
    # [[2]]
    # [1] 7122
    #
    # [[3]]
    # [1] 11563
    #
    # [[4]]
    # [1] 5088

    # ANALYSING THE STABILITY OF PCA
    # The top 50 explained 63%.  To get similar numbers on a 5-year subset, we need about 300 and 200 PCs respectively.
    # Some sectors are represented in every subset – notably real estate, financial services, basic materials and consumer cyclicals.
    # Other sectors are represented in most subsets – notably energy, industrials and technology.
    # Interestingly, basic materials (whose constituents included gold and silver, and sometimes copper and steel) didn’t show up much when we looked at the entire data set, but did show up on our subsets.
    # Even when a sector is represented across multiple subsets, the constituents change – but part of that variation may be due to random variation associated with tuning the clustering algorithm. In practice, we’d probably bulk up the clusters using an ensemble and then whittle them down using backtesting.
    # The homogeneity of clusters is generally quite good across all subsets.

    # Validating the clustering on PCA
    # quite a few NAs
    # lsr_df[is.na(lsr_df$lb20stdev)|is.na(lsr_df$lb60stdev),]
    # sum(is.na(lsr_df$lb20stdev)|is.na(lsr_df$lb60stdev))/nrow(lsr_df)
    # [1] 0.003500049
    #     Which performance metrics should we use to validate our approach?

    # It doesn’t really matter at this stage which performance metric we choose – we’d expect our validation not to depend on the choice of metric. We could either use mean strategy returns (the variables lb20mean and lb60mean) or we could calculate a pseudo-Sharpe ratio by dividing the mean return by the standard deviation of returns.
    # The latter is a more useful metric, as it incorporates straetgy volatility, which we’re certainly interested in. The downside is that we’ll necessarily have to drop some observations due to the NA values in the standard deviation variables – but as we’ve already seen, this is of minor concern.
    # It’s conceivable that this can occur when strategies with a positive mean have low standard deviation of returns, and conversely strategies with a negative mean tend to have higher standard deviation of returns. Such a scenario is going to lead to positive Sharpes having higher magnitude than negative Sharpes, on average.

    # Co-ordinates in t-SNE space
    # tsne_coords <- tsne_pca$Y
    # rownames(tsne_coords) <- colnames(return_subset)

    # Distance between points in t-SNE space
    # tsne_dist <- as.matrix(dist(tsne_coords))
    # tsne_dist <- tsne_dist %>% 
    # tbl_df() %>%
    # mutate(stock1 = rownames(tsne_dist)) %>%
    # pivot_longer(-stock1, names_to = "stock2", values_to = "distance") %>%
    # filter(stock1 != stock2)
    # head(tsne_dist)
    # #   stock1 stock2   distance
    # #   <chr>  <chr>     <dbl>
    # # 1 A      AAL       23.3 
    # # 2 A      AAN       20.7 
    # # 3 A      AAP       15.2 
    # # 4 A      AAPL      19.5 
    # # 5 A      AAXN      14.2 
    # # 6 A      ABB        4.94

    # tsne_dist_perf <- lsr_df %>%
    # filter(startofyear >= as.Date("2014-01-01") & startofyear < as.Date("2015-01-01")) %>%
    # select(stock1, stock2, startofyear, lb20mean, lb60mean) %>%
    # left_join(tsne_dist, by=c("stock1"="stock1", "stock2" = "stock2"))
    # head(tsne_dist_perf)
    # #   stock1 stock2 startofyear lb20mean  lb60mean  distance
    # #   <chr>  <chr>  <date>         <dbl>    <dbl>    <dbl>
    # # 1 A      AAL    2014-01-01  -0.00598  -0.0581    23.3 
    # # 2 A      AAP    2014-01-01   0.0316    0.0292    15.2 
    # # 3 A      AAXN   2014-01-01  -0.0849   -0.0580    14.2 
    # # 4 A      ABB    2014-01-01   0.0597    0.0447     4.94
    # # 5 A      ABBV   2014-01-01   0.0143    0.0129    15.4 
    # # 6 A      ACAD   2014-01-01  -0.0314    0.0218    22.6

    # tsne_dist_perf %>%
    # ggplot(aes(x=distance, y=lb20mean)) +
    # geom_point(alpha=0.3) +
    # geom_smooth(method='lm')

    # tsne_dist_perf %>%
    # ggplot(aes(x=distance, y=lb20mean)) +
    # stat_binhex() +
    # geom_smooth(method='lm')

    # That’s a cool plot, but again we see that on average the relationship between t-SNE distance and pairs trading performance is essentially zero, on average.

    # You seem to be making a point of saying “on average.” Why? But there’s a solution to this problem: the quantile or factor plot.

    # In a quantile plot, we take our factor – in this case, distance – and split it into percentiles. That is, the bottom x% of distance values get grouped together in the bottom percentile. The next x% of values go into the next bucket and so on. You’ll often see four or five percentiles used, but you can use any number, and it really depends on the application.

    # tsne_dist_perf %>%
    # mutate(quantile = cut(distance, quantile(distance, seq(0, 1, 0.05), na.rm=T), labels=F)) %>%
    # group_by(quantile) %>%
    # summarise(mean_ret20 = mean(lb20mean)) %>%
    # na.omit() %>%
    # ggplot(aes(x=quantile, y=mean_ret20)) +
    # geom_col()

    # That’s better! We can clearly see that in the tails of our data, we have something interesting going on. Look at the bottom two percentiles. We used 20 percentiles in total here, so the bottom two correspond to the bottom 10% of distance values. That is, the 10% of pairs that were closest in t-SNE space.

    #     That’s a really excellent point. Let’s see if it holds up on the next year of backtest results, which weren’t used at all in calculating the t-SNE co-ordinates:
    # tsne_dist_perf <- lsr_df %>%
    #   filter(startofyear >= as.Date("2015-01-01") & startofyear < as.Date("2016-01-01")) %>%
    #   select(stock1, stock2, startofyear, lb20mean, lb60mean) %>%
    #   left_join(tsne_dist, by=c("stock1"="stock1", "stock2" = "stock2"))
    # tsne_dist_perf %>%
    #   mutate(quantile = cut(distance, quantile(distance, seq(0, 1, 0.05), na.rm=T), labels=F)) %>%
    #   group_by(quantile) %>%
    #   summarise(mean_ret20 = mean(lb20mean)) %>%
    #   na.omit() %>%
    #   ggplot(aes(x=quantile, y=mean_ret20)) +
    #   geom_col()


    #     That’s a really pleasing result. The closest stocks in the in-sample were the only ones to show positive returns in the out-of-sample backtest. In fact, the out-of-sample factor plot looks nicer than the in-sample, but that’s probably just due to randomness.

    # Let’s repeat this process for each year in our backtest data (we’ll also save a dataframe of clusters for each subset for use in the next part of the analysis):
    # ### repeat process for all years in backtest set
    # start_dates <- c("2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01")
    # end_dates <- c("2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01")
    # cluster_subsets <- list()
    # for(i in 1:(length(start_dates)-1)) {
    # return_subset <- subset_returns(start_dates[i], end_dates[i], ret, max_nans=125)
    # # PCA
    # pc <- prcomp(scale(return_subset), rank.=50)  # explains ~63% variance
    # # Apply DBSCAN to PCA loadings
    # scaled_pca_factor_df <- scale(pc$rotation, center=TRUE, scale=TRUE)
    # db_pca <- tune_eps_auto(scaled_pca_factor_df, seq(2, 13, 0.1))
    # # Plot using t_SNE
    # tsne_pca <- Rtsne(na.omit(scaled_pca_factor_df), dims = 2, perplexity=30, verbose=TRUE, max_iter = 1000, normalize=FALSE, pca=FALSE)
    # print(plot_tsne_dbscan_clusters(tsne_pca, db_pca, paste0("t-SNE from PCA subset ", start_dates[i], ", eps=", db_pca$eps)))
    # # Co-ordinates in t-SNE space
    # tsne_coords <- tsne_pca$Y
    # rownames(tsne_coords) <- colnames(return_subset)
    # # Distance between points in t-SNE space
    # tsne_dist <- as.matrix(dist(tsne_coords))
    # tsne_dist <- tsne_dist %>% 
    #     tbl_df() %>%
    #     mutate(stock1 = rownames(tsne_dist)) %>%
    #     pivot_longer(-stock1, names_to = "stock2", values_to = "distance") %>%
    #     filter(stock1 != stock2)
    
    # # in-sample performance
    # tsne_dist_perf <- lsr_df %>%
    #     filter(startofyear >= as.Date(start_dates[i]) & startofyear < as.Date(end_dates[i])) %>%
    #     select(stock1, stock2, startofyear, lb20mean, lb60mean) %>%
    #     left_join(tsne_dist, by=c("stock1"="stock1", "stock2" = "stock2"))
    
    # ins_plot <- tsne_dist_perf %>%
    #     mutate(quantile = cut(distance, quantile(distance, seq(0, 1, 0.05), na.rm=T), labels=F)) %>%
    #     group_by(quantile) %>%
    #     summarise(mean_ret20 = mean(lb20mean)) %>%
    #     na.omit() %>%
    #     ggplot(aes(x=quantile, y=mean_ret20)) +
    #     geom_col() +
    #     ggtitle(paste0("In-sample factor plot ", start_dates[i]))
    # print(ins_plot)
    
    # # out-of-sample performance
    # tsne_dist_perf <- lsr_df %>%
    #     filter(startofyear >= as.Date(start_dates[i+1]) & startofyear < as.Date(end_dates[i+1])) %>%
    #     select(stock1, stock2, startofyear, lb20mean, lb60mean) %>%
    #     left_join(tsne_dist, by=c("stock1"="stock1", "stock2" = "stock2"))
    
    # outs_plot <- tsne_dist_perf %>%
    #     mutate(quantile = cut(distance, quantile(distance, seq(0, 1, 0.05), na.rm=T), labels=F)) %>%
    #     group_by(quantile) %>%
    #     summarise(mean_ret20 = mean(lb20mean)) %>%
    #     na.omit() %>%
    #     ggplot(aes(x=quantile, y=mean_ret20)) +
    #     geom_col() +
    #     ggtitle(paste0("Out-of-sample factor plot ", start_dates[i]))
    # print(outs_plot)
    
    # # save cluster dataframes for next part
    # cl <- as.data.frame(cbind(colnames(return_subset), as.numeric(db_pca$cluster)), stringsAsFactors=FALSE)
    # colnames(cl) <- c("ticker", "cluster")
    # cluster_subsets <- c(cluster_subsets, list(cl))

    #     Here’s some code that takes the clusters found for the first data subset in the previous analysis and:

    # extracts the intra-cluster pairs
    # extracts the performance metrics for those pairs from lsr_df for the corresponding in-sample and out-of-sample years
    # calculates the mean return to each cluster in our heuristic backtest in the in-sample and out-of-sample years, as well as the number of pairs in each cluster
    # subset_num <- 1
    # cl <- cluster_subsets[[subset_num]]
    # num_clusters <- length(unique(cl$cluster))
    # traded_pairs <- list()
    # is_mean_returns <- list()
    # oos_mean_returns <- list()
    # num <- 0
    # for(i in 1:(num_clusters-1)) {  #skip zeroth
    #   num <- num + choose(length(cl[cl$cluster==i, "ticker"]), 2)  # to cross check correct length
    #   traded_pairs <- bind_rows(traded_pairs, lsr_df %>%
    #     filter(stock1 %in% cl[cl$cluster==i, 'ticker']
    #            & stock2 %in% cl[cl$cluster==i, 'ticker']
    #            & startofyear %in% c(as.Date(start_dates[subset_num]), as.Date(start_dates[subset_num+1]))))
    
    #   is_mean_returns <- c(is_mean_returns, 
    #                        list(lsr_df %>% 
    #                               filter(stock1 %in% cl[cl$cluster==i, 'ticker'] 
    #                                      & stock2 %in% cl[cl$cluster==i, 'ticker']
    #                                      & startofyear == as.Date(start_dates[subset_num])) %>%
    #                               summarise(meanret20 = mean(lb20mean), count=n())))
    
    #   oos_mean_returns <- c(oos_mean_returns, 
    #                         list(lsr_df %>% 
    #                                filter(stock1 %in% cl[cl$cluster==i, 'ticker'] 
    #                                       & stock2 %in% cl[cl$cluster==i, 'ticker']
    #                                       & startofyear == as.Date(start_dates[subset_num+1])) %>%
    #                                 summarise(meanret20 = mean(lb20mean), count=n())))
    # }
    # is_mean_returns
    # oos_mean_returns


    #     Next, we plot histograms of the of the mean returns to each of our traded pairs in the in-sample and out-of-sample periods:
    # # histogram of mean returns of traded pairs
    # traded_pairs %>%
    #   ggplot(aes(x=lb20mean)) +
    #   geom_histogram(bins=100) +
    #   facet_wrap(~startofyear) +
    #   ggtitle("Histogram of mean returns - all traded pairs")

    #     Nice! They both have a slight skew to the positive side of the x-axis, inline with their positive mean returns.

    # But that’s kind of meaningless without comparing it to the distribution of returns in the wider universe:
    # # histogram of mean returns of wider universe
    # lsr_df %>%
    #   filter(startofyear %in% c(as.Date(start_dates[subset_num]), as.Date(start_dates[subset_num+1]))) %>%
    #   ggplot(aes(x=lb20mean)) +
    #   geom_histogram(bins=100) +
    #   facet_wrap(~startofyear) +
    #   ggtitle("Histogram of mean returns - entire universe")

    #     It’s a little hard to see,  but these don’t exhibit the same return skew as our intra-cluster pairs. This is more easily seen by summarising the mean returns from the two samples:
    # # mean of mean returns of traded pairs
    # traded_pairs %>%
    #   group_by(startofyear) %>%
    #   summarise(mean_meanret20 = mean(lb20mean))
    # # A tibble: 2 x 2
    # #   startofyear       mean_meanret20
    # #   <date>               <dbl>
    # # 1 2014-01-01         0.00557
    # # 2 2015-01-01         0.00117
    # # mean of mean return of wider universe
    # lsr_df %>%
    #   filter(startofyear %in% c(as.Date(start_dates[subset_num]), as.Date(start_dates[subset_num+1]))) %>%
    #   group_by(startofyear) %>%
    #   summarise(mean_meanret20 = mean(lb20mean))
    # # A tibble: 2 x 2
    # #   startofyear       mean_meanret20
    # #   <date>               <dbl>
    # # 1 2014-01-01        -0.00255
    # # 2 2015-01-01        -0.00256

    # repeat the cluster validation process that I started here by running the code above for the remaining four data subsets. On Slack, in the #2-unsupervised-learning-pairs-selection channel, share your thoughts on the following questions:

    # Do all the data subsets validate as nicely as this first one?
    # What are the implications of this for our clustering workflow?
    # Do you think our assumption that clustering on statistical factors can help identify pairs that are not only similar, but are also profitable, is a valid one?
    # Is there remaining uncertainty? Or are things very clear cut and obvious? How do you feel about that?
