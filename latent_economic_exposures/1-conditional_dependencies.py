But the machine learning world is very broad these days. There are other tools we can use to both further validate our clusters and to quantify the strength of the relationships between stocks within clusters.

One such tool is the Graphical Lasso, which we can use to find structure and dependencies within our universe of stocks by estimating a sparse inverse covariance matrix.

Said differently, even though they’re correlated, they don’t explain one another’s returns. A third variable, the S&P500 (to which they both have a strong relationship), is the main driver of their correlation.

That’s where the Graphical Lasso comes in. It’s an algorithm for estimating a sparse inverse covariance matrix, where we can tune the sparsity of the matrix via a regularisation parameter.

### GRAPHICAL LASSO ###
#######################

# Calculate returns
ret <- prices_df %>%
  group_by(ticker) %>%
  tq_transmute(select=close, mutate_fun=periodReturn, period="daily")

# subset and process returns
subset_rets <- subset_returns("2015-01-01", "2020-01-01", ret)

##### EXTRACT CLUSTERS
# PCA
pc <- prcomp(scale(subset_rets), rank.=200)
summary(pc)

# apply DBSCAN to PCA loadings
scaled_pca_factor_df <- scale(pc$rotation, center=TRUE, scale=TRUE)
db_pca_final <- tune_eps_auto(scaled_pca_factor_df, seq(9, 11, 0.1))

# Create dataframe of clusters
cl <- as.data.frame(cbind(colnames(subset_rets), as.numeric(db_pca_final$cluster)), stringsAsFactors=FALSE)
colnames(cl) <- c("ticker", "cluster")
num_clusters <- length(unique(cl$cluster))

# Plot using t_SNE
tsne_pca <- Rtsne(na.omit(scaled_pca_factor_df), dims = 2, perplexity=30, verbose=TRUE, max_iter = 1000, normalize=FALSE, pca=FALSE)
plot_tsne_dbscan_clusters(tsne_pca, db_pca_final, paste0("t-SNE from PCA subset  data, eps=", db_pca_final$eps))

cluster_snapshot(subset_rets, db_pca_final$cluster)
# cluster 3: basic materials (gold miners)
# cluster 10: consumer cyclicals (residential construction)

# make covariance matrix on returns
S <- subset_rets %>%
  scale(center=TRUE, scale=TRUE) %>%
  cov(use='complete')

View(S[1:10, 1:10])

# get tickers for two interesting clusters
tickers <- cl[cl$cluster %in%  c(3, 10), 'ticker']
tickers %in% rownames(S)
tickers %in% colnames(subset_rets)


# make covariance matrix on returns
S <- subset_rets %>%
  scale(center=TRUE, scale=TRUE) %>%
  cov(use='complete')

View(S[1:10, 1:10])

# get tickers for two interesting clusters
tickers <- cl[cl$cluster %in%  c(3, 10), 'ticker']
tickers %in% rownames(S)
tickers %in% colnames(subset_rets)

As a first step, we’ll ask glasso to find the unregularised inverse covariance matrix. We wouldn’t expect this to do a great job of highlighting the known dependencies (or ndependencies) in our data:
# estimate precision matrix using glasso
# network estimation - we'll try to recover two unrelated clusters
# tune rho such that glasso recovers their dependency structure
rho = 0.
invcov <- glasso(S[tickers, tickers], rho=rho)
P <- invcov$wi
colnames(P) <- colnames(S[tickers, tickers])
rownames(P) <- colnames(P)
View(P)

One way to view the output is as a heatmap of the inverse covariance matrix (remember zeroes correspond to conditional independence):
# plot heatmap - zeros indicate conditional indpenendence
mycol <- colorRampPalette(c("blue", "cyan", "yellow", "orange", "red"))(ncol(P))
heatmap(abs(P), Rowv=NA, Colv=NA, col = mycol, RowSideColors = mycol, main=paste0("Precision Matrix, rho=", rho))

Here’s how to build a network graph from our sparse inverse covariance matrix, where we add aesthetics such as colouring by cluster and setting the edge thickness by relationship strength:
### build network graph
diag(P) <- 0
ig_wt <- graph_from_adjacency_matrix(abs(P), mode="undirected", weighted=TRUE)
# color by cluster
V(ig_wt)$cluster <- as.numeric(cl[cl$cluster %in% c(3, 10), 'cluster'])  # V() extracts vertices - we can then add attributes
cols <- c("deepskyblue2", "gold")
V(ig_wt)$color <- ifelse(V(ig_wt)$cluster==10, cols[1], cols[2])
# edge width by weight
E(ig_wt)$width <- E(ig_wt)$weight*10  # E() extract edges - we can then add attributes


For instance, we can get a quick overview of the network dependence structure by plotting our network graph with nodes arranged in a circle:
### plot network graph
# circle layout
par(mar=c(4, 1, 1, 1))
graph_attr(ig_wt, "layout") <- layout_in_circle
plot(ig_wt, vertex.size=22, vertex.label.color="gray10", main=paste0("rho=", rho, ": Circular Layout"))
legend(x=-1.5, y=-1.1, c("Consumer Cyclicals", "Basic Materials"), pch=21,
       col="#777777", pt.bg=cols, pt.cex=2, cex=.8, bty="n", ncol=1)

Another way to view the graph that highlights the strong connections is by using a force-directed algorithm. These algorithms model the strength of the connections as forces within a physical system (such as balls connected with springs, or electrically charged particles) and find an equilibrium state where the forces balance. This tends to highlight strong relationships.

There are a ton of ways to do this, but here’s one that works well for this data set (longer edges correspond to weaker relationships):
# use force directed layout - longer edges = smaller weights
wts <- abs(P[!upper.tri(P)])
wts <- wts[wts>0]
layout_with_wts <- layout_with_fr(ig_wt, weights=(wts/min(wts)))  # can resue this layout to plto evoution.
plot(ig_wt, vertex.size=30, layout=layout_with_wts, vertex.label.color="gray10") 
legend(x=-1.5, y=-1.1, c("Basic Materials","Consumer Cyclicals"), pch=21,
       col="#777777", pt.bg=cols, pt.cex=2, cex=.8, bty="n", ncol=1)


Some comments
We can see that for ρ=0.025 the inter-cluster connections are already starting to be severed. At ρ=0.125 we see them completely severed. Notice also that as we increase ρ, the strength of the connections, denoted by the thickness of the connecting edges, decreases, even when we’re reasonably sure there should be a strong connection.

So in practice, we find ourselves faced with something of a trade-off: we would like to tune ρ such that the clusters separate, but without losing the information expressed in the edge thickness. It seems that with ρ of around 0.125, we should be able to achieve that – at least for these two clusters.

For completeness, here’s what happens for higher values of ρ.