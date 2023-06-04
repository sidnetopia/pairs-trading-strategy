###### APPLY GLASSO TO FULL DATA SET #####
##########################################

# Calculate returns
ret <- prices_df %>%
  group_by(ticker) %>%
  tq_transmute(select=close, mutate_fun=periodReturn, period="daily")

# make wide dataframe of returns
ret_wide <- ret %>%
  spread(key=ticker, value=daily.returns) %>% 
  select(-date)

# subset and process returns - drop anything with lots of NA, fill remaining NA with mean
subset_rets <- subset_returns("1997-01-01", "2020-01-01", ret, max_nans = 250)
pc <- prcomp(scale(subset_rets), rank.=50)  
summary(pc)

# apply DBSCAN to PCA loadings
scaled_pca_factor_df <- scale(pc$rotation, center=TRUE, scale=TRUE)
db_pca_final <- tune_eps_auto(scaled_pca_factor_df, seq(2, 7, 0.1))

# Create dataframe of clusters
cl <- as.data.frame(cbind(colnames(subset_rets), as.numeric(db_pca_final$cluster)), stringsAsFactors=FALSE)
colnames(cl) <- c("ticker", "cluster")
num_clusters <- length(unique(cl$cluster))

# Plot using t_SNE
tsne_pca <- Rtsne(na.omit(scaled_pca_factor_df), dims = 2, perplexity=30, verbose=TRUE, max_iter = 1000, normalize=FALSE, pca=FALSE)
plot_tsne_dbscan_clusters(tsne_pca, db_pca_final, paste0("t-SNE from PCA all  data, eps=", db_pca_final$eps))

cluster_snapshot(subset_rets, db_pca_final$cluster)

### Apply glasso

# get back returns without nan filled with mean
subset_rets <- ret %>%
  filter(ticker %in% colnames(subset_rets)) %>%
  spread(key=ticker, value=daily.returns) %>% 
  select(-date)

# make covariance matrix on returns
S <- subset_rets %>%
  scale(center=TRUE, scale=TRUE) %>%
  cov(use='p')
View(S[1:10, 1:10])

# estimate precision matrix using glasso
rho = 0.5
invcov <- glasso(S, rho=rho)  # This gives invcov of entire universe - takes some time to compute, depending on rho
P <- invcov$wi
colnames(P) <- colnames(S)
rownames(P) <- colnames(P)

# check symmetry
if(!isSymmetric(P)) {
  P[lower.tri(P)] = t(P)[lower.tri(P)]  
}

### build network graph

diag(P) <- 0
ig_wt <- graph_from_adjacency_matrix(abs(P), mode="undirected", weighted=TRUE)

# color by cluster
V(ig_wt)$cluster <- as.numeric(cl$cluster) 
cols <- colorRamps::primary.colors(n=num_clusters)
cols <- paste0(cols, "7F") # add 50% transparency to colors
V(ig_wt)$color <- cols[V(ig_wt)$cluster+1]
V(ig_wt)$frame.color <- "#0000007F"  # also make vertex outlines transparent

# edge width by weight
E(ig_wt)$width <- E(ig_wt)$weight  

### plot network graph
# use force directed layout - longer edges = smaller weights
wts <- abs(P[!upper.tri(P)])
wts <- wts[wts!=0]
layout_with_wts <- layout_with_fr(ig_wt, weights=(wts/(2000*min(wts))))
par(mar=c(4, 1, 1, 1))
plot(ig_wt, vertex.size=9, layout=layout_with_wts, vertex.label=NA, main=paste0("rho=", rho)) 
legend(x=-2., y=1.1, as.character(sort(as.numeric(unique(cl$cluster[cl$cluster!=0])))), pch=21,
       col="#777777", pt.bg=cols, pt.cex=1.25, cex=.75, bty="n", ncol=1, title="Cluster")


Keep in mind that we still don’t know how useful this evidence and additional information is (we’ll be exploring that in the next lesson). But still, this is a nice result because it provides further evidence that the clusters we found using DBSCAN and t-SNE are sensible. Stocks that the other approaches treated as similar do indeed tend to have stronger conditional dependencies.


# drop vertices with no edges
isolated <-  which(degree(ig_wt) == 0)
ig_wt = delete.vertices(ig_wt, isolated)
layout_with_wts <- layout_with_wts[-isolated, ]
plot(ig_wt, vertex.size=9, layout=layout_with_wts, vertex.label=NA, main=paste0("rho=", rho)) 
legend(x=-1.5, y=1.1, as.character(sort(as.numeric(unique(cl$cluster)))), pch=21,
       col="#777777", pt.bg=cols, pt.cex=1.25, cex=.75, bty="n", ncol=1, title="Cluster")


### make interactive graph

# reset transparency
cols <- colorRamps::primary.colors(n=num_clusters)
V(ig_wt)$color <- cols[V(ig_wt)$cluster+1]

# make interactive graph
g_js <- graphjs(g=ig_wt, 
                layout=layout_with_fr(ig_wt, weights=10*E(ig_wt)$width, dim=3),
                vertex.size = 0.2, 
                vertex.shape = names(V(ig_wt)),
                vertex.frame.color=NA,
                vertex.frame.width=0,
                vertex.label=names(V(ig_wt)),
                showLabels=T,
                edge.alpha=0.6,
                bg="grey",
                main=paste0("rho=", rho))

# save graph
graph_filename <- "path_to_your_research_outputs/cluster-vis.html"
saveWidget(g_js, file=graph_filename)

# open in browser
browseURL(graph_filename)

