# Calculate returns
ret <- prices_df %>%
  group_by(ticker) %>%
  tq_transmute(select=close, mutate_fun=periodReturn, period="daily")
# make wide dataframe of returns
ret_wide <- ret %>%
  spread(key=ticker, value=daily.returns) %>% 
  select(-date)

# make inverse covariance matrices for each subset for various rho. save for later
start_dates <- c("2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01")
end_dates <- c("2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01")
i <- 1
for(rho in c(0.3, 0.5, 0.7)) {
  for(i in 1:length(start_dates)) {
    return_subset <- subset_returns(start_dates[i], end_dates[i], ret, max_nans=125, fill_mean=FALSE)
    S <- return_subset %>%
      scale(center=TRUE, scale=TRUE) %>%
      cov(use='p')
    
    invcov <- glasso(S, rho=rho)  # takes (potentially much) longer to fit on full data set
    P <- invcov$wi
    colnames(P) <- colnames(S)
    rownames(P) <- colnames(P)
    saveRDS(P, file=paste0("C:/Users/Kris/Documents/rw-ml-bootcamp/unsupervised-learning/", start_dates[i], "-", end_dates[i], "_rho_", rho, ".rds"))
  }
}

### Are stronger connections associated with pairs trading profitability?
start_dates <- c("2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01")
end_dates <- c("2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01")

N <- length(start_dates)*2
results <- data.frame(start=rep("", N), 
                      end=rep("", N), 
                      in_sample = rep("", N),
                      num_nonzeros=rep(NA, N), 
                      num_zeros=rep(NA, N),
                      mean_ret20_zero = rep(NA, N),
                      mean_ret60_zero = rep(NA, N),
                      mean_ret20_nonzero = rep(NA, N),
                      mean_ret60_nonzero = rep(NA, N),
                      stringsAsFactors=FALSE)

for(i in seq(1, N/2, 1)) {
  P <- readRDS(filenames[i])
  
  # bucket by connection strength
  conn <- abs(P) %>%   # note taking absolute values
    tbl_df() %>%
    mutate(stock1 = rownames(P)) %>%
    pivot_longer(-stock1, names_to = "stock2", values_to = "par.corr") %>%
    filter(stock1 != stock2)
  
  # how many non-zero connections?
  num_nonzeros <- conn %>%
    filter(par.corr != 0) %>%
    summarise(num_nonzeros = n())
  
  num_zeros <- conn %>%
    filter(par.corr == 0) %>%
    summarise(num_zeros = n())
  
  # in-sample performance
  is_perf <- lsr_df %>%
    filter(startofyear >= as.Date(start_dates[i]) & startofyear < as.Date(end_dates[i])) %>%
    select(stock1, stock2, startofyear, lb20mean, lb60mean) %>%
    left_join(conn, by=c("stock1"="stock1", "stock2" = "stock2"))
  
  mean_ret_zero <- is_perf %>%
    filter(par.corr == 0) %>%
    summarise(mean_ret20_zero = mean(lb20mean, na.rm=T), mean_ret60_zero = mean(lb60mean, na.rm=T))
  
  mean_ret_nonzero <- is_perf %>%
    filter(par.corr != 0) %>%
    summarise(mean_ret20_nonzero = mean(lb20mean, na.rm=T), mean_ret60_nonzero = mean(lb60mean, na.rm=T))
  
  results[(2*i-1), ] <- c(start_dates[i], end_dates[i], "in-sample", num_nonzeros, num_zeros, 
                    mean_ret_zero, mean_ret_nonzero)
  
  # out-of-sample performance
  oos_perf <- lsr_df %>%
    filter(startofyear >= as.Date(start_dates[i+1]) & startofyear < as.Date(end_dates[i+1])) %>%
    select(stock1, stock2, startofyear, lb20mean, lb60mean) %>%
    left_join(conn, by=c("stock1"="stock1", "stock2" = "stock2"))
  
  mean_ret_zero_oos <- oos_perf %>%
    filter(par.corr == 0) %>%
    summarise(mean_ret20 = mean(lb20mean, na.rm=T), mean_ret60 = mean(lb60mean, na.rm=T))
  
  mean_ret_nonzero_oos <- oos_perf %>%
    filter(par.corr != 0) %>%
    summarise(mean_ret20 = mean(lb20mean, na.rm=T), mean_ret60 = mean(lb60mean, na.rm=T))
  
  results[2*i, ] <- c(start_dates[i], end_dates[i], "out-of-sample", num_nonzeros, num_zeros, 
                      mean_ret_zero_oos, mean_ret_nonzero_oos)
  
  print(is_perf %>% 
    filter(par.corr != 0) %>%
    mutate(quantile = ntile(par.corr, 5)) %>%
    group_by(quantile) %>%
    summarise(mean_ret20 = mean(lb20mean, na.rm=T)) %>%
    ggplot(aes(x=quantile, y=mean_ret20)) +
      geom_col() +
      ggtitle(paste0("In-sample factor plot, non-zero partial correlation ", start_dates[i])))
  
  print(oos_perf %>%
    filter(par.corr != 0) %>%
    mutate(quantile = ntile(par.corr, 5)) %>%
    group_by(quantile) %>%
    summarise(mean_ret20 = mean(lb20mean, na.rm=T)) %>%
    ggplot(aes(x=quantile, y=mean_ret20)) +
      geom_col() +
      ggtitle(paste0("Out-of-sample factor plot, non-zero partial correlation ", start_dates[i])))
} 

We loop through each subset of our data, read in the relevant inverse covariance matrix, and calculate the following metrics and store them in the results dataframe:

Number of conditionally independent pairs (zero in the inverse covariance matrix)
Number of conditionally dependent pairs (non-zero in the inverse covariance matrix)
The in-sample mean return of conditionally independent pairs
The in-sample mean return of conditionally dependent pairs
The out-of-sample mean return of conditionally independent pairs
The out-of-sample mean return of conditionally dependent pairs

Finally, we plot a time series of the returns to pairs connected by the graphical lasso versus pairs identified as being conditionally independent. We plot this for the in-sample and out-sample subsets:
# zeros vs nonzeros
results %>% 
  tbl_df() %>%
  # filter(in_sample == T) %>%
  select(start, mean_ret20_zero, mean_ret20_nonzero, in_sample) %>%
  pivot_longer(-c(start, in_sample), names_to = 'partial.corr', values_to = 'mean_ret') %>%
  ggplot(aes(x=as.Date(start), y=mean_ret)) +
  geom_line(aes(colour=partial.corr)) +
  facet_wrap(~in_sample) +
  labs(x = "In-sample start date", y = "Mean monthly return")


If you repeat the process for ρ=0.3,0.7 you’ll find similar results although the overall outperformance is a bit more marginal. Perhaps that’s suggestive of a sweet-spot for ρ, but I’m not reading too much into that for now.

Finally, for comparison, we make a factor plot of the in-sample and out-of-sample returns by magnitude of the Pearson correlation coefficient (we take only pairs whose in-sample correlation was greater than 0.6, a somewhat arbitrary but not unreasonable filter):
# more useful than sorting on pearson correlation?
for(i in seq(1, N/2, 1)) {
  
  return_subset <- subset_returns(start_dates[i], end_dates[i], ret, max_nans=125, fill_mean=FALSE)
  correl <- return_subset %>%
    cor(use='p')
  
  correl <- abs(correl) %>%   # note taking absolute values
    tbl_df() %>%
    mutate(stock1 = rownames(correl)) %>%
    pivot_longer(-stock1, names_to = "stock2", values_to = "corr") %>%
    filter(stock1 != stock2) %>%
    filter(corr > 0.6)
  
  # in-sample performance
  is <- lsr_df %>%
    filter(startofyear >= as.Date(start_dates[i]) & startofyear < as.Date(end_dates[i])) %>%
    select(stock1, stock2, startofyear, lb20mean, lb60mean) %>%
    left_join(correl, by=c("stock1"="stock1", "stock2" = "stock2")) 
  
  is_perf <- is %>%
    summarise(mean_ret20 = mean(lb20mean, na.rm=T), mean_ret60 = mean(lb60mean, na.rm=T))
  
  # out-of-sample performance
  oos <- lsr_df %>%
    filter(startofyear >= as.Date(start_dates[i+1]) & startofyear < as.Date(end_dates[i+1])) %>%
    select(stock1, stock2, startofyear, lb20mean, lb60mean) %>%
    left_join(correl, by=c("stock1"="stock1", "stock2" = "stock2"))
  
  oos_perf <- oos %>%
    summarise(mean_ret20 = mean(lb20mean, na.rm=T), mean_ret60 = mean(lb60mean, na.rm=T))
  
  print(is %>% 
          mutate(quantile = ntile(corr, 5)) %>%
          group_by(quantile) %>%
          summarise(mean_ret20 = mean(lb20mean, na.rm=T)) %>%
          ggplot(aes(x=quantile, y=mean_ret20)) +
          geom_col() +
          ggtitle(paste0("In-sample correlation factor plot ", start_dates[i])))
  
  print(oos %>% 
          mutate(quantile = ntile(corr, 5)) %>%
          group_by(quantile) %>%
          summarise(mean_ret20 = mean(lb20mean, na.rm=T)) %>%
          ggplot(aes(x=quantile, y=mean_ret20)) +
          geom_col() +
          ggtitle(paste0("Out-of-sample correlation factor plot ", start_dates[i])))
}


What does this all mean in practical terms?
It means that we’ve added another weak predictor of pairs trading profitability to our arsenal. The fact that it outperforms the naive Pearson correlation approach – which looks a lot like random noise – is further indication that we’ve done something useful and that our model is adding value.

look more directly at fundamental data to assess fundamental similarity.

The sector the company operates in
The industry it operates in
The location of the company
The accounting currency
The market capitalisation of the stock
The enterprise value of the company
The $ revenue
The $ net income
The % owned by institutional shareholders
Ratios such as:
price to earnings
price to sales
debt to equity
return on equity
return on assets.
