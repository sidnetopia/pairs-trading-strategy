We can also look at similarity in financial features, such as market cap, revenue, or price to earnings ratios. These are continuous variables. Or, as I like to call them, numbers!

In this lesson, we’re going to do some basic univariate factor analysis to determine whether similarity in these financial features might be predictive of pairs trading returns.

In our modelling, we only want to include features that we are reasonably confident has some predictive power to identify profitable pairs.

So if we don’t see some evidence of weak predictive power in a single feature by itself, we are unlikely to try to include it on our model.

The procedure for this factor analysis is similar to the work we’ve done before:

Sort pairs into deciles by the difference in the chosen feature (from snapshot_df)
Plot the mean returns of those deciles.
The way we calculate the differences depends on the units of the feature:

For $ features, we use the log difference.
For % features, we just take the difference.    

Market Capitalisation
Do pairs with similar market capitalisations outperform those with very different market capitalisations?

First, we want to know what the distribution of the market cap looks like throughout our sample. A good way to do that is to plot a histogram.
snapshot_df %>%
  ggplot(aes(x=marketcap)) +
  geom_histogram(bins=100, position = 'identity')

That’s a very skewed distribution. When we see something like this, we are going to want to try a log transformation of it to make it a bit “better behaved”.
snapshot_df %>%
  ggplot(aes(x=log(marketcap))) +
  geom_histogram(bins=100, position = 'identity')


Now we’re going to do the factor analysis we:

join our backtest results lsr_df to the fundamental data snapshot snapshot_df for each stock
calculate a difference feature marketcapdiff which is the difference of the log of each market cap
sort the data into 10 deciles by increasing marketcapdiff
group by each decile and plot the mean of the annualised backtest returns.
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(marketcapdiff = abs(log(marketcap.x) - log(marketcap.y))) %>%
  mutate(quantile = ntile(marketcapdiff, 10)) %>%
  filter(!is.na(quantile)) %>%
  group_by(quantile) %>%
  summarise(meanreturns = mean(lb20mean, na.rm = T)) %>%
  ggplot(aes(x = quantile, y = meanreturns * 1200)) + geom_bar(stat='identity') +
  ggtitle('Annual Backtest Returns (%) sorted on log difference in market cap') + 
  ylab('Annual Backtest Returns (%)') + xlab('Log Difference in Market Cap Decile')


We see a fairly clear effect, and a wide spread between the top and bottom deciles.

There is also a suggestion that this sort is more effective at identifying what we shouldn’t be trading. This could be very useful as a filter if this feature is somewhat uncorrelated with other features we are going to use.

Enterprise Value
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(evdiff = log(enterprisevalue.x) - log(enterprisevalue.y)) %>%
  mutate(quantile = ntile(evdiff, 10)) %>%
  filter(!is.na(quantile)) %>%
  group_by(quantile) %>%
  summarise(meanreturns = mean(lb20mean, na.rm = T)) %>%
  ggplot(aes(x = quantile, y = meanreturns * 1200)) + geom_bar(stat='identity') +
  ggtitle('Annual Backtest Returns (%) sorted on log difference in enterprise value') + 
  ylab('Annual Backtest Returns (%)') + xlab('Log Difference in ev decile')

Very similar to market cap. “Shape” is slightly better behaved.

Revenue
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(revdiff = log(revenue.x) - log(revenue.y)) %>%
  mutate(quantile = ntile(revdiff, 10)) %>%
  filter(!is.na(quantile)) %>%
  group_by(quantile) %>%
  summarise(meanreturns = mean(lb20mean, na.rm = T)) %>%
  ggplot(aes(x = quantile, y = meanreturns * 1200)) + geom_bar(stat='identity') +
  ggtitle('Annual Backtest Returns (%) sorted on log difference in revenue') + 
  ylab('Annual Backtest Returns (%)') + xlab('Log difference in revenue decile')

A beautiful shape, but the spread between decile 1 and decile 10 is relatively small.

Net Income
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(revdiff = netincome.x - netincome.y) %>%
  mutate(quantile = ntile(revdiff, 10)) %>%
  filter(!is.na(quantile)) %>%
  group_by(quantile) %>%
  summarise(meanreturns = mean(lb20mean, na.rm = T)) %>%
  ggplot(aes(x = quantile, y = meanreturns * 1200)) + geom_bar(stat='identity') +
  ggtitle('Annual Backtest Returns (%) sorted on $ difference in net income') + 
  ylab('Annual Backtest Returns (%)') + xlab('Difference in net income decile')

Seems to be more predictive of bad pairs trading returns.



Price to Earnings Ratio
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(dff = pe.x.x - pe.x.y) %>%
  mutate(quantile = ntile(dff, 10)) %>%
  filter(!is.na(quantile)) %>%
  group_by(quantile) %>%
  summarise(meanreturns = mean(lb20mean, na.rm = T)) %>%
  ggplot(aes(x = quantile, y = meanreturns * 1200)) + geom_bar(stat='identity') +
  ggtitle('Annual Backtest Returns (%) sorted on $ difference in PE ratio') + 
  ylab('Annual Backtest Returns (%)') + xlab('Difference in PE ratio decile')

Seems to be more predictive of bad pairs trading returns.

Price to Sales Ratio
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(dff = ps.x.x - ps.x.y) %>%
  mutate(quantile = ntile(dff, 10)) %>%
  filter(!is.na(quantile)) %>%
  group_by(quantile) %>%
  summarise(meanreturns = mean(lb20mean, na.rm = T)) %>%
  ggplot(aes(x = quantile, y = meanreturns * 1200)) + geom_bar(stat='identity') +
  ggtitle('Annual Backtest Returns (%) sorted on $ difference in PS ratio') + 
  ylab('Annual Backtest Returns (%)') + xlab('Difference in PS ratio decile')

Similar overall pattern. Again, the first decile is very interesting. Stocks with very similar Price to Sales ratios seem to do poorly, on average.

Price to Book Ratio
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(dff = pb.x.x - pb.x.y) %>%
  mutate(quantile = ntile(dff, 10)) %>%
  filter(!is.na(quantile)) %>%
  group_by(quantile) %>%
  summarise(meanreturns = mean(lb20mean, na.rm = T)) %>%
  ggplot(aes(x = quantile, y = meanreturns * 1200)) + geom_bar(stat='identity') +
  ggtitle('Annual Backtest Returns (%) sorted on $ difference in PB ratio') + 
  ylab('Annual Backtest Returns (%)') + xlab('Difference in PB ratio decile')
Similar overall pattern. Again, the first decile is very interesting. Stocks with very similar Price to Book ratios seem to do poorly, on average.

Return on Assets
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(dff = roa.x.x - roa.x.y) %>%
  mutate(quantile = ntile(dff, 10)) %>%
  filter(!is.na(quantile)) %>%
  group_by(quantile) %>%
  summarise(meanreturns = mean(lb20mean, na.rm = T)) %>%
  ggplot(aes(x = quantile, y = meanreturns * 1200)) + geom_bar(stat='identity') +
  ggtitle('Annual Backtest Returns (%) sorted on $ difference in RoA ratio') + 
  ylab('Annual Backtest Returns (%)') + xlab('Difference in RoA ratio decile')
Similar to previous examplers.

Number of Shareholders
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(dff = no_shrholders.x - no_shrholders.y) %>%
  mutate(quantile = ntile(dff, 10)) %>%
  filter(!is.na(quantile)) %>%
  group_by(quantile) %>%
  summarise(meanreturns = mean(lb20mean, na.rm = T)) %>%
  ggplot(aes(x = quantile, y = meanreturns * 1200)) + geom_bar(stat='identity') +
  ggtitle('Annual Backtest Returns (%) sorted on difference in no of shareholders') + 
  ylab('Annual Backtest Returns (%)') + xlab('Difference in no of shareholders')

Institutional Ownership Percentage
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(dff = instutionalpct.x - instutionalpct.y) %>%
  mutate(quantile = ntile(dff, 10)) %>%
  filter(!is.na(quantile)) %>%
  group_by(quantile) %>%
  summarise(meanreturns = mean(lb20mean, na.rm = T)) %>%
  ggplot(aes(x = quantile, y = meanreturns * 1200)) + geom_bar(stat='identity') +
  ggtitle('Annual Backtest Returns (%) sorted on difference in insitutional ownership %') + 
  ylab('Annual Backtest Returns (%)') + xlab('Difference in institutional ownership %')

Institutional Put/Call Ratio
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(dff = (puts.x / calls.x) - (puts.y / calls.y)) %>%
  mutate(quantile = ntile(dff, 10)) %>%
  filter(!is.na(quantile)) %>%
  group_by(quantile) %>%
  summarise(meanreturns = mean(lb20mean, na.rm = T)) %>%
  ggplot(aes(x = quantile, y = meanreturns * 1200)) + geom_bar(stat='identity') +
  ggtitle('Annual Backtest Returns (%) sorted on difference in insitutional put/call ratio') + 
  ylab('Annual Backtest Returns (%)') + xlab('Difference in institutional put/call ratio')

There is a reasonable amount of evidence here that similarity of financial metrics has some ability to predict broad pairs trading returns.

It is likely that this information does not align perfectly with the previous work we did on returns data – and, as such, will be a useful input into our pairs classification model.

It appeared that pairs consisting of stocks with very similar financial ratios did not perform as well as pairs consisting of stocks with similar (but not too similar) financial ratios. This deserves more investigation.

Assignment
Factor Analysis on More Features
Perform similar analysis on the following features:

Return on Equity
Debt to Equity
Exchange
Location


