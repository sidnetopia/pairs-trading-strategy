We have amassed evidence that the following is correlated with pairs trading performance:

past pairs trading backtest performance
similarity in PCA-derived economic exposures
conditional dependencies in returns (using the sparse covariance matrix graphical lasso)
both pairs being in the same sector and industry
similarity in a number of fundamental financial measures (including market cap, enterprise value and financial performance measures.)
That last point needs a bit of extra work, before we try to include it in our model.

Well, let’s think about what we were doing in that last lesson. We were:

sorting pairs based on current financial data
assessing pairs profitability based on past data, looking back in time.
So, we were using future knowledge to try to predict the past. 

Really, we need to validate against data that we could have only known at the time. So we’re gonna need some new data. We’re gonna need some historic snapshots of fundamental data.

Why didn’t we just do that to start with?

Because speed is important. And because it is easier to disprove something than it is to prove something.

We had snapshot fundamental data readily available, so it was quick and easy to do the analysis on that data. If the analysis hadn’t looked interesting, we’d have left it there and moved on. We wouldn’t have bothered to do the work to get a new data set.

Assignment – Conditional Factor Analysis
Our work so far has concentrated on these features’ ability to sort the entire population of pairs by their likely profitability.

In a pairs trading workflow, we are not looking to sort the entire group of millions of pairs – we will concentrate on those which have made money in the past, or those that show convergence/divergence behaviour in their price processes.

With this in mind, can you investigate the effectiveness of these similarity features on sorting the profitability of the 10% of pairs that were most profitable in a previous 2-3 year period?