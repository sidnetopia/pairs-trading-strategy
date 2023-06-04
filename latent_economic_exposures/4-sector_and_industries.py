snapshot_df %>%
  filter(sector != 'NA') %>%
  group_by(sector) %>%
  summarise(count = n()) %>%
  arrange(count) %>%
  ggplot(aes(x = reorder(sector, count), y = count)) +
  geom_bar(stat = 'identity') +
  xlab('Sector') + ylab('count') + ggtitle('Count of Stocks per Sector') +
  coord_flip()

We have more Technology stocks than any other sector.
We have less Communication Services stocks than any other sector.
Lower down the hierarchy, we have Industry. Let’s visualise how that works…

for (thesector in unique(snapshot_df$sector)) {
  (snapshot_df %>%
    filter(sector == thesector) %>%
    group_by(sector, industry) %>%
    summarise(count = n()) %>%
    arrange(count) %>%
    ggplot(aes(x = reorder(industry, count), y = count)) +
    geom_bar(stat = 'identity') +
    xlab('Industry') + ylab('count') + ggtitle(paste(thesector, 'sector')) +
    coord_flip()) %>%
    print()
}

Sector Analysis
Full Sample Univariate Factor Analysis
Let’s see if pairs of stocks from the same sector are, on average, more profitable than pairs of stocks from different sectors.

split all backtest results into two groups depending on whether the stocks are in the same or a different sector
plot the mean return of each of the groups.
lsr_df %>%
  na.omit() %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(samesector = case_when(sector.x == sector.y ~ 'same', TRUE ~ 'different')) %>%
  group_by(samesector) %>%
  summarise(meanmonthlyreturn = mean(lb20mean, na.rm = T)) %>%
  ggplot(aes(x = samesector, y = meanmonthlyreturn)) + geom_bar(stat='identity')

    take all our backtest returns in the lsr_dfdata frame
    exclude any rows for which we have NAs (due to limited data availability)
    lookup fundamental data from snapshot_dffor stock1 and stock2
    create a variable samesector which is TRUE when the sector for stock1 is the same as the sector for stock 2
    group by samesector and plot mean backtest returns.

    It’s a very simple chart, but it shows the effect we are interested in…

    Backtest returns are, on average, positive when both stocks are in the same sector.
    Backtest returns are, on average, negative when the two stocks are in different sectors.
    But that y-axis doesn’t make it look like a very big effect!

    lsr_df %>%
    na.omit() %>%
    inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
    inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
    mutate(samesector = case_when(sector.x == sector.y ~ 'same', TRUE ~ 'different')) %>%
    group_by(samesector) %>%
    summarise(meanannualreturn = mean(lb20mean, na.rm = T) * 1200)

    Now, allow me to go on a little aside about what we do when we get a new piece of data we think might be useful in a predictive model.

First, we do The Simplest Thing That Could Possibly Work, which is usually a full-sample sort like we’ve just done. We take all our data, divide it into groups, and look at the mean return of each group.

Then, if there’s something there, we dig a bit further. The next step is to move from summary means, to looking at the distribution of returns in each group to check that the results aren’t dominated by outliers.

Then, if there’s still something there, we’re going to want to look at time series plots. We want to plot how stable the effect is over time.

Then, if there’s still something there, we’ll poke a bit harder to try to understand it… What we do here will be dependent on the data we’re looking at – but generally we’re interesting in understanding at what point we can “break” the effect, to the point that the noise is overwhelming…

Sorted Return Distributions
Step number two is to move away from just looking at mean returns and instead plot a histogram of returns for our “same sector” and “different sector” groups.
lsr_df %>%
  na.omit() %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(samesector = case_when(sector.x == sector.y ~ 'same', TRUE ~ 'different')) %>%
  ggplot(aes(x=lb20mean, color=samesector)) +
  geom_histogram(bins=100, alpha = 0.5, position = 'identity')

Let’s define an outlier (for now) as an annual observation with an average monthly return of over 50%
outlier <- lsr_df %>%
  na.omit() %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(samesector = case_when(sector.x == sector.y ~ 'same', TRUE ~ 'different')) %>%
  filter(abs(lb20mean) > 0.5)
  head(outlier)

When we examine them, the outlier pair observations are concentrated in pairs containing a few jumpy stocks…
outlier %>% group_by(stock1) %>% summarise(countone = n()) %>% arrange(-countone)

outlier %>% group_by(stock2) %>% summarise(counttwo = n()) %>% arrange(-counttwo)

First, let’s look at how the outliers impact each group:
outlier %>%
  group_by(samesector) %>%
  summarise(
    count = n(),
    meanreturn = mean(lb20mean)
  )


We have more outliers in the “different” group (not a surprise)
The mean return of the “same” group is more negative than the “different” group.
This means that the outliers have the impact of lessening the effect we are interested in.

In future analysis, we’re going to filter these outliers out of the analysis.

Let’s plot the histogram without these outliers:
lsr_df %>%
  na.omit() %>% 
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(samesector = case_when(sector.x == sector.y ~ 'same', TRUE ~ 'different')) %>%
  ggplot(aes(x=lb20mean, color=samesector)) +
  geom_histogram(bins=100, alpha = 0.5, position = 'identity'

What do we see?

There are fewer pairs in the same sector (obviously)
The “same” distribution is slightly shifted to the right.
The histogram doesn’t make it that easy to compare the distributions, so let’s plot a density plot instead.
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(samesector = case_when(sector.x == sector.y ~ 'same', TRUE ~ 'different')) %>%
  ggplot(aes(x=lb20mean, color=samesector)) +
  geom_density(alpha = 0.5, position = 'identity')

  We see a higher peaked distribution for the “same” category, with more slightly positive observations, and fewer observations on the downside.

I don’t really see that…

Ok, I’ll “zoom in” and highlight what I am looking at…
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(samesector = case_when(sector.x == sector.y ~ 'same', TRUE ~ 'different')) %>%
  ggplot(aes(x=lb20mean, color=samesector)) +
  geom_density(alpha = 0.5, position = 'identity') +
  xlim(-0.1, 0.1)


Ok, so we’ve looked at distributions and seen that our effect isn’t caused by outliers. What’s next?

Stability of the effect over time
Step 3 is to look at the stability of the effect over time. Ideally, we’d like to see a fairly persistent outperformance of the “same” bucket vs the “different” bucket.

We’re going to plot the difference between the “same” group and the “different” group for each calendar year.

(We do this for nearly everything we look at… we always want to know how stable an effect is across time.)

same <- lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(samesector = case_when(sector.x == sector.y ~ 'same', TRUE ~ 'different')) %>%
  filter(samesector == 'same') %>%
  group_by(startofyear, samesector) %>%
  summarise(samereturn = mean(lb20mean, na.rm = T))

different <- lsr_df %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(samesector = case_when(sector.x == sector.y ~ 'same', TRUE ~ 'different')) %>%
  filter(samesector == 'different') %>%
  group_by(startofyear, samesector) %>%
  summarise(differentreturn = mean(lb20mean, na.rm = T))

same %>%
  inner_join(different, by = 'startofyear') %>%
  mutate(spread = (samereturn - differentreturn) * 1200) %>%
  ggplot(aes(x = startofyear, y = spread)) +
    geom_line() +
    geom_hline(yintercept = 0) +
    ggtitle('Annual spread (%) between same and different sector')


Before we move on, let’s plot a few performance measures for each sector to see if any patterns might emerge.

I want to look at backtest performance for all of the “same” sector combinations…
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  filter(!is.na(sector.x), !is.na(sector.y)) %>%
  filter(sector.x == sector.y) %>%
  group_by(sector.x, sector.y) %>%
  summarise(count = n(), meanannualreturn = mean(lb20mean) * 1200) %>%
  ggplot(aes(x = reorder(sector.x, meanannualreturn), y = meanannualreturn)) +
    geom_bar(stat = 'identity') +
    xlab('Sector') + ylab('Mean Annual Backtest Returns') + ggtitle('Same Sector Performance by Sector') +

In the code below, we plot the performance of “same sector” pairs, by Sector for each calendar year.
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  filter(!is.na(sector.x), !is.na(sector.y)) %>%
  filter(sector.x == sector.y) %>%
  group_by(sector.x, sector.y, startofyear) %>%
  summarise(count = n(), meanannualreturn = mean(lb20mean) * 1200) %>%
  ggplot(aes(x = reorder(sector.x, meanannualreturn), y = meanannualreturn)) +
    geom_bar(stat = 'identity') +
    xlab('Sector') + ylab('count') + ggtitle('Same Sector Performance by Sector') +
    coord_flip() +
    facet_wrap(~startofyear)

There isn’t any obvious consistency there. The same sectors don’t seem to persistently out or under-perform.

But even though we have a ton of data, it’s still a ton of noisy data, and a year is only 252 trading days.

So let’s split the time period in half and plot the first half (2014-2016) and the second half (2017-2019).
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  filter(!is.na(sector.x), !is.na(sector.y)) %>%
  filter(sector.x == sector.y) %>%
  mutate(half = case_when(startofyear <= '2016-01-01' ~ '2014-2016', TRUE ~ '2017-2019')) %>%
  group_by(sector.x, sector.y, half) %>%
  summarise(count = n(), meanannualreturn = mean(lb20mean) * 1200) %>%
  ggplot(aes(x = reorder(sector.x, meanannualreturn), y = meanannualreturn)) +
    geom_bar(stat = 'identity') +
    xlab('Sector') + ylab('Mean Annual Returns') + ggtitle('Same Sector Performance by Sector') +
    coord_flip() +
    facet_wrap(~half)

combo <- lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  filter(!is.na(sector.x), !is.na(sector.y)) %>%
  group_by(sector.x, sector.y) %>%
  summarise(count = n(), meanannualreturn = mean(lb20mean) * 1200)
# Now we can have the sectors both ways around, so let's join the table to itself 
x <- combo %>%
  filter(sector.x != sector.y) %>%
  full_join(filter(combo, sector.x != sector.y), by = c('sector.x' = 'sector.y', 'sector.y' = 'sector.x')) %>%
  mutate(comboreturn = ((meanannualreturn.x * count.x)  + (meanannualreturn.y * count.y)) / (count.x + count.y)) %>% 
  arrange(-comboreturn) %>% 
  select(sector.x, sector.y, comboreturn) %>% 
  ungroup() 
x <- x %>% slice(seq(1,nrow(x),2))
y <- combo %>%
  filter(sector.x == sector.y) %>%
  select(sector.x, sector.y, comboreturn = meanannualreturn)
bind_rows(x, y) %>%
  arrange(-comboreturn)

Industry Analysis
Next, let’s look at the level down, which is Industry.

We’ll go through the same procedure:

Group by same and different industry and plot mean returns
Plot the return distribution histogram and density plot for each group
Look at stability of any effect with respect to time
Dig deeper, if appropriate.
Let’s plot mean returns as a bar chart…
lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(sameindustry = case_when(industry.x == industry.y ~ 'same', TRUE ~ 'different')) %>%
  group_by(sameindustry) %>%
  summarise(meanmonthlyreturn = mean(lb20mean, na.rm = T)) %>%
  ggplot(aes(x = sameindustry, y = meanmonthlyreturn)) + geom_bar(stat='identity')  

lsr_df %>%
  na.omit() %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(samesector = case_when(industry.x == industry.y ~ 'same', TRUE ~ 'different')) %>%
  group_by(samesector) %>%
  summarise(meanannualreturn = mean(lb20mean, na.rm = T) * 1200)

lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(sameindustry= case_when(industry.x == industry.y ~ 'same', TRUE ~ 'different')) %>%
  ggplot(aes(x=lb20mean, color=sameindustry)) +
  geom_histogram(bins=100, alpha = 0.5, position = 'identity')

lsr_df %>%
  na.omit() %>%
  filter(abs(lb20mean) < 0.5) %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(sameindustry= case_when(industry.x == industry.y ~ 'same', TRUE ~ 'different')) %>%
  ggplot(aes(x=lb20mean, color=sameindustry)) +
  geom_density(alpha = 0.5, position = 'identity')

same <- lsr_df %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(sameindustry = case_when(industry.x == industry.y ~ 'same', TRUE ~ 'different')) %>%
  filter(sameindustry == 'same') %>%
  group_by(startofyear, sameindustry) %>%
  summarise(samereturn = mean(lb20mean, na.rm = T))
different <- lsr_df %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(sameindustry = case_when(industry.x == industry.y ~ 'same', TRUE ~ 'different')) %>%
  filter(sameindustry == 'different') %>%
  group_by(startofyear, sameindustry) %>%
  summarise(differentreturn = mean(lb20mean, na.rm = T))
same %>%
  inner_join(different, by = 'startofyear') %>%
  mutate(spread = (samereturn - differentreturn) * 1200) %>%
  ggplot(aes(x = startofyear, y = spread)) +
    geom_line() +
    geom_hline(yintercept = 0) +
    ggtitle('Annual spread (%) between same and different industry')

same <- lsr_df %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(sameindustry = case_when(industry.x == industry.y ~ 'same', TRUE ~ 'different')) %>%
  filter(sameindustry == 'same') %>%
  group_by(sector.x, startofyear, sameindustry) %>%
  summarise(samereturn = mean(lb20mean, na.rm = T))

different <- lsr_df %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(sameindustry = case_when(industry.x == industry.y ~ 'same', TRUE ~ 'different')) %>%
  filter(sameindustry == 'different') %>%
  group_by(sector.x, startofyear, sameindustry) %>%
  summarise(differentreturn = mean(lb20mean, na.rm = T))

same %>%
  inner_join(different, by = c('sector.x','startofyear')) %>%
  mutate(spread = (samereturn - differentreturn) * 1200) %>%
  ggplot(aes(x = startofyear, y = spread)) +
    geom_line() +
    geom_hline(yintercept = 0) +
    ggtitle('Annual spread (%) between same and different industry') +
    facet_wrap(~sector.x)


Let’s look to see if we can get any insight into the question: under what conditions it might be useful to take the industry into account?

So let’s plot the above, for each sector individually…
same <- lsr_df %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(sameindustry = case_when(industry.x == industry.y ~ 'same', TRUE ~ 'different')) %>%
  filter(sameindustry == 'same') %>%
  group_by(sector.x, startofyear, sameindustry) %>%
  summarise(samereturn = mean(lb20mean, na.rm = T))
different <- lsr_df %>%
  inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
  inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
  mutate(sameindustry = case_when(industry.x == industry.y ~ 'same', TRUE ~ 'different')) %>%
  filter(sameindustry == 'different') %>%
  group_by(sector.x, startofyear, sameindustry) %>%
  summarise(differentreturn = mean(lb20mean, na.rm = T))
same %>%
  inner_join(different, by = c('sector.x','startofyear')) %>%
  mutate(spread = (samereturn - differentreturn) * 1200) %>%
  ggplot(aes(x = startofyear, y = spread)) +
    geom_line() +
    geom_hline(yintercept = 0) +
    ggtitle('Annual spread (%) between same and different industry') +
    facet_wrap(~sector.x)

Again, it’s a struggle to read too much into that, other than there’s a noisy positive spread for most sectors.

Finally, let’s plot same-industry performance for each of the industries individually…
for (s in na.omit(unique(snapshot_df$sector))) {
  (lsr_df %>%
    na.omit() %>%
    filter(abs(lb20mean) < 0.5) %>%
    inner_join(snapshot_df, by = c('stock1' = 'ticker')) %>%
    inner_join(snapshot_df, by = c('stock2' = 'ticker')) %>%
    filter(!is.na(sector.x), !is.na(sector.y), !is.na(industry.x), !is.na(industry.y)) %>%
    filter(sector.x == s) %>%
    filter(industry.x == industry.y) %>%
    mutate(half = case_when(startofyear <= '2016-01-01' ~ '2014-2016', TRUE ~ '2017-2019')) %>%
    group_by(industry.x, half) %>%
    summarise(count = n(), meanannualreturn = mean(lb20mean) * 1200) %>%
    ggplot(aes(x = reorder(industry.x, meanannualreturn), y = meanannualreturn)) +
      geom_bar(stat = 'identity') +
      xlab('Industry') + ylab('Mean Annual Returns') + ggtitle(paste('Same Industry Performance for', s)) +
      coord_flip() +
      facet_wrap(~half)) %>% 
  print()
}


What do you think? Do you think there’s any signal in the madness there?

Certainly, it appears that same sector / same industry is a useful filter… but is there anything else that might be justified given what we’ve observed?

 