WITH bigtable AS (
SELECT
  s1.ticker as stock1,
  s2.ticker as stock2,
  ROW_NUMBER() OVER (PARTITION BY stock1, stock2 ORDER BY s1.date ASC) as rownumber,
  s1.date,
  s1.close as close1,
  s2.close as close2,
  SAFE_DIVIDE(s1.close, s2.close) as spreadclose,
  AVG(SAFE_DIVIDE(s1.close, s2.close)) OVER (PARTITION BY stock1, stock2 ORDER BY s1.date ASC ROWS 19 PRECEDING) as sma20,
  STDDEV(SAFE_DIVIDE(s1.close, s2.close)) OVER (PARTITION BY stock1, stock2 ORDER BY s1.date ASC ROWS 19 PRECEDING) as stdev20,
  AVG(SAFE_DIVIDE(s1.close, s2.close)) OVER (PARTITION BY stock1, stock2 ORDER BY s1.date ASC ROWS 59 PRECEDING) as sma60,
  STDDEV(SAFE_DIVIDE(s1.close, s2.close)) OVER (PARTITION BY stock1, stock2 ORDER BY s1.date ASC ROWS 59 PRECEDING) as stdev60,
  AVG(SAFE_DIVIDE(s1.close, s2.close)) OVER (PARTITION BY stock1, stock2 ORDER BY s1.date ASC ROWS 89 PRECEDING) as sma90,
  STDDEV(SAFE_DIVIDE(s1.close, s2.close)) OVER (PARTITION BY stock1, stock2 ORDER BY s1.date ASC ROWS 89 PRECEDING) as stdev90
FROM `rw-algotrader.mlb_prices.prices_df` s1
INNER JOIN `rw-algotrader.mlb_prices.prices_df` s2 ON s1.date = s2.date
INNER JOIN `rw-algotrader.mlb_prices.pairs` p ON p.stock1 = s1.ticker AND p.stock2 = s2.ticker
-- Due to the window functions (SMAs, StDevs) we have to process more data than we actually need
WHERE s1.date >=  DATE_SUB(processyear, INTERVAL 6 MONTH) and s1.date < DATE_ADD(processyear, INTERVAL 1 YEAR))
SELECT
  stock1,
  stock2,
  date,
  close1,
  close2,
  spreadclose,
  sma20,
  stdev20,
  sma60,
  stdev60,
  sma90,
  stdev90
 FROM bigtable
 WHERE date >= processyear AND rownumber >= 90 ) -- only return the stuff we want

 WITH yearlylsr AS SELECT
stocki, stock2, DATE_TRUNC startofmonth, YEAR) as startofyear, AVG (1620) as 1b20mean, STDDEV (1620) as 1b20stdev, AVG (1660) as lb60mean, STDDEV (1660) as 1b60stdev, AVG (1690) as 1b98mean,
STDDEV (1690) as 1b90stdev FROM `rw-algotrader.mlb_prices.linearstrategyreturns` WHERE startofmonth >= '2018-01-01' GROUP BY stocki, stock2, DATE_TRUNC (startofmonth, YEAR)
SELECT * FROM yearlylsr ORDER BY stocki, stock2, startofyear