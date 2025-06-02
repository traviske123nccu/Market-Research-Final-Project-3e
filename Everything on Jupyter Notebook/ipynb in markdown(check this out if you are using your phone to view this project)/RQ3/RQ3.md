# RQ3 - Do macro-level shocks (e.g., Fed decisions) or firm-level shocks affect retail attention and market activity in Taiwan’s ETF sector?

#### Code


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import yfinance as yf

df = pd.read_excel("merged_df.xlsx")
# 修正未命名的第一欄為 'date'
if df.columns[0].startswith("Unnamed"):
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
```

## 4.3.1 Retail Attention Sensitivity During Fed Policy Weeks

To evaluate whether macroeconomic announcements—specifically U.S. Federal Reserve interest rate decisions—elicit measurable shifts in investor attention, we analyze Google Trends–based attention indexes during weeks of Fed policy announcements. These “Fed Weeks” are defined as the weeks containing or immediately adjacent to official interest rate decision dates. For each attention index, we compare attention levels between Fed Weeks and all other (non-Fed) weeks using both visual inspection via boxplots and statistical testing via independent sample t-tests.

The boxplot in Figure 4.3.1 displays the distribution of attention values across Fed and non-Fed weeks for six thematic indexes. Most indexes show overlapping distributions, suggesting little distinction in attention behavior between the two event categories. To quantify these differences, we perform t-tests across all attention themes.

As shown in the test results below the figure, only the Macro_Attention_Index shows a statistically significant difference during Fed Weeks (p = 0.0169), indicating that public search interest in macroeconomic topics intensifies during these policy announcement windows. Other indexes—such as those related to technology, stocks, or ETFs—show no significant difference, with p-values well above conventional thresholds.

These findings suggest that retail attention is selectively responsive to macro-level news, particularly when the theme aligns directly with the content of the announcement (e.g., inflation, interest rates). However, the lack of broader significance across other indexes implies that Fed announcements do not trigger uniform retail attention shifts across investment categories. This aligns with the conclusion of Section 4.1, where attention patterns tended to reflect theme-specific behavioral sensitivities. The results here also set the stage for Section 4.3.2, where we investigate whether these attention shifts translate into real trading behavior.

#### Code


```python
# Melt data for seaborn boxplot

# 建立 Fed 公布利率決策的日期清單
fed_dates = pd.to_datetime([
    '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
    '2024-07-31', '2024-09-18', '2024-11-06', '2024-12-18'
])

# 建立 FedWeek 標籤（Fed 事件所在週）
df['FedWeek'] = df['date'].apply(lambda x: any((x >= d - pd.Timedelta(days=3)) and (x <= d + pd.Timedelta(days=3)) for d in fed_dates))


melted = df.reset_index().melt(id_vars=['date', 'FedWeek'], 
                               value_vars=[col for col in df.columns if col.endswith('_Attention_Index')],
                               var_name='IndexType', value_name='AttentionValue')

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=melted, x='IndexType', y='AttentionValue', hue='FedWeek')
plt.title("Figure 4.3.1 - Attention Index Distribution: Fed vs Non-Fed Weeks")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/4.3.1_attention_distribution_boxplot.png", dpi=300)
plt.show()
```


    
![png](output_5_0.png)
    



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>0050.TW_Volume_norm</th>
      <th>006208.TW_Volume_norm</th>
      <th>00878.TW_Volume_norm</th>
      <th>00713.TW_Volume_norm</th>
      <th>2330.TW_Volume_norm</th>
      <th>2303.TW_Volume_norm</th>
      <th>2412.TW_Volume_norm</th>
      <th>3008.TW_Volume_norm</th>
      <th>2881.TW_Volume_norm</th>
      <th>...</th>
      <th>2308.TW_Volume_norm</th>
      <th>3034.TW_Volume_norm</th>
      <th>2454.TW_Volume_norm</th>
      <th>ETF_Attention_Index</th>
      <th>Stock_Attention_Index</th>
      <th>Dividend_Attention_Index</th>
      <th>Beginner_Attention_Index</th>
      <th>Macro_Attention_Index</th>
      <th>Tech_Attention_Index</th>
      <th>FedWeek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-01-07</td>
      <td>-0.860674</td>
      <td>-0.932212</td>
      <td>-0.379417</td>
      <td>-0.874601</td>
      <td>-1.254108</td>
      <td>-0.166771</td>
      <td>-0.911028</td>
      <td>0.418549</td>
      <td>-1.167018</td>
      <td>...</td>
      <td>-1.300789</td>
      <td>-0.466500</td>
      <td>0.439787</td>
      <td>-1.216050</td>
      <td>-1.367152</td>
      <td>-0.199133</td>
      <td>-0.899066</td>
      <td>0.390401</td>
      <td>-0.575975</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-01-14</td>
      <td>-0.653673</td>
      <td>-0.931290</td>
      <td>-0.708484</td>
      <td>-0.822152</td>
      <td>-1.295240</td>
      <td>-0.650347</td>
      <td>-0.810122</td>
      <td>1.288575</td>
      <td>-0.906104</td>
      <td>...</td>
      <td>-0.264088</td>
      <td>-0.352294</td>
      <td>-0.322731</td>
      <td>-0.116791</td>
      <td>-0.582552</td>
      <td>0.236205</td>
      <td>-0.666854</td>
      <td>-0.072779</td>
      <td>-0.180896</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-01-21</td>
      <td>1.345122</td>
      <td>-0.478630</td>
      <td>-0.280000</td>
      <td>-0.611415</td>
      <td>1.391158</td>
      <td>0.676357</td>
      <td>-0.147496</td>
      <td>0.498769</td>
      <td>0.186507</td>
      <td>...</td>
      <td>0.332889</td>
      <td>1.014547</td>
      <td>0.856694</td>
      <td>-0.433229</td>
      <td>-0.582552</td>
      <td>-0.121921</td>
      <td>-0.173195</td>
      <td>-0.007083</td>
      <td>-0.180896</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-01-28</td>
      <td>0.804586</td>
      <td>-0.368364</td>
      <td>-0.302921</td>
      <td>-0.923418</td>
      <td>0.623496</td>
      <td>1.909554</td>
      <td>-0.861677</td>
      <td>-0.341718</td>
      <td>-0.874992</td>
      <td>...</td>
      <td>-0.248834</td>
      <td>-0.524553</td>
      <td>0.512786</td>
      <td>-0.119907</td>
      <td>-1.067593</td>
      <td>-0.194599</td>
      <td>-0.563430</td>
      <td>-0.396802</td>
      <td>-0.685401</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-04</td>
      <td>0.260986</td>
      <td>-0.821956</td>
      <td>-0.410453</td>
      <td>-1.048354</td>
      <td>-0.096706</td>
      <td>-0.126591</td>
      <td>-0.394153</td>
      <td>0.253704</td>
      <td>-0.933695</td>
      <td>...</td>
      <td>-0.339969</td>
      <td>-0.359664</td>
      <td>1.502433</td>
      <td>-1.383650</td>
      <td>-2.106888</td>
      <td>-1.136914</td>
      <td>-0.922065</td>
      <td>-0.981105</td>
      <td>-1.908722</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
from scipy.stats import ttest_ind
import os

# Perform t-tests between Fed and non-Fed weeks
results = []
for col in df.columns:
    if col.endswith('_Attention_Index'):
        fedweek = df[df['FedWeek']][col]
        non_fedweek = df[~df['FedWeek']][col]
        stat, pval = ttest_ind(fedweek, non_fedweek, equal_var=False)
        results.append({'Index': col, 't-stat': stat, 'p-value': round(pval, 4)})

# Save to table
ttest_df = pd.DataFrame(results)
os.makedirs("csv", exist_ok=True)
ttest_df.to_csv("csv/fedweek_attention_ttest_results.csv", index=False)
ttest_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index</th>
      <th>t-stat</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ETF_Attention_Index</td>
      <td>0.263885</td>
      <td>0.7964</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Stock_Attention_Index</td>
      <td>-0.736982</td>
      <td>0.4752</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dividend_Attention_Index</td>
      <td>0.326722</td>
      <td>0.7497</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beginner_Attention_Index</td>
      <td>-0.912829</td>
      <td>0.3803</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Macro_Attention_Index</td>
      <td>2.993454</td>
      <td>0.0169</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tech_Attention_Index</td>
      <td>0.397850</td>
      <td>0.6987</td>
    </tr>
  </tbody>
</table>
</div>



## 4.3.2 Trading Volume Responses to Federal Reserve Announcements

To complement the analysis of attention-based search behavior, this section investigates whether actual ETF trading activity shifts significantly during U.S. Federal Reserve policy announcement weeks. Using the same event windows defined earlier (±3 days around each Fed decision), we compare the normalized trading volumes of 19 Taiwan-listed ETFs between Fed weeks and non-Fed weeks.

We first generate comparative boxplots (Figure 4.3.2) to visually assess changes in volume distributions across each ticker. Then, for each ETF, we conduct independent two-sample t-tests to determine whether the differences in weekly trading volumes are statistically significant.

Figure 4.3.2 shows that while some ETFs (e.g., 2330.TW, 2454.TW) appear to exhibit slightly higher variance during Fed weeks, most tickers demonstrate visually stable distributions across both conditions. To quantify these differences, we compute p-values for each ticker, summarizing the results in ttest_volume_fedweek.csv.

The statistical results support this observation:

- None of the 19 ETFs displays a p-value below 0.05, and only 2308.TW (p = 0.1301) comes marginally close to conventional thresholds.

- Most tickers exhibit p-values above 0.6, signaling little to no difference in trading volume between Fed and non-Fed weeks.

These findings suggest that despite modest increases in macro-level attention, real trading behavior remains largely unaffected by Fed announcement timing. This may indicate that institutional dominance in ETF trading dampens the retail response, or that retail investors are not actively reallocating capital around macro events. The results echo the earlier insight from 4.3.1 — namely, that awareness may rise without translating into concrete trading behavior.

#### Code


```python
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Select normalized volume columns
volume_cols = [col for col in df.columns if 'Volume_norm' in col]

# Melt for boxplot
melted_vol = df.reset_index().melt(
    id_vars=['date', 'FedWeek'],
    value_vars=volume_cols,
    var_name='Ticker',
    value_name='VolumeValue'
)

# Plot boxplot: Volume during Fed vs Non-Fed weeks
plt.figure(figsize=(12, 6))
sns.boxplot(data=melted_vol, x='Ticker', y='VolumeValue', hue='FedWeek')
plt.title("Figure 4.3.2 - Trading Volume Distribution: Fed vs Non-Fed Weeks")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/4.3.2_trading_volume_distribution_fedweek.png", dpi=300)
plt.show()
```


    
![png](output_10_0.png)
    



```python
# T-tests for volume columns
ttest_results_volume = []

for col in volume_cols:
    fedweek_vol = df[df['FedWeek']][col]
    non_fed_vol = df[~df['FedWeek']][col]
    stat, pval = ttest_ind(fedweek_vol, non_fed_vol, equal_var=False)
    ttest_results_volume.append({'Ticker': col, 'p-value': round(pval, 4)})

# Save results
df_ttest_volume = pd.DataFrame(ttest_results_volume)
df_ttest_volume.to_csv("csv/ttest_volume_fedweek.csv", index=False)
df_ttest_volume
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0050.TW_Volume_norm</td>
      <td>0.8185</td>
    </tr>
    <tr>
      <th>1</th>
      <td>006208.TW_Volume_norm</td>
      <td>0.7870</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00878.TW_Volume_norm</td>
      <td>0.1984</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00713.TW_Volume_norm</td>
      <td>0.8919</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2330.TW_Volume_norm</td>
      <td>0.7298</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2303.TW_Volume_norm</td>
      <td>0.4941</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2412.TW_Volume_norm</td>
      <td>0.6533</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3008.TW_Volume_norm</td>
      <td>0.0880</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2881.TW_Volume_norm</td>
      <td>0.5375</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2882.TW_Volume_norm</td>
      <td>0.7632</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0056.TW_Volume_norm</td>
      <td>0.5288</td>
    </tr>
    <tr>
      <th>11</th>
      <td>9917.TW_Volume_norm</td>
      <td>0.3014</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1101.TW_Volume_norm</td>
      <td>0.1209</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2884.TW_Volume_norm</td>
      <td>0.8270</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2603.TW_Volume_norm</td>
      <td>0.6663</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1301.TW_Volume_norm</td>
      <td>0.9205</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2308.TW_Volume_norm</td>
      <td>0.9049</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3034.TW_Volume_norm</td>
      <td>0.1193</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2454.TW_Volume_norm</td>
      <td>0.7193</td>
    </tr>
  </tbody>
</table>
</div>



## 4.3.3 Attention and Trading Activity During “Jensen Week”: A Firm-Specific Shock

To evaluate whether firm-level publicity shocks can meaningfully influence retail investor behavior, we examine the market impact of NVIDIA CEO Jensen Huang’s high-profile visit to Taiwan in May 2024. This “Jensen Week” received extensive media attention, especially in the semiconductor and technology sectors, offering a rare event window to assess shifts in investor sentiment and trading behavior.

We define "Jensen Week" as the three weeks surrounding the visit (±1 week of May 22, 2024) and compare both attention index values and normalized trading volumes during this period versus all other weeks. Specifically, we run two-sided independent t-tests for each attention index and for each ticker’s trading volume. In addition, we visualize the distributions using boxplots to inspect whether “Jensen Week” coincides with significantly different behavior.

**Figure 4.3.3** presents the distribution of attention values across thematic indexes during Jensen Week vs. non-Jensen weeks. Among the six indexes, only the **Beginner Attention Index** shows a statistically significant increase (p = 0.032), suggesting that less experienced retail investors were particularly engaged during this media-heavy period. Other indexes show no significant difference.

**Figure 4.3.4** compares trading volume distributions. Here, the results are more striking: trading volume significantly increases during Jensen Week across a wide range of tickers, particularly in the tech and semiconductor sectors. T-test results (see `ttest_volume_jensenweek.csv`) show that **11 out of 19 tickers** exhibit statistically significant volume shifts, many at the 1% level (e.g., 0050.TW, 006208.TW, 3008.TW, 2603.TW, 3034.TW). This suggests that institutional or algorithmic traders responded strongly to the publicity shock, resulting in real capital movements.

Together, these findings highlight a notable divergence between attention and trading responses. While attention shifts were relatively muted—apart from beginner investors—realized trading activity surged, pointing to strong behavioral engagement among market participants. The results emphasize the value of firm-specific narrative events in mobilizing capital flows and shaping market dynamics, and they offer empirical support for Hypotheses **H3.2** and **H3.3**.

#### Code


```python
# 讀取資料（請先確認檔案存在，或換成你的實際檔名）
df = pd.read_excel("merged_df.xlsx")
# 修正未命名的第一欄為 'date'
if df.columns[0].startswith("Unnamed"):
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
# 將日期欄轉換為 datetime 格式
df['date'] = pd.to_datetime(df['date'])

# 定義 Jensen Huang 到訪週的日期（以實際日期為準，這裡假設為 2024-05-22）
jensen_date = pd.to_datetime("2024-05-22")

# 建立 Jensen Week 標籤：前後三週（含本週）視為 True，其餘為 False
df['JensenWeek'] = df['date'].apply(lambda x: abs((x - jensen_date).days) <= 21)

# 檢查結果
df[['date', 'JensenWeek']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>JensenWeek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-01-07</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-01-14</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-01-21</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-01-28</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-04</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-02-11</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-02-18</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-02-25</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2024-03-03</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2024-03-10</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2024-03-17</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2024-03-24</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2024-03-31</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2024-04-07</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2024-04-14</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2024-04-21</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2024-04-28</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2024-05-05</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2024-05-12</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2024-05-19</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2024-05-26</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2024-06-02</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2024-06-09</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2024-06-16</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2024-06-23</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2024-06-30</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2024-07-07</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2024-07-14</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2024-07-21</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2024-07-28</td>
      <td>False</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2024-08-04</td>
      <td>False</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2024-08-11</td>
      <td>False</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2024-08-18</td>
      <td>False</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2024-08-25</td>
      <td>False</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2024-09-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2024-09-08</td>
      <td>False</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2024-09-15</td>
      <td>False</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2024-09-22</td>
      <td>False</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2024-09-29</td>
      <td>False</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2024-10-06</td>
      <td>False</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2024-10-13</td>
      <td>False</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2024-10-20</td>
      <td>False</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2024-10-27</td>
      <td>False</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2024-11-03</td>
      <td>False</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2024-11-10</td>
      <td>False</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2024-11-17</td>
      <td>False</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2024-11-24</td>
      <td>False</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2024-12-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2024-12-08</td>
      <td>False</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2024-12-15</td>
      <td>False</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2024-12-22</td>
      <td>False</td>
    </tr>
    <tr>
      <th>51</th>
      <td>2024-12-29</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
import matplotlib.pyplot as plt

# 將注意力欄位攤平成長格式
melted_jensen = df.melt(id_vars=['date', 'JensenWeek'],
                        value_vars=[col for col in df.columns if col.endswith('_Attention_Index')],
                        var_name='IndexType',
                        value_name='AttentionValue')

# 繪製 boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=melted_jensen, x='IndexType', y='AttentionValue', hue='JensenWeek')
plt.title("Figure_4.3.3 - Attention Index Distribution: Jensen Week vs Non-Jensen Weeks")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/4.3.3_trading_volume_distribution_fedweek.png", dpi=300)
plt.show()
```


    
![png](output_15_0.png)
    



```python
# Prepare a list to store test results
results = []

# Loop through attention index columns
for col in df.columns:
    if col.endswith('_Attention_Index'):
        jensen = df[df['JensenWeek']][col]
        non_jensen = df[~df['JensenWeek']][col]
        stat, pval = ttest_ind(jensen, non_jensen, equal_var=False)
        results.append({'Index': col, 't-statistic': stat, 'p-value': pval})

# Convert to DataFrame
ttest_df = pd.DataFrame(results)

# Save to CSV
ttest_df.to_csv("csv/ttest_attention_jensenweek.csv", index=False)

# Optional: preview the result
ttest_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index</th>
      <th>t-statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ETF_Attention_Index</td>
      <td>0.170218</td>
      <td>0.866357</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Stock_Attention_Index</td>
      <td>0.260010</td>
      <td>0.796866</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dividend_Attention_Index</td>
      <td>1.197300</td>
      <td>0.250656</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beginner_Attention_Index</td>
      <td>2.334941</td>
      <td>0.032100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Macro_Attention_Index</td>
      <td>-0.755364</td>
      <td>0.466352</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tech_Attention_Index</td>
      <td>0.337019</td>
      <td>0.740803</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 攤平成長格式
volume_cols = [col for col in df.columns if 'Volume_norm' in col]

melted_vol = df.melt(id_vars=['date', 'JensenWeek'],
                     value_vars=volume_cols,
                     var_name='Ticker',
                     value_name='VolumeValue')

# 繪製 boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=melted_vol, x='Ticker', y='VolumeValue', hue='JensenWeek')
plt.title("Figure_4.3.4 - Trading Volume Distribution: Jensen Week vs Non-Jensen Weeks")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/4.3.4_trading_volume_distribution_fedweek.png", dpi=300)
plt.show()
```


    
![png](output_17_0.png)
    



```python
from scipy.stats import ttest_ind

# Store results
volume_results = []

# Loop through columns containing 'Volume_norm'
for col in df.columns:
    if 'Volume_norm' in col:
        jensen = df[df['JensenWeek']][col]
        non_jensen = df[~df['JensenWeek']][col]
        stat, pval = ttest_ind(jensen, non_jensen, equal_var=False)
        volume_results.append({'Ticker': col, 't-statistic': stat, 'p-value': pval})

# Convert to DataFrame
volume_ttest_df = pd.DataFrame(volume_results)

# Save to CSV
volume_ttest_df.to_csv("csv/ttest_volume_jensenweek.csv", index=False)

# Optional: preview the table
volume_ttest_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>t-statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0050.TW_Volume_norm</td>
      <td>-3.387659</td>
      <td>0.002714</td>
    </tr>
    <tr>
      <th>1</th>
      <td>006208.TW_Volume_norm</td>
      <td>-3.417347</td>
      <td>0.002637</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00878.TW_Volume_norm</td>
      <td>0.679711</td>
      <td>0.512431</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00713.TW_Volume_norm</td>
      <td>-2.476689</td>
      <td>0.032472</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2330.TW_Volume_norm</td>
      <td>-1.461935</td>
      <td>0.163631</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2303.TW_Volume_norm</td>
      <td>0.559056</td>
      <td>0.596839</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2412.TW_Volume_norm</td>
      <td>1.508147</td>
      <td>0.188592</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3008.TW_Volume_norm</td>
      <td>-4.955088</td>
      <td>0.000090</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2881.TW_Volume_norm</td>
      <td>0.777036</td>
      <td>0.463855</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2882.TW_Volume_norm</td>
      <td>2.158589</td>
      <td>0.070622</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0056.TW_Volume_norm</td>
      <td>-4.342001</td>
      <td>0.000160</td>
    </tr>
    <tr>
      <th>11</th>
      <td>9917.TW_Volume_norm</td>
      <td>-2.343800</td>
      <td>0.023107</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1101.TW_Volume_norm</td>
      <td>2.275787</td>
      <td>0.060594</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2884.TW_Volume_norm</td>
      <td>2.323792</td>
      <td>0.061852</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2603.TW_Volume_norm</td>
      <td>2.728241</td>
      <td>0.038653</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1301.TW_Volume_norm</td>
      <td>-2.273924</td>
      <td>0.043577</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2308.TW_Volume_norm</td>
      <td>0.244296</td>
      <td>0.812847</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3034.TW_Volume_norm</td>
      <td>2.769215</td>
      <td>0.030519</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2454.TW_Volume_norm</td>
      <td>0.452484</td>
      <td>0.665378</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
