# Acquiring the Data
<div class="alert alert-block alert-danger">
<b>Disclaimer:</b>

The provided scripts are raw and exploratory in nature. The research questions and objectives were iteratively refined throughout the project as data availability improved and our understanding of the dataset deepened.
</div>

---
## Data Acquisition and Construction of Attention Indexes

This study begins by constructing six thematic Attention Indexes using Google Trends data, each reflecting a distinct retail investor focus in the Taiwanese market: ETFs, individual stocks, dividends, macro-sensitive sectors, technology stocks, and beginner-friendly investments. For each theme, multiple related keywords were selected and queried using the pytrends API. The search volume data for 2024 was normalized and aggregated to form composite weekly indexes that quantify shifts in public interest. These indexes serve as behavioral indicators capturing attention dynamics across different investment mindsets.

To align investor attention with actual market activity, we retrieved weekly trading volumes for 19 representative TWSE-listed tickers using the yfinance library. These stocks were chosen based on their relevance to the attention themes, ensuring consistency between behavioral and market-based data. Trading volumes were normalized, and the resulting dataset was merged with the attention indexes along a weekly time axis to create a unified panel.

This dataset forms the basis for addressing our core research questions:

1. Do changes in public attention precede movements in trading activity (RQ1)?

2. Can attention indexes improve short-term predictive models (RQ2)?

3. Do major external events—such as U.S. Fed meetings or visits by industry leaders—simultaneously affect both investor attention and market engagement (RQ3)?

---


```python
# If you have never used pytrends, you should install it
#!pip install pytrends
import pandas as pd
from pytrends.request import TrendReq
import time
import yfinance as yf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```

## Building Subgroup Attention Indexes Using Google Trends

<div class="alert alert-block alert-danger">
<b>Warning:</b>

The following cell (I have turned it into a markdown cell just in case.) might fail if you run it too many times, as pytrends limit requests per IP address. For some reason, I can't get the same exact code to acquire the data, maybe my IP address is blocked by Google Trends. However, you still may get the data if you are careful with the process.
</div>

This section of the research builds a comprehensive picture of retail investor attention in Taiwan by analyzing search behavior from Google Trends. Instead of relying on a single keyword, we group related search terms into thematic clusters—such as ETFs, dividends, macroeconomics, and beginner investing—and create composite "attention indexes" that represent different investor mindsets. These indexes serve as behavioral signals that we can later compare to actual trading activity, test for predictability, and observe under macroeconomic shocks. By capturing multiple dimensions of attention, we aim to better understand how public interest reflects or influences financial market behavior.

```
# Initialize pytrends
pytrends = TrendReq(hl='zh-TW', tz=360)

# Define keyword subgroups
subgroups = {
    "ETF_Attention_Index": ['ETF 投資', '0050', '高股息 ETF', '00878', 'ETF 定期定額'],
    "Stock_Attention_Index": ['投資 股票', '台股 投資', '2330', '台積電', '當沖'],
    "Dividend_Attention_Index": ['高股息', '殖利率', '存股', '金融股', '配息'],
    "Beginner_Attention_Index": ['股票是什麼', '怎麼投資', '證券開戶', '股市新手', '股票入門'],
    "Macro_Attention_Index": ['升息', '通膨', '美國股市', 'FED', '經濟衰退'],
    "Tech_Attention_Index": ['半導體', '台積電', 'AI 投資', '高科技股', 'IC 設計']
}

# Timeframe and location
timeframe = '2024-01-01 2024-12-31'
geo = 'TW'

# Container for results
index_dfs = []

# Loop with 5-second delay
for index_name, keyword_list in subgroups.items():
    try:
        print(f"Fetching: {index_name}...")
        pytrends.build_payload(keyword_list, timeframe=timeframe, geo=geo)
        time.sleep(5)  # Delay to avoid 429 rate limit
        
        df = pytrends.interest_over_time().drop(columns='isPartial')
        df.columns = [col.replace(" ", "_") for col in df.columns]
        
        # Normalize
        df_norm = (df - df.mean()) / df.std()
        df_norm[index_name] = df_norm.mean(axis=1)
        
        index_dfs.append(df_norm[[index_name]])
    except Exception as e:
        print(f"Failed to fetch {index_name}: {e}")
        continue

# Merge all into one DataFrame
attention_index_df = pd.concat(index_dfs, axis=1)

# Show preview
attention_index_df.head()

# Save to Excel
attention_index_df.to_excel('attention_index_data.xlsx')
```

<div class="alert alert-warning">
<b>Message:</b> 
    
In case it doesn't run successfuly, I provided a link to the acquired data. Please check it out, I wouldn't delete it before the Spring semester of 2025 ends. And, if I do, I'm pretty sure that I'll put the `.csv` file in my repository.

</div>

Here is the link: [https://docs.google.com/spreadsheets/d/1TDK94m3D_oqx_hV-NZ5SwGBJXWGo9XmR/edit?usp=sharing&ouid=103068230126415922496&rtpof=true&sd=true](https://docs.google.com/spreadsheets/d/1TDK94m3D_oqx_hV-NZ5SwGBJXWGo9XmR/edit?usp=sharing&ouid=103068230126415922496&rtpof=true&sd=true)

<div class="alert alert-block alert-danger">
<b>Warning:</b>

You need to put the `attention_index_data.xlsx` file in the same folder as this Python script in order for the cell below to run.
</div>


```python
attention_index_df = pd.read_excel('attention_index_data.xlsx', index_col=0)
```

## Merging Weekly Market Volume with Attention Indexes

This step connects behavioral data with actual market behavior. By combining Google Trends-based attention indexes with real-world trading volume, we create a unified dataset that allows us to explore how investor interest aligns with or influences financial activity. This merged view enables descriptive comparisons (e.g., trend co-movement), statistical correlation analysis (RQ1), and predictive modeling (RQ2). It also allows us to examine whether external events like Fed announcements shift both attention and market engagement (RQ3). Aligning these time series on a weekly basis ensures consistency and comparability across all variables.

### Mapping Attention Indexes to Representative TWSE Stocks

To ensure that our stock universe reflects the themes captured by each attention index, we selected representative TWSE stocks for each attention category:

| Attention Index             | Keywords                                                                 | Matched Tickers                                          |
|----------------------------|--------------------------------------------------------------------------|----------------------------------------------------------|
| **ETF_Attention_Index**     | "ETF 投資", "0050", "高股息 ETF", "00878", "ETF 定期定額"                 | "0050.TW", "006208.TW", "00878.TW", "00713.TW"            |
| **Stock_Attention_Index**   | "投資 股票", "台股 投資", "2330", "台積電", "當沖"                         | "2330.TW", "2303.TW", "2412.TW", "3008.TW"                |
| **Dividend_Attention_Index**| "高股息", "殖利率", "存股", "金融股", "配息"                               | "2881.TW", "2882.TW", "0056.TW", "1101.TW"                |
| **Beginner_Attention_Index**| "股票是什麼", "怎麼投資", "證券開戶", "股市新手", "股票入門"             | "9917.TW", "2603.TW", "2884.TW"                           |
| **Macro_Attention_Index**   | "升息", "通膨", "美國股市", "FED", "經濟衰退"                             | "1301.TW", "2308.TW"                                     |
| **Tech_Attention_Index**    | "半導體", "台積電", "AI 投資", "高科技股", "IC 設計"                      | "3034.TW", "2454.TW"                                     |

This logic ensures that our volume-based market signals are well-aligned with the **public attention captured in search behavior**, providing a meaningful basis for correlation and predictive analysis.


```python
# Define tickers you care about
tickers = [
    '0050.TW', '006208.TW', '00878.TW', '00713.TW',   # ETF-related
    '2330.TW', '2303.TW', '2412.TW', '3008.TW',       # Stock-following
    '2881.TW', '2882.TW', '0056.TW', '9917.TW', '1101.TW',  # Dividend
    '2884.TW', '2603.TW',                             # Beginner-friendly
    '1301.TW', '2308.TW',                             # Macro-sensitive
    '3034.TW', '2454.TW'                              # Tech-specific
]

start_date = '2024-01-01'
end_date = '2025-01-01'

# Download daily data
prices = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# Resample weekly volume and normalize
volume_dfs = []
for ticker in tickers:
    vol = prices[ticker]['Volume'].resample('W-SUN').sum()
    vol_norm = (vol - vol.mean()) / vol.std()
    volume_dfs.append(vol_norm.rename(f"{ticker}_Volume_norm"))

# Combine all volumes
volume_df = pd.concat(volume_dfs, axis=1)

# Merge with attention index
merged_df = pd.merge(volume_df, attention_index_df, left_index=True, right_index=True, how='inner')

# Preview merged data
merged_df.head()
```

    YF.download() has changed argument auto_adjust default to True


    [*********************100%***********************]  19 of 19 completed





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
      <th>0050.TW_Volume_norm</th>
      <th>006208.TW_Volume_norm</th>
      <th>00878.TW_Volume_norm</th>
      <th>00713.TW_Volume_norm</th>
      <th>2330.TW_Volume_norm</th>
      <th>2303.TW_Volume_norm</th>
      <th>2412.TW_Volume_norm</th>
      <th>3008.TW_Volume_norm</th>
      <th>2881.TW_Volume_norm</th>
      <th>2882.TW_Volume_norm</th>
      <th>...</th>
      <th>1301.TW_Volume_norm</th>
      <th>2308.TW_Volume_norm</th>
      <th>3034.TW_Volume_norm</th>
      <th>2454.TW_Volume_norm</th>
      <th>ETF_Attention_Index</th>
      <th>Stock_Attention_Index</th>
      <th>Dividend_Attention_Index</th>
      <th>Beginner_Attention_Index</th>
      <th>Macro_Attention_Index</th>
      <th>Tech_Attention_Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-01-07</th>
      <td>-0.860674</td>
      <td>-0.932212</td>
      <td>-0.379417</td>
      <td>-0.874601</td>
      <td>-1.254108</td>
      <td>-0.166771</td>
      <td>-0.911028</td>
      <td>0.418549</td>
      <td>-1.167018</td>
      <td>-1.203771</td>
      <td>...</td>
      <td>-1.403838</td>
      <td>-1.300789</td>
      <td>-0.466500</td>
      <td>0.439787</td>
      <td>-1.216050</td>
      <td>-1.367152</td>
      <td>-0.199133</td>
      <td>-0.899066</td>
      <td>0.390401</td>
      <td>-0.575975</td>
    </tr>
    <tr>
      <th>2024-01-14</th>
      <td>-0.653673</td>
      <td>-0.931290</td>
      <td>-0.708484</td>
      <td>-0.822152</td>
      <td>-1.295240</td>
      <td>-0.650347</td>
      <td>-0.810122</td>
      <td>1.288575</td>
      <td>-0.906104</td>
      <td>-0.998040</td>
      <td>...</td>
      <td>-0.842457</td>
      <td>-0.264088</td>
      <td>-0.352294</td>
      <td>-0.322731</td>
      <td>-0.116791</td>
      <td>-0.582552</td>
      <td>0.236205</td>
      <td>-0.666854</td>
      <td>-0.072779</td>
      <td>-0.180896</td>
    </tr>
    <tr>
      <th>2024-01-21</th>
      <td>1.345122</td>
      <td>-0.478630</td>
      <td>-0.280000</td>
      <td>-0.611415</td>
      <td>1.391158</td>
      <td>0.676357</td>
      <td>-0.147496</td>
      <td>0.498769</td>
      <td>0.186507</td>
      <td>-0.045758</td>
      <td>...</td>
      <td>0.223543</td>
      <td>0.332889</td>
      <td>1.014547</td>
      <td>0.856694</td>
      <td>-0.433229</td>
      <td>-0.582552</td>
      <td>-0.121921</td>
      <td>-0.173195</td>
      <td>-0.007083</td>
      <td>-0.180896</td>
    </tr>
    <tr>
      <th>2024-01-28</th>
      <td>0.804586</td>
      <td>-0.368364</td>
      <td>-0.302921</td>
      <td>-0.923418</td>
      <td>0.623496</td>
      <td>1.909554</td>
      <td>-0.861677</td>
      <td>-0.341718</td>
      <td>-0.874992</td>
      <td>-1.063764</td>
      <td>...</td>
      <td>-0.901903</td>
      <td>-0.248834</td>
      <td>-0.524553</td>
      <td>0.512786</td>
      <td>-0.119907</td>
      <td>-1.067593</td>
      <td>-0.194599</td>
      <td>-0.563430</td>
      <td>-0.396802</td>
      <td>-0.685401</td>
    </tr>
    <tr>
      <th>2024-02-04</th>
      <td>0.260986</td>
      <td>-0.821956</td>
      <td>-0.410453</td>
      <td>-1.048354</td>
      <td>-0.096706</td>
      <td>-0.126591</td>
      <td>-0.394153</td>
      <td>0.253704</td>
      <td>-0.933695</td>
      <td>-0.997090</td>
      <td>...</td>
      <td>-0.971914</td>
      <td>-0.339969</td>
      <td>-0.359664</td>
      <td>1.502433</td>
      <td>-1.383650</td>
      <td>-2.106888</td>
      <td>-1.136914</td>
      <td>-0.922065</td>
      <td>-0.981105</td>
      <td>-1.908722</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
merged_df.to_excel('merged_df.xlsx')
```
