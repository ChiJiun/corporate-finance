# 三資產 Mean-Variance Efficient Frontier

這個專案會抓取聯發科、第一金與黃金資料，將黃金美元報價用 USD/TWD 匯率換成台幣計價，再計算年化報酬、年化共變異數矩陣、蒙地卡羅投組與效率前緣。

## 執行

```powershell
python mean_variance_frontier.py
```

## 資產與資料來源

- 聯發科：`2454.TW`
- 第一金：`2892.TW`
- 黃金：`GC=F`
- 匯率：`TWD=X`，Yahoo Finance 的 USD/TWD，也就是 1 美元兌台幣

黃金台幣價格的計算方式：

```text
Gold (TWD) = GC=F close price in USD * USD/TWD
```

## 輸出

執行後會產生 `outputs/`：

- `prices_twd.csv`：三資產台幣基礎價格
- `daily_returns.csv`：日報酬率
- `annualized_covariance_matrix.csv`：年化共變異數矩陣
- `monte_carlo_portfolios.csv`：50,000 組隨機權重模擬
- `efficient_frontier.csv`：最佳化求出的效率前緣
- `normalized_prices_twd.png`：標準化價格走勢
- `efficient_frontier.png`：蒙地卡羅散點與效率前緣圖

