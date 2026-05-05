from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.exceptions import JSONDecodeError
from scipy.optimize import minimize


ASSETS = {
    "MediaTek": "2454.TW",
    "First Financial": "2892.TW",
    "Gold (TWD)": "GC=F",
}
FX_TICKER = "TWD=X"  # Yahoo chart API: TWD per 1 USD.
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
N_SIMULATIONS = 50_000
RANDOM_SEED = 42
N_FRONTIER_POINTS = 120


@dataclass(frozen=True)
class Portfolio:
    weights: np.ndarray
    annual_return: float
    annual_volatility: float
    sharpe: float


def fetch_yahoo_chart(ticker: str, start: str, end: str | None, prefer_adjusted: bool) -> pd.Series:
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_date = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) if end else pd.Timestamp.utcnow()
    end_ts = int(end_date.timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error for {ticker}: {chart['error']}")

    result = chart.get("result") or []
    if not result or not result[0].get("timestamp"):
        raise RuntimeError(f"No Yahoo chart data returned for {ticker}")

    data = result[0]
    dates = pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
    quote = data["indicators"]["quote"][0]
    close = quote.get("close")
    values = None
    if prefer_adjusted:
        adjclose = data["indicators"].get("adjclose") or []
        if adjclose and adjclose[0].get("adjclose"):
            values = adjclose[0]["adjclose"]
    if values is None:
        values = close

    series = pd.Series(values, index=dates, name=ticker)
    return pd.to_numeric(series, errors="coerce").dropna().sort_index()


def fetch_prices_from_yahoo(start: str, end: str | None) -> pd.DataFrame:
    mediatek = fetch_yahoo_chart(ASSETS["MediaTek"], start, end, prefer_adjusted=True).rename(
        "MediaTek"
    )
    first_financial = fetch_yahoo_chart(
        ASSETS["First Financial"], start, end, prefer_adjusted=True
    ).rename("First Financial")
    gold_usd = fetch_yahoo_chart(ASSETS["Gold (TWD)"], start, end, prefer_adjusted=False).rename(
        "Gold (USD)"
    )
    usd_twd = fetch_yahoo_chart(FX_TICKER, start, end, prefer_adjusted=False).rename("USD/TWD")

    raw = pd.concat([mediatek, first_financial, gold_usd, usd_twd], axis=1).sort_index()
    raw[["Gold (USD)", "USD/TWD"]] = raw[["Gold (USD)", "USD/TWD"]].ffill()
    raw["Gold (TWD)"] = raw["Gold (USD)"] * raw["USD/TWD"]
    prices = raw[["MediaTek", "First Financial", "Gold (TWD)"]].dropna()

    if len(prices) < 30:
        raise RuntimeError(f"Only {len(prices)} aligned Yahoo observations.")

    return prices


def month_starts(start: str, end: str | None) -> list[date]:
    start_dt = pd.Timestamp(start).date().replace(day=1)
    end_dt = (pd.Timestamp(end).date() if end else date.today()).replace(day=1)
    return [ts.date() for ts in pd.date_range(start_dt, end_dt, freq="MS")]


def parse_twse_date(value: str) -> pd.Timestamp:
    year, month, day = value.split("/")
    return pd.Timestamp(int(year) + 1911, int(month), int(day))


def fetch_twse_stock(stock_no: str, start: str, end: str | None) -> pd.Series:
    frames = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    months = month_starts(start, end)

    for month_start in months:
        url = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
        params = {
            "date": month_start.strftime("%Y%m%d"),
            "stockNo": stock_no.replace(".TW", ""),
            "response": "json",
        }
        payload = None
        for attempt in range(3):
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            try:
                payload = response.json()
                break
            except JSONDecodeError:
                if attempt == 2:
                    preview = response.text[:200].replace("\n", " ")
                    if month_start == months[-1]:
                        payload = {"stat": "NO_DATA", "data": []}
                        break
                    raise RuntimeError(
                        f"TWSE returned non-JSON data for {stock_no} "
                        f"{month_start:%Y-%m}: {preview}"
                    ) from None
                sleep(1.0 + attempt)

        if payload is None:
            continue
        if payload.get("stat") == "OK" and payload.get("data"):
            frame = pd.DataFrame(payload["data"], columns=payload["fields"])
            frame["Date"] = frame["日期"].map(parse_twse_date)
            frame["Close"] = pd.to_numeric(
                frame["收盤價"].str.replace(",", "", regex=False),
                errors="coerce",
            )
            frames.append(frame[["Date", "Close"]])
        sleep(0.1)

    if not frames:
        raise RuntimeError(f"No TWSE data returned for stock {stock_no}")

    series = pd.concat(frames).dropna().drop_duplicates("Date").set_index("Date")["Close"]
    return series.loc[pd.Timestamp(start) : pd.Timestamp(end) if end else None].sort_index()


def fetch_usd_twd(start: str, end: str | None) -> pd.Series:
    url = "https://cpx.cbc.gov.tw/API/DataAPI/Get"
    response = requests.get(url, params={"FileName": "BP01D01en"}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    columns = ["Date"] + [item["data"] for item in payload["data"]["structure"]["Table1"]]
    frame = pd.DataFrame(payload["data"]["dataSets"], columns=columns)
    frame["Date"] = pd.to_datetime(frame["Date"], format="%Y%m%d")
    frame["NTD/USD"] = pd.to_numeric(frame["NTD/USD"], errors="coerce")
    series = frame.dropna(subset=["NTD/USD"]).set_index("Date")["NTD/USD"]
    return series.loc[pd.Timestamp(start) : pd.Timestamp(end) if end else None].sort_index()


def fetch_gold_usd(start: str, end: str | None) -> pd.Series:
    url = "https://freegoldapi.com/data/latest.csv"
    frame = pd.read_csv(url)
    frame = frame[frame["source"].eq("yahoo_finance")].copy()
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    daily = frame.dropna(subset=["date", "price"])
    series = daily.drop_duplicates("date").set_index("date")["price"]
    series = series.loc[pd.Timestamp(start) : pd.Timestamp(end) if end else None].sort_index()
    if series.empty:
        raise RuntimeError(
            "No daily gold data available for the requested period. "
            "Use a start date in 2025 or later, or provide another daily gold source."
        )
    if series.index.min() > pd.Timestamp(start) + pd.Timedelta(days=30):
        raise RuntimeError(
            "Fallback gold source does not cover the requested start date with daily data. "
            "Yahoo chart data is required for this sample period."
        )
    return series


def fetch_prices(start: str, end: str | None) -> pd.DataFrame:
    try:
        return fetch_prices_from_yahoo(start, end)
    except Exception as yahoo_error:
        print(f"Yahoo chart source failed, trying public-source fallback: {yahoo_error}")

    mediatek = fetch_twse_stock(ASSETS["MediaTek"], start, end).rename("MediaTek")
    first_financial = fetch_twse_stock(ASSETS["First Financial"], start, end).rename("First Financial")
    usd_twd = fetch_usd_twd(start, end).rename("USD/TWD")
    gold_usd = fetch_gold_usd(start, end).rename("Gold (USD)")

    raw = pd.concat([mediatek, first_financial, gold_usd, usd_twd], axis=1).sort_index()
    raw[["Gold (USD)", "USD/TWD"]] = raw[["Gold (USD)", "USD/TWD"]].ffill()
    raw["Gold (TWD)"] = raw["Gold (USD)"] * raw["USD/TWD"]
    prices = raw[["MediaTek", "First Financial", "Gold (TWD)"]].dropna()

    if len(prices) < 30:
        raise RuntimeError(f"Only {len(prices)} aligned observations. Check source data coverage.")

    return prices


def portfolio_stats(
    weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame
) -> Portfolio:
    annual_return = float(weights @ mean_returns.to_numpy())
    annual_volatility = float(np.sqrt(weights @ cov_matrix.to_numpy() @ weights))
    sharpe = (
        (annual_return - RISK_FREE_RATE) / annual_volatility
        if annual_volatility > 0
        else np.nan
    )
    return Portfolio(weights, annual_return, annual_volatility, sharpe)


def minimize_volatility_for_return(
    target_return: float, mean_returns: pd.Series, cov_matrix: pd.DataFrame
) -> Portfolio:
    n_assets = len(mean_returns)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: w @ mean_returns.to_numpy() - target_return},
    )

    result = minimize(
        lambda w: np.sqrt(w @ cov_matrix.to_numpy() @ w),
        x0=np.repeat(1.0 / n_assets, n_assets),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed for target {target_return:.4f}: {result.message}")

    return portfolio_stats(result.x, mean_returns, cov_matrix)


def maximize_sharpe(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Portfolio:
    n_assets = len(mean_returns)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    result = minimize(
        lambda w: -portfolio_stats(w, mean_returns, cov_matrix).sharpe,
        x0=np.repeat(1.0 / n_assets, n_assets),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"Sharpe optimization failed: {result.message}")

    return portfolio_stats(result.x, mean_returns, cov_matrix)


def minimum_variance(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Portfolio:
    n_assets = len(mean_returns)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    result = minimize(
        lambda w: np.sqrt(w @ cov_matrix.to_numpy() @ w),
        x0=np.repeat(1.0 / n_assets, n_assets),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"Minimum variance optimization failed: {result.message}")

    return portfolio_stats(result.x, mean_returns, cov_matrix)


def maximum_return(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Portfolio:
    weights = np.zeros(len(mean_returns))
    weights[int(np.argmax(mean_returns.to_numpy()))] = 1.0
    return portfolio_stats(weights, mean_returns, cov_matrix)


def monte_carlo_portfolios(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_simulations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weight_matrix = rng.dirichlet(np.ones(len(mean_returns)), size=n_simulations)

    returns = weight_matrix @ mean_returns.to_numpy()
    volatilities = np.sqrt(np.einsum("ij,jk,ik->i", weight_matrix, cov_matrix.to_numpy(), weight_matrix))
    sharpes = (returns - RISK_FREE_RATE) / volatilities

    simulated = pd.DataFrame(weight_matrix, columns=[f"w_{name}" for name in mean_returns.index])
    simulated["annual_return"] = returns
    simulated["annual_volatility"] = volatilities
    simulated["sharpe"] = sharpes
    return simulated


def plot_frontier(
    simulated: pd.DataFrame,
    frontier: pd.DataFrame,
    points: dict[str, Portfolio],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    scatter = ax.scatter(
        simulated["annual_volatility"],
        simulated["annual_return"],
        c=simulated["sharpe"],
        cmap="viridis",
        s=8,
        alpha=0.35,
        label="Monte Carlo portfolios",
    )
    ax.plot(
        frontier["annual_volatility"],
        frontier["annual_return"],
        color="#d62728",
        linewidth=2.5,
        label="Efficient frontier",
    )

    markers = {
        "Maximum return": ("*", "#ff7f0e", 220),
        "Maximum Sharpe": ("D", "#1f77b4", 90),
        "Minimum variance": ("X", "#2ca02c", 100),
    }
    for label, portfolio in points.items():
        marker, color, size = markers[label]
        ax.scatter(
            portfolio.annual_volatility,
            portfolio.annual_return,
            marker=marker,
            color=color,
            s=size,
            edgecolor="black",
            linewidth=0.7,
            label=label,
            zorder=5,
        )

    ax.set_title("Mean-Variance Efficient Frontier")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_prices(prices: pd.DataFrame, output_path: Path) -> None:
    normalized = prices / prices.iloc[0]
    ax = normalized.plot(figsize=(11, 6), linewidth=1.6)
    ax.set_title("Normalized Prices, TWD Basis")
    ax.set_xlabel("")
    ax.set_ylabel("Growth of 1 TWD")
    ax.grid(True, alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Return Correlation Matrix")
    ax.set_xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(correlation_matrix.index)), correlation_matrix.index)

    for row in range(len(correlation_matrix.index)):
        for col in range(len(correlation_matrix.columns)):
            value = correlation_matrix.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="black")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def portfolio_table(points: dict[str, Portfolio], asset_names: list[str]) -> pd.DataFrame:
    rows = []
    for label, portfolio in points.items():
        row = {
            "portfolio": label,
            "annual_return": portfolio.annual_return,
            "annual_volatility": portfolio.annual_volatility,
            "sharpe": portfolio.sharpe,
        }
        for asset_name, weight in zip(asset_names, portfolio.weights, strict=True):
            row[f"w_{asset_name}"] = weight
        rows.append(row)
    return pd.DataFrame(rows)


def plot_portfolio_weights(portfolios: pd.DataFrame, asset_names: list[str], output_path: Path) -> None:
    weight_columns = [f"w_{name}" for name in asset_names]
    plot_data = portfolios.set_index("portfolio")[weight_columns]
    plot_data.columns = asset_names

    ax = plot_data.plot(kind="bar", stacked=True, figsize=(10, 6), width=0.65)
    ax.set_title("Optimized Portfolio Weights")
    ax.set_xlabel("")
    ax.set_ylabel("Portfolio weight")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(asset_names))
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{value:.0%}" if value >= 0.04 else "" for value in container.datavalues],
            label_type="center",
            fontsize=9,
        )
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_return_distribution(daily_returns: pd.DataFrame, output_path: Path) -> None:
    ax = daily_returns.plot(kind="hist", bins=45, alpha=0.5, figsize=(10, 6))
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Daily return")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="y", alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_portfolio(label: str, portfolio: Portfolio, asset_names: list[str]) -> None:
    print(f"\n{label}")
    print(f"  Annual return:     {portfolio.annual_return:8.2%}")
    print(f"  Annual volatility: {portfolio.annual_volatility:8.2%}")
    print(f"  Sharpe ratio:      {portfolio.sharpe:8.3f}")
    for name, weight in zip(asset_names, portfolio.weights, strict=True):
        print(f"  {name:18s}: {weight:8.2%}")


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    prices = fetch_prices(start=START_DATE, end=END_DATE)
    daily_returns = prices.pct_change().dropna()
    mean_returns = daily_returns.mean() * TRADING_DAYS
    cov_matrix = daily_returns.cov() * TRADING_DAYS

    simulated = monte_carlo_portfolios(
        mean_returns,
        cov_matrix,
        n_simulations=N_SIMULATIONS,
        seed=RANDOM_SEED,
    )

    min_return = float(mean_returns.min())
    max_return = float(mean_returns.max())
    target_returns = np.linspace(min_return, max_return, N_FRONTIER_POINTS)
    frontier_portfolios = [
        minimize_volatility_for_return(target, mean_returns, cov_matrix)
        for target in target_returns
    ]
    frontier = pd.DataFrame(
        {
            "annual_return": [p.annual_return for p in frontier_portfolios],
            "annual_volatility": [p.annual_volatility for p in frontier_portfolios],
            "sharpe": [p.sharpe for p in frontier_portfolios],
        }
    )
    for i, name in enumerate(mean_returns.index):
        frontier[f"w_{name}"] = [p.weights[i] for p in frontier_portfolios]

    points = {
        "Maximum return": maximum_return(mean_returns, cov_matrix),
        "Maximum Sharpe": maximize_sharpe(mean_returns, cov_matrix),
        "Minimum variance": minimum_variance(mean_returns, cov_matrix),
    }
    asset_names = list(mean_returns.index)
    optimized_portfolios = portfolio_table(points, asset_names)
    correlation_matrix = daily_returns.corr()

    prices.to_csv(output_dir / "prices_twd.csv")
    daily_returns.to_csv(output_dir / "daily_returns.csv")
    mean_returns.to_csv(output_dir / "annualized_mean_returns.csv", header=["annual_return"])
    cov_matrix.to_csv(output_dir / "annualized_covariance_matrix.csv")
    correlation_matrix.to_csv(output_dir / "correlation_matrix.csv")
    simulated.to_csv(output_dir / "monte_carlo_portfolios.csv", index=False)
    frontier.to_csv(output_dir / "efficient_frontier.csv", index=False)
    optimized_portfolios.to_csv(output_dir / "optimized_portfolios.csv", index=False)
    plot_prices(prices, output_dir / "normalized_prices_twd.png")
    plot_frontier(simulated, frontier, points, output_dir / "efficient_frontier.png")
    plot_correlation_heatmap(correlation_matrix, output_dir / "correlation_heatmap.png")
    plot_portfolio_weights(optimized_portfolios, asset_names, output_dir / "optimized_weights.png")
    plot_return_distribution(daily_returns, output_dir / "daily_return_distribution.png")

    print(f"Price sample: {prices.index.min().date()} to {prices.index.max().date()}")
    print("\nAnnualized mean returns")
    print(mean_returns.apply(lambda x: f"{x:.2%}").to_string())
    print("\nAnnualized covariance matrix")
    print(cov_matrix.to_string(float_format=lambda x: f"{x: .6f}"))

    for label, portfolio in points.items():
        print_portfolio(label, portfolio, asset_names)

    print("\nFiles written to outputs/")


if __name__ == "__main__":
    main()
