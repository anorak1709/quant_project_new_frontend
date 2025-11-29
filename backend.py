from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

app = Flask(__name__)
CORS(app)

def daily_returns(series):
    """Calculate daily returns from a price series"""
    return series.pct_change().dropna()

def cagr(series, periods_per_year=252):
    """Calculate Compound Annual Growth Rate"""
    r = daily_returns(series)
    total_ret = series.iloc[-1] / series.iloc[0] - 1
    years = len(series) / periods_per_year
    return (1 + total_ret)**(1/years) - 1

def annual_vol(series, periods_per_year=252):
    """Calculate annualized volatility"""
    r = daily_returns(series)
    return r.std() * np.sqrt(periods_per_year)

def sharpe(series, rf_annual=0.0, periods_per_year=252):
    """Calculate Sharpe ratio"""
    r = daily_returns(series)
    rf_daily = (1 + rf_annual)**(1/periods_per_year) - 1
    excess = r - rf_daily
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)

def max_drawdown(series):
    """Calculate maximum drawdown"""
    running_max = series.cummax()
    drawdown = series / running_max - 1
    return drawdown.min()

@app.route('/api/analyze', methods=['POST'])
def analyze_portfolio():
    try:
        data = request.json
        holdings = data.get('holdings', {})
        benchmark = data.get('benchmark', '^NSEI')
        start_date = data.get('start_date', '2018-01-01')
        risk_free_rate = data.get('risk_free_rate', 0.065)
        end_date = datetime.today().strftime("%Y-%m-%d")
        
        if not holdings:
            return jsonify({'error': 'No holdings provided'}), 400
        
        # Download price data
        tickers = list(holdings.keys()) + [benchmark]
        print(f"Downloading data for: {tickers}")
        
        px = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
        
        # Handle single ticker case
        if len(tickers) == 2:  # 1 stock + 1 benchmark
            px = pd.DataFrame(px)
            px.columns = [list(holdings.keys())[0]]
            
        # Clean data
        px = px.dropna(how="all").ffill()
        
        # Build portfolio value series
        portfolio_value = pd.Series(0.0, index=px.index)
        missing_tickers = []
        
        for ticker, qty in holdings.items():
            if ticker in px.columns:
                portfolio_value = portfolio_value.add(px[ticker] * qty, fill_value=0.0)
            else:
                missing_tickers.append(ticker)
        
        if missing_tickers:
            return jsonify({
                'error': f'Unable to fetch data for: {", ".join(missing_tickers)}. Please check ticker symbols.'
            }), 400
        
        # Get benchmark data
        if benchmark not in px.columns:
            # Try downloading benchmark separately
            benchmark_data = yf.download(benchmark, start=start_date, end=end_date, auto_adjust=True)["Close"]
            px[benchmark] = benchmark_data
        
        benchmark_price = px[benchmark].dropna()
        
        # Align dates
        df_compare = pd.DataFrame({
            "Portfolio": portfolio_value,
            "Benchmark": benchmark_price
        }).dropna()
        
        if len(df_compare) < 30:
            return jsonify({
                'error': 'Insufficient data points. Try a different start date or check if tickers are valid.'
            }), 400
        
        # Normalize for chart
        norm = df_compare / df_compare.iloc[0]
        
        # Calculate metrics
        portfolio_metrics = {
            'cagr': float(cagr(df_compare['Portfolio'])),
            'annual_vol': float(annual_vol(df_compare['Portfolio'])),
            'sharpe': float(sharpe(df_compare['Portfolio'], rf_annual=risk_free_rate)),
            'max_drawdown': float(max_drawdown(df_compare['Portfolio']))
        }
        
        benchmark_metrics = {
            'cagr': float(cagr(df_compare['Benchmark'])),
            'annual_vol': float(annual_vol(df_compare['Benchmark'])),
            'sharpe': float(sharpe(df_compare['Benchmark'], rf_annual=risk_free_rate)),
            'max_drawdown': float(max_drawdown(df_compare['Benchmark']))
        }
        
        # Prepare chart data (sample every 5 days to reduce data size)
        chart_data = []
        sampled = norm.iloc[::5]  # Sample every 5th row
        for idx, row in sampled.iterrows():
            chart_data.append({
                'date': idx.strftime('%Y-%m-%d'),
                'portfolio': round(row['Portfolio'], 4),
                'benchmark': round(row['Benchmark'], 4)
            })
        
        return jsonify({
            'portfolio': portfolio_metrics,
            'benchmark': benchmark_metrics,
            'chart_data': chart_data,
            'date_range': {
                'start': df_compare.index[0].strftime('%Y-%m-%d'),
                'end': df_compare.index[-1].strftime('%Y-%m-%d')
            }
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Portfolio Optimizer API is running'})

if __name__ == '__main__':
    print("Starting Portfolio Optimizer Backend...")
    print("Server running on http://localhost:5000")
    app.run(debug=True, port=5000)
