import yfinance as yf

stock = yf.Ticker("AAPL")

hist = stock.history(period="1y") 
print(hist.head())

print(stock.info)  

print(stock.financials)
print(stock.balance_sheet)
print(stock.cashflow)
