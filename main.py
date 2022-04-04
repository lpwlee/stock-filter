import pandas as pd
import numpy
import matplotlib.pyplot as plt
import cufflinks as cf
from pandas_datareader import data
import futu as ft
import datetime, time
from IPython.display import clear_output

pd.set_option('display.max_column', 1000)
pd.set_option('display.max_rows', 20)

print ('Opening Futu connection')
quote_ctx = ft.OpenQuoteContext(host='127.0.0.1', port=11111)

market = ft.Market.HK

ret, basicinfo_STOCK = quote_ctx.get_stock_basicinfo(market, stock_type=ft.SecurityType.STOCK) # Stock
ret, basicinfo_BOND = quote_ctx.get_stock_basicinfo(market, stock_type=ft.SecurityType.BOND) # BOND
ret, basicinfo_IDX = quote_ctx.get_stock_basicinfo(market, stock_type=ft.SecurityType.IDX) #INDEX
ret, basicinfo_PLATE = quote_ctx.get_stock_basicinfo(market, stock_type=ft.SecurityType.PLATE) #PLATE
ret, basicinfo_FUTURE = quote_ctx.get_stock_basicinfo(market, stock_type=ft.SecurityType.FUTURE) #FUTURE
ret, basicinfo_ETF = quote_ctx.get_stock_basicinfo(market, stock_type=ft.SecurityType.ETF) #ETF
ret, basicinfo_WARRANT = quote_ctx.get_stock_basicinfo(market, stock_type=ft.SecurityType.WARRANT) #WARRANT
ret, basicinfo_BWRT = quote_ctx.get_stock_basicinfo(market, stock_type=ft.SecurityType.BWRT) #BASKET WARRANT


print(basicinfo_STOCK)

quote_ctx.stop()
quote_ctx.close()
