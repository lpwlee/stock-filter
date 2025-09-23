from futu import OpenQuoteContext, SubType, Session

# Set up the API connection
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# Define the stock ticker (e.g., Hong Kong stock)
stock_code = 'HK.00005'  # Example: Tencent Holdings Limited

# Subscribe to basic quote data
try:
    ret_sub, sub_data = quote_ctx.subscribe(
        code_list=[stock_code],                       # List of stock codes to subscribe to
        subtype_list=[SubType.QUOTE],                 # Subscription type for basic quotes
        is_first_push=True,                           # Get the first push of data
        subscribe_push=True,                          # Receive push updates
        is_detailed_orderbook=False,                  # Not subscribing to detailed order book
        extended_time=False,                          # Not including extended trading hours
        session=Session.NONE                          # No specific session, include all
    )
    
    if ret_sub != 0:
        print(f"Error subscribing to data: {sub_data}")
        raise Exception("Subscription failed.")

    # Retrieve stock data
    ret, data = quote_ctx.get_stock_quote(stock_code)
    if ret == 0:
        print(f"Stock data for {stock_code}:\n", data)
    else:
        print(f"Error fetching data: {data}")
finally:
    # Unsubscribe from the stock and close the connection
    quote_ctx.unsubscribe([stock_code], [SubType.QUOTE])  # Unsubscribe to clean up
    quote_ctx.close()