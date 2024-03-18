from prophet import Prophet
import yfinance as yf

ticker = "HRC=F"
daysToPredict = 14
daysToShow = 25

def get_data(ticker):
    print("Getting data")

    # data = yf.download(ticker, start=start_date, end=end_date)
    data = yf.download(ticker)

    return data

def preprocess_data(data):
    print("Preprocessing data")

    data.reset_index(inplace=True)
    closeData = data[["Date", "Close"]]

    closeData.reset_index(inplace=True)

    close = closeData.rename(columns={"Date": "ds", "Close": "y"})
    return close

def fit(data):
    print("Fitting data")

    m = Prophet()
    m.fit(data)

    return m

def predict_data(m, data, periodsToPredict):
    print("Predicting data")

    future = m.make_future_dataframe(periods = periodsToPredict)

    forecast = m.predict(future)

    return forecast

def print_prediction(prediction, number):
    part = prediction[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(number)

    print(part)

data = get_data(ticker)

preprocessedData = preprocess_data(data)

m = fit(preprocessedData)

prediction = predict_data(m, preprocessedData, daysToPredict)

print_prediction(prediction, daysToShow)
