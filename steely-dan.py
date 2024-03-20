from prophet import Prophet
import pandas as pd

cruFile = "CRU Weekly Historical Data.xlsx"
cruDateColumn = "DATE"
prophetDateColumn = "ds"
prophetDataPointColumn = "y"
weeksToPredict = 52 // 2
weeksToShow = 52 // 2 + 8
weeklyFrequency = "W"
columnsToPredict = ["HR / ton", "CR / ton"]
columnsToShow = ["ds", "trend", "yhat", "weekly"]

def get_data(filename):
    print("Getting data")

    data = pd.ExcelFile(filename)

    sheets = pd.read_excel(data, skiprows = 1)

    return sheets

def preprocess_data(data, cruDataPointColumn):
    print("Preprocessing data")

    timeAndData = data[[cruDateColumn, cruDataPointColumn]]

    renameColumns = {cruDateColumn: prophetDateColumn, cruDataPointColumn: prophetDataPointColumn}

    prophetForm = timeAndData.rename(columns=renameColumns)

    return prophetForm

def get_covid_event():
    covid_spans = pd.DataFrame([
        {"holiday": "lockdown", "ds": "2020-03-21", "lower_window": 0, "ds_upper": "2022-04-22"}
    ])

    for t_col in ["ds", "ds_upper"]:
        covid_spans[t_col] = pd.to_datetime(covid_spans[t_col])

    covid_spans["upper_window"] = (covid_spans["ds_upper"] - covid_spans["ds"]).dt.days

    return covid_spans

def fit(data):
    print("Fitting data")

    m = Prophet(yearly_seasonality=20, holidays=get_covid_event())
    m.fit(data)

    return m

def predict_data(m, data, periodsToPredict, frequency):
    print("Predicting data")

    future = m.make_future_dataframe(periods=periodsToPredict, freq=frequency)

    forecast = m.predict(future)

    return forecast

def print_prediction(column, prediction, number):
    print(prediction.columns.tolist())
    part = prediction[columnsToShow].tail(number)

    print("Prediction for \"" + column + "\":")

    print(part)

data = get_data(cruFile)

for column in columnsToPredict:
    preprocessedData = preprocess_data(data, column)

    m = fit(preprocessedData)

    prediction = predict_data(m, preprocessedData, weeksToPredict, weeklyFrequency)

    print_prediction(column, prediction, weeksToShow)
