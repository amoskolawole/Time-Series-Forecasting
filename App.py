import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib 

model = joblib.load("forecast_model.pkl")
sales_df = pd.read_csv("SalesData.csv")

sales_df["Date"] = pd.to_datetime(sales_df["Date"])
X = sales_df[["Lag_1", "Lag_2", "Lag_4", "Week"]]

#App Title
st.title("Walmart Sales Forecasting App")
#App Description
st.write(
    """
    This project forecasts future Walmart sales using
    Random Forest Regression and time-series lag features.
    """)

#Historical Sales Chat
fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    sales_df["Date"],
    sales_df["Weekly_Sales"]
)

ax.set_title("Historical Walmart Sales")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")

st.pyplot(fig)


#History Creation
history = list(sales_df["Weekly_Sales"].tail(4))

#Get Current Time Info
current_week = sales_df["Week"].iloc[-1]
                            #current_year = totalsales["Year"].iloc[-1]

future_pred = []

#Future Prediction Loop
for i in range(4):

    future_data = pd.DataFrame({
        "Lag_1": [history[-1]],
        "Lag_2": [history[-2]],
        "Lag_4": [history[-4]],
        #"Month": [1],   # temporary

        "Week": [current_week + i + 1],

    })

    pred = model.predict(future_data)[0]

    future_pred.append(pred)

    history.append(pred)

#Create Future Dates
future_dates = pd.date_range(
    start=sales_df["Date"].iloc[-1],
    periods=5,
    freq="W"
)[1:]

#Future Forecast Chart
st.subheader("Next 4 Weeks Sales Forecast")

fig2, ax2 = plt.subplots(figsize=(12,5))

# historical
ax2.plot(
    sales_df["Date"],
    sales_df["Weekly_Sales"],
    label="Historical Sales"
)

# forecast
ax2.plot(
    future_dates,
    future_pred,
    marker="o",
    linestyle="--",
    label="Forecast"
)

ax2.set_title("Future Sales Forecast")
ax2.set_xlabel("Date")
ax2.set_ylabel("Sales")

ax2.legend()

st.pyplot(fig2)

#Forecast Table
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": future_pred
})
st.subheader("Forecast Table")

st.dataframe(forecast_df)

#Feature Importance DataFrame Creation
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)
st.subheader("Feature Importance")

#Plot Importance
fig3, ax3 = plt.subplots(figsize=(10,5))

ax3.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax3.set_title("Feature Importance")

plt.xticks(rotation=45)

st.pyplot(fig3)

st.sidebar.title("About")
st.sidebar.write(
    """
    Walmart sales forecasting project using
    Random Forest Regression and lag features.
    """
)
st.sidebar.write("Model Performance")
st.sidebar.write("R² Score: 0.36")
