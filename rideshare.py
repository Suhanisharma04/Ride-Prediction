import pandas as pd
import matplotlib.pyplot as plt
import pickle
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("TaxiRideShare .csv", low_memory=False)

#cleaning price
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df[df['price'].notna()]


df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

# Removing outliers
df = df[df['price'] < df['price'].quantile(0.99)]


print("Mean :", df['price'].mean())
print("Median :", df['price'].median())
print("Mode :", df['price'].mode().iloc[0])


plt.figure(figsize=(8, 4))
plt.hist(df['price'].values, bins=20, edgecolor='black')
plt.title("Distribution of Ride Prices")
plt.xlabel("Price ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show(block=True)


hourly = df.groupby('hour')['price'].mean()
plt.figure(figsize=(8, 4))
plt.plot(hourly.index, hourly.values, marker='o', linestyle='-')
plt.title("Average Ride Price by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Price ($)")
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.show(block=True)


corrs = df.corr(numeric_only=True)['price'].sort_values(ascending=False)
print("\nTop correlations with price:")
print(corrs.head(10))

core_cols = ['hour', 'distance', 'visibility', 'surge_multiplier',
             'cab_type', 'name', 'source', 'price']
df = df[core_cols].copy()


df['surge_multiplier'] = df['surge_multiplier'].fillna(1.0)
df['visibility'] = df['visibility'].fillna(df['visibility'].median())


df = pd.get_dummies(df, columns=['cab_type', 'name', 'source'], drop_first=True)

print(df.head())
print("Total rows:", df.shape[0])
print("Total features:", df.shape[1])

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\nModel evaluation:")
print("  Mean Squared Error :", mean_squared_error(y_test, y_pred))
print("  R²                 :", r2_score(y_test, y_pred))

pickle.dump({"model": model, "columns": X.columns.tolist()},
            open("ride_price_model.pkl", "wb"))


package = pickle.load(open("ride_price_model.pkl", "rb"))
model = package["model"]
model_columns = package["columns"]


def predict_price(hour, distance, visibility, surge_multiplier, cab_company, ride_type, source_area):

    input_data = {
        'hour': hour,
        'distance': distance,
        'visibility': visibility,
        'surge_multiplier': surge_multiplier
    }

    for col in model_columns:
        if col.startswith('cab_type_'):
            input_data[col] = 1 if f"cab_type_{cab_company}" == col else 0
        elif col.startswith('name_'):
            input_data[col] = 1 if f"name_{ride_type}" == col else 0
        elif col.startswith('source_'):
            input_data[col] = 1 if f"source_{source_area}" == col else 0


    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0

    row = pd.DataFrame([input_data])[model_columns]
    return float(model.predict(row)[0])


iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(0, 23, step=1, label="Hour of Day"),
        gr.Number(label="Trip Distance (miles)"),
        gr.Number(label="Visibility (miles)"),
        gr.Number(label="Surge Multiplier", value=1.0),
        gr.Radio(["Uber", "Lyft"], label="Cab Company"),
        gr.Dropdown(["UberX", "UberPool", "UberBlack", "Lyft XL", "Lyft"], label="Ride Type"),
        gr.Dropdown(["Hayes Valley", "South of Market", "Financial District", "Mission", "Pacific Heights", "Other"], label="Source Area")
    ],
    outputs=gr.Number(label="Predicted Price ($)"),
    title="Ride-share Price Estimator",
    description="Enter trip details to estimate fare based on time, distance, weather, ride type, and more."
)

iface.launch()
