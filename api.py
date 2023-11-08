from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from matplotlib.pyplot import clf
from numpy import vectorize
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn import *
from sklearn.preprocessing import add_dummy_feature
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import schema
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],)
categories = {
    'Food': ['restaurant', 'eatery', 'diner', 'bistro', 'brasserie', 'tavern', 'pizzeria', 'steakhouse', 'steak',
             'bar', 'sushi bar', 'burger', 'bbq', 'noodle', 'breakfast', 'break fast', 'lunch', 'diner',
             'dining', 'buffet', 'seafood', 'ramen', 'sweet', 'bakery', 'ice cream',
             'icecream', 'grill', 'hunger', 'hungry', 'sandwich', 'donut', 'donuts', 'sandwiches',
             'doughnuts', 'hungerstation', 'hungerstat', 'mcdonald', 'mcdonalds', 'kfc',
             'dunkin', 'krispy kreme', 'baik', 'albaik', 'tazaj', 'fakeih',
             'shawarma', 'shawerma', 'dominos', 'burgerizzr'],
    'Coffee': ['cafee', 'coffee', 'espresso', 'bistro', 'teahouse', 'café', 'riastery', 'coffeehouse',
               'caf', 'bean', 'beans', 'caffe', 'cafe', 'roast', 'roastrie', "barn's",
               'barns', 'barnscafe', 'barncafe', 'starbucks', 'bon', 'dose',
               'overdose', "joffrey's"
               ],
    'Grocery': ['supermarket', 	'super market', 'market', 'grocery store', 'hypermarket', 'hyper', 'superstore',
                'fruits', 'fruit', 'foodstuf', 'foodstuff', 'danube', 'lulu', 'panda',
                'carrefour', 'manuel', 'markets', 'othaim', 'tamimi',
                'retail', 'bindawood', 'meed'],
    'Health': ['hospital', 	'clinic', 	'medical', 	'health', 	'care', 	'pharmacy', 	'doctor', 	'dentist',
               'emergency', 	'er', 	'therapy', 	'care center', 	'spa', 	'beauty', 	'salon',
               'hair', 	'nail', 	'barber', 	'massage',
               'wax', 	'waxing',
               'body',
               'make up',
               'makeup',
               'laser hair',
               'fitness',
               'fit',
               'yoga',
               'weight',
               'health',
               'workout',
               'training',
               'gym',
               'crossfit',
               'cardio',
               "nahdi",
               "alnahdi",
               "whites",
               "al nahdi",
               "barbershop"],
    'Transportation': ['gas', 	'petrol', 	'station', 	'gasstation',
                       'fuel', 	'refueling',
                       'filling station',
                       'diesel',
                       'pump',
                       'naft',
                       'motor',
                       'vehicle',
                       "parkin",
                       "parking",
                       "air",
                       "airline",
                       "airlines",
                       "aldrees",
                       "petromin",
                       "sahel",
                       "al-adel",
                       "petrosven",
                       "quraish",
                       "al musaidya",
                       "al-andalus",
                       "andalus",
                       "sasco"],
    'Shopping': ['retail', 	'mall', 	'shop', 	'shopping', 	'boutique', 	'outlet', 	'plaza', 	'center',
                 'retailer', 	'fashion', 	'clothing', 	'shoe', 	'gift', 	'art',
                 'jewelry', 	'clothes', 	'storefront',
                 'book', 	'books',
                 'store',
                 'bookstore',
                 'sport',
                 'sports',
                 "cente",
                 "zara",
                 "jarir",
                 "prada",
                 "gucci",
                 "next",
                 "redtag",
                 "boulevard",
                 "thobe",
                 "lomar",
                 "sindi",
                 "h&m",
                 "burberry",
                 "jewellery",
                 "tiffany",
                 "jewelry",
                 "cartier",
                 "rolex",
                 "ikea",
                 "extra",
                 "saco"]
}


@app.post("/classifyTransaction")
async def classify_transaction(transaction: schema.Transaction):
    loaded_model = joblib.load('./model.joblib')
    loaded_vectorizer = joblib.load('./vectorizer.joblib')

    new_instance = [transaction.message]

    new_instance_counts = loaded_vectorizer.transform(new_instance)

    for category, words in categories.items():
        for word in words:
            word_feature = [new_instance[0].lower().count(word)]
            new_instance_counts = add_dummy_feature(
                new_instance_counts, word_feature)

    new_prediction = loaded_model.predict(new_instance_counts)
    confidence_levels = loaded_model.predict_proba(new_instance_counts)
    predicted_category = new_prediction[0]

    predicted_category_index = list(
        loaded_model.classes_).index(predicted_category)

    confidence_for_predicted_category = confidence_levels[0][predicted_category_index]
    result = {
        "category": predicted_category,
        "probability": round((confidence_for_predicted_category * 100), 2)
    }

    return result


@app.post("/forecastSpendings")
async def forecast_spendings(spendings: schema.Spendings):
    try:
        print(spendings)
        monthlySpending = daily_to_monthly(spendings.spendings)
        df_no_outliers = monthlySpending.copy()
        for category in monthlySpending.columns:
            df_no_outliers = remove_outliers(df_no_outliers, category)

        forecasted_next_month = {}

        for category in df_no_outliers.columns:
            forecasted_value = forecast_arima_with_differencing(
                df_no_outliers[category])
            forecasted_next_month[category] = forecasted_value

        results = {
            'Category': list(df_no_outliers.columns),
            'Forecast': [forecasted_next_month[category] for category in df_no_outliers.columns]
        }

        results_df = pd.DataFrame(results)
        results_df.set_index('Category', inplace=True)
        return results_df
    except Exception as e:
        print(e)
        return {}


def daily_to_monthly(spendings):
    df = pd.DataFrame(spendings)

    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

    # Set 'date' to the last day of each month
    df['date'] = df['date'].dt.to_period('M').dt.to_timestamp('M')

    # Convert the timestamp to a string without the time component
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Pivot the table to get categories as columns and sum the amounts
    df_pivot = df.pivot_table(index='date', columns='category',
                              values='amount', aggfunc='sum').fillna(0.0)

    # Ensure all desired categories are present
    desired_categories = ["Food", "Coffee", "Transit",
                          "Health", "Grocery", "Shopping", "Bills"]
    for category in desired_categories:
        if category not in df_pivot.columns:
            df_pivot[category] = 0.0

    # Reorder columns and sort by date
    df_pivot = df_pivot[desired_categories]
    df_pivot.sort_values(by='date', inplace=True)
    return df_pivot


def forecast_arima_with_differencing(series, order=(1,1,0), forecast_steps=1):
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_steps)
        return forecast[0]


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df
