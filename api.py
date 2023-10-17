from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from matplotlib.pyplot import clf
from numpy import vectorize
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
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
categories = {
    'Food': ['restaurant', 'eatery', 'diner', 'bistro', 'brasserie', 'tavern', 'pizzeria', 'steakhouse', 'steak',
             'bar', 'sushi bar', 'burger', 'bbq', 'noodle', 'breakfast', 'break fast', 'lunch', 'diner',
             'dining', 'buffet', 'seafood', 'ramen', 'sweet', 'bakery', 'ice cream',
             'icecream', 'grill','hunger','hungry','sandwich','donut','donuts','sandwiches',
             'doughnuts','hungerstation','hungerstat','mcdonald','mcdonalds','kfc',
             'dunkin','krispy kreme','baik','albaik','tazaj','fakeih',
             'shawarma','shawerma','dominos','burgerizzr'],
    'Coffee': ['cafee', 'coffee', 'espresso', 'bistro', 'teahouse', 'café', 'riastery', 'coffeehouse',
               'caf', 'bean', 'beans', 'caffe', 'cafe', 'roast', 'roastrie', "barn's",
               'barns', 'barnscafe', 'barncafe', 'starbucks', 'bon', 'dose',
               'overdose', "joffrey's"
              ],
    'Grocery': ['supermarket', 	'super market'	,'market'	,'grocery store'	,'hypermarket'	,'hyper'	,'superstore'	,
               	'fruits'	,'fruit'	,'foodstuf'	,'foodstuff'	,'danube'	,'lulu'	,'panda'	,
               	'carrefour'	,'manuel'	,'markets'	,'othaim'	,'tamimi'	,
               	'retail','bindawood'	,'meed'],
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
    'Shopping': ['retail', 	'mall' , 	'shop' , 	'shopping' , 	'boutique' , 	'outlet' , 	'plaza' , 	'center',
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
            new_instance_counts = add_dummy_feature(new_instance_counts, word_feature)

    new_prediction = loaded_model.predict(new_instance_counts)
    confidence_levels = loaded_model.predict_proba(new_instance_counts)
    predicted_category = new_prediction[0]
    print("The predicted category is:", predicted_category)

    predicted_category_index = list(loaded_model.classes_).index(predicted_category)

    confidence_for_predicted_category = confidence_levels[0][predicted_category_index]
    print(f"Confidence for {predicted_category}: {confidence_for_predicted_category * 100}")
    return predicted_category
