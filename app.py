from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging to display debug information
logging.basicConfig(level=logging.DEBUG)

# Load the dataset
hotels = pd.read_csv('C:/Users/athis/OneDrive/Desktop/travel_stay_finder/hotels.csv', encoding='ISO-8859-1')

# Initialize Flask app
app = Flask(__name__)

# Debugging function to print unique values in 'city' and 'state'
def debug_unique_values():
    logging.debug(f"Unique Cities: {hotels['city'].unique()}")
    logging.debug(f"Unique States: {hotels['state'].unique()}")

# Filter hotels by location (no star rating filtering)
def filter_hotels_by_location(location):
    # Trim spaces and convert both to lowercase to ensure case-insensitivity
    location = location.strip().lower()

    # Trim and convert city and state to lowercase for case-insensitive comparison
    hotels['city'] = hotels['city'].str.strip().str.lower()
    hotels['state'] = hotels['state'].str.strip().str.lower()

    # Print unique values in 'city' and 'state' for debugging
    debug_unique_values()

    # Filter by location (city/state)
    filtered_hotels = hotels[
        (hotels['city'].str.contains(location, case=False, na=False)) |
        (hotels['state'].str.contains(location, case=False, na=False))
    ]
    
    logging.debug(f"Filtered Hotels: {filtered_hotels.shape[0]}")  # Log the number of filtered hotels

    # Select specific columns to display
    filtered_hotels = filtered_hotels[['property_name', 'city', 'state', 'hotel_star_rating', 'hotel_description']]
    
    return filtered_hotels

# Recommendation model using content-based filtering
def recommend_hotels(location, top_n=5):
    # Combine city and state for recommendation purposes
    hotels['location'] = hotels['city'].fillna('') + ', ' + hotels['state'].fillna('')
    
    # Create a TF-IDF Vectorizer to transform the location data
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(hotels['location'])
    
    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get the index of the location the user searched for
    location_indices = hotels[hotels['location'].str.contains(location, case=False, na=False)].index.tolist()
    
    logging.debug(f"Location Indices: {location_indices}")  # Log indices of the matched locations
    
    # Get the top N most similar hotels based on the search location
    similar_hotels = []
    for index in location_indices:
        sim_scores = list(enumerate(cosine_sim[index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_similar = sim_scores[1:top_n+1]  # Skip the first one as it's the hotel itself
        
        for i in top_similar:
            hotel_index = i[0]
            similar_hotels.append(hotels.iloc[hotel_index])
    
    return pd.DataFrame(similar_hotels).drop_duplicates()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    location = request.form['location']
    
    # Filter hotels by location
    result = filter_hotels_by_location(location)
    
    # If no hotels found, return a message
    if result.empty:
        return render_template('results.html', tables=None)
    
    # Convert result into a list of dictionaries for easy display in template
    result_list = result.to_dict(orient='records')

    return render_template('results.html', tables=result_list)

@app.route('/filter_results', methods=['POST'])
def filter_results():
    # Get the star rating from the form
    star_rating = request.form.get('star_rating')

    # If there is a selected star rating
    if star_rating:
        # Filter the hotels that match the selected star rating
        filtered_hotels = [hotel for hotel in tables if hotel['hotel_star_rating'] == star_rating]
    else:
        # If no star rating is selected, show all hotels
        filtered_hotels = tables

    # Return the filtered hotels
    return render_template('results.html', tables=filtered_hotels)

if __name__ == '__main__':
    app.run(debug=True)
