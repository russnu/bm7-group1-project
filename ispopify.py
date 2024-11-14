#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
import io

from PIL import Image

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report

# Importing Models
import joblib
popularity_model = joblib.load('assets/models/popularity_model.joblib')
genre_model = joblib.load('assets/models/genre_model.joblib')
#######################
# Page configuration
st.set_page_config(
    page_title="IsPopify",
    page_icon="assets/spotify_icon.png",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
def features_heatmap(features_correlation, width, height, key, heatmap_title):
    
    heatmap_fig = px.imshow(
        features_correlation,
        text_auto=True,
        aspect="equal",
        zmin=-1,
        zmax=1,
        title=heatmap_title,
        color_continuous_scale = ["#6A0AD4", "#ffffff",  "#80ad00"],
        color_continuous_midpoint = 0
    )
    heatmap_fig.update_layout(
        width=width,
        height=height,
        xaxis=dict(tickangle=45),
        margin=dict(b=120),
        title=dict(font=dict(size=24, weight='bold'), xanchor='left', x=0)
    )
    with st.container():
        st.plotly_chart(heatmap_fig, theme=None, use_container_width=True, key=f"heatmap_{key}")
     #------------------------------------------------------------------------------------------#
def scatter_plot(df, column, width, height, key, scatterplot_title):
    scatter_plot = px.scatter(df, 
                              x=df[column], 
                              y=df['track_popularity'],
                              trendline="ols", 
                              color_discrete_sequence=['#80ad00'],
                              title=scatterplot_title)
    scatter_plot.update_traces(line=dict(color='#6A0AD4'), selector=dict(mode='lines'))
    scatter_plot.update_layout(
        width=width,
        height=height,
        title=dict(font=dict(size=24))
    )
    st.plotly_chart(scatter_plot, use_container_width=True, key=f"scatter_plot_{key}")
#------------------------------------------------------------------------------------------#   
def feature_importance_plot(feature_importance_df, width, height, key):
    feature_importance_fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        orientation='h',
        color_discrete_sequence=['#80ad00']
    )
    feature_importance_fig.update_layout(
        width=width,
        height=height
    )
    st.plotly_chart(feature_importance_fig, use_container_width=True, key=f"feature_importance_plot_{key}")
#------------------------------------------------------------------------------------------#
def genres_by_track_decade_plots(data, width, height):
    genres = data['playlist_genre'].unique()
    fig = make_subplots(rows=3, cols=2, subplot_titles=[f"Distribution of {genre.title()} Genre by Decade" for genre in genres])
    for i, genre in enumerate(genres):
        genre_data = data[data['playlist_genre'] == genre]
        row = (i // 2) + 1
        col = (i % 2) + 1
        fig.add_trace(
            go.Histogram(x=genre_data['track_decade'], name=genre.title(), opacity=0.6),
            row=row, col=col
        )
    fig.update_layout(width=width,
                      height=height,
                      title=dict(font=dict(size=24)), 
                      title_text="Distribution of Genres by Decade", showlegend=False)
    fig.update_xaxes(title_text="Track Decade", dtick=10)
    fig.update_yaxes(title_text="Count")
    st.plotly_chart(fig, use_container_width=True)
#========================================================================================================================#
# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page
  
# Custom sidebar title styling 
st.markdown(
    """<style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,800;1,800&display=swap');
        .custom-title {
            font-family: 'Montserrat', 'Arial';
            font-size: 24px;
            color: #80ad00;
        }
       </style>""", unsafe_allow_html=True
)
#========================================================================================================================#
# Sidebar
with st.sidebar:

    col = st.columns([1, 3], vertical_alignment="center")
    with col[0]:
        st.image("assets/spotify_icon.png", use_column_width=True)
    
    with col[1]:
         st.markdown('<div class="custom-title">IsPopify</div>', unsafe_allow_html=True)


    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. John Russel Segador\n2. Kurt Dhaniel Tejada\n3. John Schlieden Agor")

#========================================================================================================================#
# Load data
dataset = pd.read_csv("data/spotify_songs.csv")
#========================================================================================================================#
def preprocessed_dataset():
    global dataset
    dataset_copy = dataset.copy(deep=True)
    dataset_copy = dataset_copy.dropna()
    #-------------------------------------------------------------------------------------------#
    def refineDates(release_date):
        if (len(release_date) == 10):
            return release_date
        elif (len(release_date) == 4):
            return release_date + "-01-01"
        else:
            return release_date + "-01"
    dataset_copy['track_album_release_date'] = dataset_copy['track_album_release_date'].apply(refineDates)
    dataset_copy['track_album_release_date'] = pd.to_datetime(dataset_copy['track_album_release_date'])
    dataset_copy['track_year'] = dataset_copy['track_album_release_date'].dt.year
    dataset_copy['track_month'] = dataset_copy['track_album_release_date'].dt.month
    dataset_copy['track_decade'] = (dataset_copy['track_year'] // 10) * 10
    dataset_copy['track_age'] = dataset_copy['track_year'].apply(lambda x : 2024 - x)
    #-------------------------------------------------------------------------------------------#
    le_genre = LabelEncoder()
    dataset_copy['playlist_genre_encoded'] = le_genre.fit_transform(dataset_copy['playlist_genre'])
    le_subgenre = LabelEncoder()
    dataset_copy['playlist_subgenre_encoded'] = le_subgenre.fit_transform(dataset_copy['playlist_subgenre'])
    #-------------------------------------------------------------------------------------------#
    # Calculate thresholds
    low_threshold = dataset_copy['track_popularity'].quantile(0.33)
    high_threshold = dataset_copy['track_popularity'].quantile(0.66)

    # Define the categorization function
    def categorize_popularity(popularity, low_threshold, high_threshold):
        if popularity < low_threshold:
            return 'low_popularity'
        elif popularity < high_threshold:
            return 'medium_popularity'
        else:
            return 'high_popularity'

    # Apply the function
    dataset_copy['popularity_level'] = dataset_copy['track_popularity'].apply(categorize_popularity, args=(low_threshold, high_threshold))
    
    oe_popularity = OrdinalEncoder(categories=[['low_popularity', 'medium_popularity', 'high_popularity']])
    dataset_copy['popularity_level_encoded'] = oe_popularity.fit_transform(dataset_copy[['popularity_level']])
    #-------------------------------------------------------------------------------------------#
    artist_popularity = dataset_copy.groupby('track_artist')['track_popularity'].mean().reset_index()
    artist_popularity.columns = ['track_artist', 'artist_popularity']
    dataset_copy = pd.merge(dataset_copy, artist_popularity, on='track_artist', how='left')
    #-------------------------------------------------------------------------------------------#
    return dataset_copy, le_genre
#--------------------------------------------------------------------------------------------------------------------------#
preprocessed_data, le_genre = preprocessed_dataset()
#--------------------------------------------------------------------------------------------------------------------------#
def popularity_train_test():
    global preprocessed_data
    data_filtered = preprocessed_data.copy(deep=True)
    object_columns = data_filtered.select_dtypes(include=['object']).columns
    columns_to_drop = list(object_columns) + ['track_album_release_date', 'track_decade']
    data_filtered = data_filtered.drop(columns=columns_to_drop)
    
    popularity_model_df = data_filtered

    X = popularity_model_df.drop(columns=['track_popularity', 'popularity_level_encoded'])
    y = popularity_model_df['popularity_level_encoded']
    
    popularity_X_train, popularity_X_test, popularity_y_train, popularity_y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    
    return X, popularity_X_train, popularity_X_test, popularity_y_train, popularity_y_test
#--------------------------------------------------------------------------------------------------------------------------#   
def genre_train_test():
    global preprocessed_data
    
    data_filtered = preprocessed_data.copy(deep=True)
    object_columns = data_filtered.select_dtypes(include=['object']).columns
    columns_to_drop = list(object_columns) + ['track_album_release_date', 'popularity_level_encoded']
    data_filtered = data_filtered.drop(columns=columns_to_drop)
    
    genre_model_df = data_filtered
    X = genre_model_df.drop(columns=['playlist_genre_encoded', 
                                    'playlist_subgenre_encoded'])
    y = genre_model_df['playlist_genre_encoded']
    
    genre_X_train, genre_X_test, genre_y_train, genre_y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    
    return X, genre_X_train, genre_X_test, genre_y_train, genre_y_test
#--------------------------------------------------------------------------------------------------------------------------#   
popularity_model_df, popularity_X_train, popularity_X_test, popularity_y_train, popularity_y_test = popularity_train_test()
genre_model_df, genre_X_train, genre_X_test, genre_y_train, genre_y_test = genre_train_test()
#========================================================================================================================#
# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")
    # Your content for the ABOUT page goes here
    st.markdown("""
    A Streamlit web application that utilize **Exploratory Data Analysis**, **Data Preprocessing**, and **Machine Learning** to analyze and classify songs from the Spotify API with the 30000 Spotify Songs dataset. Model uses **Random Forest Classifier** to train **Popularity Level Classification** and **Genre Classification**.

    ### Pages:
    1. **Dataset** - The dataset contains around 30,000 Songs from the Spotify API. The data about the songs include song id, song name, song artist, track popularity, information about the album, and information about the genre.
                     Also, the dataset includes audio features related to the songs' characteristics, like danceability, energy, loudness, tempo, acousticness, among others.
    2. **EDA** - Exploratory Data Analysis of 30000 Spotify Songs. Contains the distribution and relationships of different data into various graphs.
    3. **Data Cleaning / Pre-processing** - Data cleaning and pre-processing steps this includes Dropping Null Values, Adding Track Age and Track Decade Column, Encoding Genre and Subgenre, Categorizing Popularity, and Adding Artist Popularity Column.
    4. **Machine Learning** - Using **Random Forest Classifier** to train **Popularity Level Classification**, and **Genre Classification**. This page also includes each model's model evaluation, and feature importance.
    5. **Prediction** - Prediction page that makes use of the models to predict the song's popularity level and genre.
    6. **Conclusion** - Summary of the analysis and findings from the exploratory data analysis and model training processes.
    """)





# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.markdown("""

   The dataset contains almost 30,000 Songs from the Spotify API. The data about the songs include song id, song name, song artist, track popularity, 
   information about the album, and information about the genre. Also, the dataset includes audio features related to the songs' characteristics, like danceability, energy, loudness, tempo, acousticness,
   among others[1].

   **Content**
   The table displays the columns of the dataset along with their respective data types and descriptions. 

   'Link:' https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs


    """)


    # Display the dataset
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(pd.read_csv("data/spotify_songs.csv"), use_container_width=True, hide_index=True)

    # Describe Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(pd.read_csv("data/spotify_songs.csv").describe(), use_container_width=True)


    st.markdown("""

    The results from `df.describe()` highlights the descriptive statistics about the dataset. 
    
    The average track popularity is 42.47, with popularity scores spanning from  to 100. 
    
    Audio features like danceability and energy have high averages, with 0.65 and 0.69, respectively. 
    
    Loudness ranges significantly from -46.4 dB to 1.27 dB. Track durations vary widely, 
    
    averaging around 225,800 milliseconds (about 3.8 minutes) but ranging from 4 seconds to over 8.5 minutes
                
    """)

                
    """)

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    col = st.columns((3, 3, 3), gap='medium')

    with col[0]:

        with st.expander('Legend', expanded=True):
            st.write('''
                - Data: [30000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs).
                - :orange[**Pie Chart**]: Distribution of the Music Genre in the dataset.
                - :orange[**Scatter Plots**]: Difference of Iris species' features.
                - :orange[**Pairwise Scatter Plot Matrix**]: Highlighting *overlaps* and *differences* among Iris species' features.
                ''')


     
#======================================================================================================================================#
# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")
    # Your content for the DATA CLEANING / PREPROCESSING page goes here
    
    st.markdown("### Dropping Null Values")
    col = st.columns([1,3])
    with col[0]:
        st.markdown("*Count of null values*")
        st.dataframe(dataset.isna().sum())
    with col[1]: 
        st.markdown("*Null Values*") 
        null_columns = dataset[['track_id'] + dataset.columns[dataset.isna().any()].tolist()]
        null_values = null_columns[null_columns.isna().any(axis=1)]
        st.dataframe(null_values)   
        
        st.markdown("*Drop rows with missing values*")
        st.code("""
                data=data.dropna()
        """)
        
    st.markdown("""
                We can see that the columns `track_name`, `track_artist`, and `track_album_name` each contain ***5 null values***. 
                Notably, all of these null values originate from the same 5 rows in the dataset. Since the count of values is
                only 5, we can use `dropna()` to remove the rows with missing values. The count of the removed rows will not
                have a significant impact on the dataset size (more than 30,000) and is unlikely to skew the analysis or model results.
    """)
    st.markdown("---")
    st.markdown("### Converting Dates and Adding Track Age and Track Decade Columns")
    col = st.columns(3)
    with col[0]:
        st.markdown("*Dates with year only*")
        onlyYears = dataset['track_album_release_date'][dataset['track_album_release_date'].str.len()==4]
        onlyYears
        st.markdown(f"*Count: `{onlyYears.shape[0]}`*")
    with col[1]:
        st.markdown("*Dates with month and year*")
        monthYears = dataset['track_album_release_date'][dataset['track_album_release_date'].str.len() == 7]
        monthYears
        st.markdown(f"*Count: `{monthYears.shape[0]}`*")
    with col[2]:
        st.markdown("*Dates with day, month, and year*")
        completeDates = dataset['track_album_release_date'][dataset['track_album_release_date'].str.len()==10]
        completeDates
        st.markdown(f"*Count: `{completeDates.shape[0]}`*")
        
    st.markdown("""
                The `track_album_release_date` column of the dataset is not properly formatted, with ***1855*** 
                dates having only years and ***31*** dates having only months and years.
                """)
    
    st.code("""
            # Function to handle date anomalies
            def refineDates(release_date):
                if (len(release_date) == 10):
                    return release_date
                elif (len(release_date) == 4):
                    return release_date + "-01-01"
                else:
                    return release_date + "-01"
                    
            # Apply the function
            data['track_album_release_date'] = data['track_album_release_date'].apply(refineDates)
            """)
    st.markdown("""
                To address the improperly formatted dates, we implemented a function `refineDates` that takes a date 
                string as input and checks its length to determine how to format it correctly. If the date has 10 characters, 
                we assume it is already in the proper format *(YYYY-MM-DD)* and return it as is. If the date is only 4 
                characters long (year), we append "**-01-01**"  set it as January 1st of that year. Else if the date string is 
                neither 10 nor 4 characters, we assume it has month and year *(YYYY-MM)* we append "**-01**" to set it to the first 
                day of the month. We then apply this function to the track_album_release_date column of the dataset.
                """)
    
    st.code("""
            # Convert to datetime format and extract year and month
            data['track_album_release_date'] = pd.to_datetime(data['track_album_release_date'])
            data['track_year'] = data['track_album_release_date'].dt.year
            data['track_month'] = data['track_album_release_date'].dt.month
            
            # Add track age and track decade column
            data['track_age'] = data['track_year'].apply(lambda x : 2024 - x)
            data['track_decade'] = (data['track_year'] // 10) * 10
            """)
    st.markdown("""
                We converted the `track_album_release_date` column to datetime format then extracted the year and month into new 
                columns, `track_year` and `track_month`. Additionally, we calculated the age of each and the decade they were released,
                storing the result in the `track_age` and `track_decade` column.
                """)
    st.markdown("---")
    st.markdown("### Converting Dates and Adding Track Age and Track Decade Columns")
    st.code("""
            # Encode genre and subgenre
            le_genre = LabelEncoder()
            data['playlist_genre_encoded'] = le_genre.fit_transform(data['playlist_genre'])
            le_subgenre = LabelEncoder()
            data['playlist_subgenre_encoded'] = le_subgenre.fit_transform(data['playlist_subgenre'])
            """)
    st.markdown("""
                We used `LabelEncoder()` to convert the `playlist_genre` and `playlist_subgenre` columns from categorical 
                text values into numerical labels. By using `fit_transform()`, we assign a unique integer to each genre and subgenre,
                resulting in a new column, `playlist_genre_encoded` and `playlist_subgenre_encoded`, which contains these numeric representations.
                """)
    st.markdown("---")
    st.markdown("### Categorizing Popularity")
    
    st.code("""
            # Calculate thresholds
            low_threshold = data['track_popularity'].quantile(0.33)
            high_threshold = data['track_popularity'].quantile(0.66)

            # Define the categorization function
            def categorize_popularity(popularity, low_threshold, high_threshold):
                if popularity < low_threshold:
                    return 'low_popularity'
                elif popularity < high_threshold:
                    return 'medium_popularity'
                else:
                    return 'high_popularity'

            # Apply the function
            data['popularity_level'] = data['track_popularity'].apply(categorize_popularity, args=(low_threshold, high_threshold))
            
            # Encode track popularity levels
            oe_popularity = OrdinalEncoder(categories=[['low_popularity', 'medium_popularity', 'high_popularity']])
            data['popularity_level_encoded'] = oe_popularity.fit_transform(data[['popularity_level']])
            """)
    st.markdown("""
                We categorized the `track_popularity` in to low, medium, and high by calculating thresholds based on the `0.33 and 0.66 quantiles`.
                We then converted the categorical values of track popularity levels (low, medium, high) into numerical values using `OrdinalEncoder()`.
                """)
    st.markdown("---")
    
    st.markdown("### Adding Artist Popularity Column")
    
    st.code("""
            # Calculate artist popularity
            artist_popularity = data.groupby('track_artist')['track_popularity'].mean().reset_index()
            artist_popularity.columns = ['track_artist', 'artist_popularity']

            data = pd.merge(data, artist_popularity, on='track_artist', how='left')
            """)
    st.markdown("""
               To enhance the model, we calculated average popularity for artists by grouping the data by *`track_artist`* and computed the mean *`track_popularity`*,
               then named the resulting column as **`artist_popularity`**. After calculating the average popularity for artists, we merged the new columns into the 
               original dataset. We used the `pd.merge()` function and aligned the new column based on the `track_artist` key.
                """)
   
    #-------------------------------------------------------------------------------------------#
    heatmap_graph_df = preprocessed_data.copy()
    heatmap_graph_df = heatmap_graph_df.drop(columns=['key', 'mode', 'speechiness', 'tempo', 'popularity_level_encoded'])
    object_columns = heatmap_graph_df.select_dtypes(include=['object']).columns
    heatmap_graph_df = heatmap_graph_df.drop(columns=object_columns)
    features_correlation = heatmap_graph_df.corr().round(2)
    #-------------------------------------------------------------------------------------------#
    heatmap_title = "Audio Features and Interactions Correlation Matrix"
    features_heatmap(features_correlation, 800, 800, 1, heatmap_title)
    st.markdown("""
                The heatmap shows the correlation of various audio features and also the track popularity as the previous heatmap, but this time removing columns 
                that may not be strong predictors and adding the new columns `artist popularity`, `track month`, `track year`, `track decade`, and `track age`. 
                Notably, `artist popularity` has a strong positive correlation of ***0.74*** with track popularity, suggesting that songs by more popular artists 
                tend to be more well-received. This indicates that the popularity of the artist is an important factor that could enhance the perfomance of the model. 
                We can also see that year of release of the tracks show some correlation with genre, as `track_year` and `track_decade` has a negative correlation of 
                ***-0.44*** and ***-0.43***, respectively. `track_age`, however, has a positive correlation of ***0.44*** with genre.
                """)
    #-------------------------------------------------------------------------------------------#
    scatterplot_title = "Correlation between Artist Popularity and Track Popularity"
    scatter_plot(preprocessed_data, "artist_popularity", 300, 500, 1, scatterplot_title)
    st.markdown("""
                The scatterplot visualizes the relationship between artist popularity and track popularity, with each 
                point representing a track. A strong positive trend is apparent, indicating that as artist popularity 
                increases, track popularity also tends to increase.
                """)
    genres_by_track_decade_plots(preprocessed_data, 500, 1000)
    st.markdown("""
                We can see from the graph the count of each genre by decade. It is notable that every genre in the dataset peaked in the ***2010s***. The `Rock` genre 
                has a more balanced spread, with popularity increasing from the ***1970s***, suggesting that it was consistently popular across these decades 
                maintained a strong cultural impact over time.
                """)
    st.markdown("---")
    st.markdown("### Train-Test Split")
    st.markdown("#### *For Popularity Level Classification*")
    st.code("""
            # Select features and target variable
            data_filtered = data.copy(deep=True)
            object_columns = data_filtered.select_dtypes(include=['object']).columns
            columns_to_drop = list(object_columns) + ['track_album_release_date', 'track_decade']
            data_filtered = data_filtered.drop(columns=columns_to_drop)
            
            popularity_model_df = data_filtered

            X = popularity_model_df.drop(columns=['track_popularity', 'popularity_level_encoded'])
            y = popularity_model_df['popularity_level_encoded']
            
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
            """)
    col = st.columns(2)
    with col[0]:
        st.markdown(f"*X Train*")
        popularity_X_train
        st.markdown(f"*Count: `{popularity_X_train.shape[0]}`*")
        inner_col = st.columns(2)
        with inner_col[0]:
            st.markdown(f"*y Train*")
            popularity_y_train
            st.markdown(f"*Count: `{popularity_y_train.shape[0]}`*")
        with inner_col[1]:
            st.markdown(f"*y Test*")
            popularity_y_test
            st.markdown(f"*Count: `{popularity_y_test.shape[0]}`*")
    with col[1]:
        st.markdown(f"*X Test*")
        popularity_X_test
        st.markdown(f"*Count: `{popularity_X_test.shape[0]}`*")
        st.markdown("""
                    In this part, we prepared the dataset for training a model that predicts the popularity level of a track. We start by selecting the features and 
                    the target variable from the filtered dataset `data_filtered`, which we assign to `popularity_model_df`. The features `X` consist of all columns 
                    in `popularity_model_df` except `track_popularity` and `popularity_level_encoded`, which is the target variable `y` that we want the model to predict. 
                    
                    We then used the `train_test_split` function from `sklearn.model_selection` to split our dataset into training and testing sets. Here, we reserved
                    30% of the data for testing by using `test_size = 0.3`, which allows us to evaluate the model's performance on unseen data. Additionally, 
                    by setting a random state `random_state = 42`, we ensure that we get the same training and testing split each time we run this code.
                    """)

    st.markdown("#### *For Genre Classification*")
    st.code("""
            # Select features and target variable
            data_filtered = data.copy(deep=True)
            object_columns = data_filtered.select_dtypes(include=['object']).columns
            columns_to_drop = list(object_columns) + ['track_album_release_date']

            data_filtered = data_filtered.drop(columns=columns_to_drop)
            
            genre_model_df = data_filtered

            X = genre_model_df.drop(columns=['playlist_genre_encoded', 
                                            'playlist_subgenre_encoded'])
            y = genre_model_df['playlist_genre_encoded']
            
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
            """)
    
    col = st.columns(2)
    with col[0]:
        st.markdown(f"*X Train*")
        genre_X_train
        st.markdown(f"*Count: `{genre_X_train.shape[0]}`*")
        inner_col = st.columns(2)
        with inner_col[0]:
            st.markdown(f"*y Train*")
            genre_y_train
            st.markdown(f"*Count: `{genre_y_train.shape[0]}`*")
        with inner_col[1]:
            st.markdown(f"*y Test*")
            genre_y_test
            st.markdown(f"*Count: `{genre_y_test.shape[0]}`*")
    with col[1]:
        st.markdown(f"*X Test*")
        genre_X_test
        st.markdown(f"*Count: `{genre_X_test.shape[0]}`*")
        st.markdown("""
                    For the genre classification, we used a similar method to split the dataset into training and testing. 
                    The difference is the feature selection, as for this model, we did not drop the columns as we did in the previous 
                    model. We also included the `track_popularity` for this model, which is the target variable for the previous one.
                    We then assigned the encoded playlist genre, `playlist_genre_encoded`, as the target variable `y`.
                    """)
#======================================================================================================================================#
# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")
    # Your content for the MACHINE LEARNING page goes here
    st.markdown("### Random Forest Classifier")
    
    st.markdown("""**`Random Forest`** is an algorithm used for classification and regression as well as a variety of other tasks by supervised
                machine learning, ideal for use with large datasets, large and complex dimensions of features, and useful for explaining 
                the importance of features. The **`Random Forest Classifier`** builds an ensemble of decision trees using a randomly chosen subset of the training data.
                Each subset is then used to train an individual decision tree. The Random Forest can then make predictions by taking the majority vote or 
                averages out the results from all the trees to come up with the final output. The RandomForestClassifier module is 
                imported from the **`sklearn.ensemble`** library, which is part of the popular Scikit-Learn machine learning framework in Python.  
                """)
    st.markdown("`Source:` https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/")
    decision_tree_parts_image = Image.open('assets/rfc_figure.webp')
    st.image(decision_tree_parts_image, caption='Random Forest Classifier Figure')

    st.markdown("#### Training the Popularity Level Classification Model")
    st.code("""
            popularity_model = RandomForestClassifier(random_state = 42)
            popularity_model.fit(X_train, y_train)
            """)
    st.markdown("#### Model Evaluation")
    #--------------------------------------------------------------------------#
    popularity_y_pred = popularity_model.predict(popularity_X_test)
    popularity_accuracy = accuracy_score(popularity_y_test, popularity_y_pred)
    class_labels = ['low_popularity', 'medium_popularity', 'high_popularity']
    popularity_classification_rep = classification_report(popularity_y_test, popularity_y_pred, target_names=class_labels, output_dict=True)
    #-------------------------------------------------------------------------------------------#
    st.code("""
            y_pred = popularity_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            class_labels = ['low_popularity', 'medium_popularity', 'high_popularity']
            classification_rep = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
            """)
    
    popularity_classification_df = pd.DataFrame(popularity_classification_rep)
    popularity_classification_df = popularity_classification_df.applymap(lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else x)
    popularity_classification_df.columns = [
        col.title().replace('Avg', 'Average').replace('_', ' ')
        for col in popularity_classification_df.columns
    ]
    popularity_classification_df = popularity_classification_df.rename(index=str.title)   
    st.markdown(f""" 
                ##### *Classification Report*   
                **Accuracy** : ***`{popularity_accuracy * 100:.2f}%`***
                """)
    st.table(popularity_classification_df)
    st.markdown("""
                After training the model, it achieved an ***`accuracy of 75.46%`***, which suggests the model is performing reasonably well given the three popularity 
                levels. The precision of ***0.77*** and recall of ***0.76*** for tracks with low popularity indicates that the model identified most `low-popularity` 
                tracks but sometimes misclassifies others into this level. For `medium-popularity tracks`, precision and recall are lower, with ***0.68*** and 
                ***0.71***, respectively. This indicates that the model has more difficulty accurately identifying medium-popularity tracks. The model performs best 
                with `high-popularity tracks`, with a precision of ***0.79*** and recall of ***0.78***, suggesting that it can reliably identify tracks with high popularity.
                """)
    st.markdown("#### Feature Importance")
    col = st.columns([2, 5])
    with col[0]:
        popularity_feature_importance_df = pd.DataFrame({
            'Feature': popularity_X_train.columns,
            'Importance': popularity_model.feature_importances_
        })
        st.markdown(f"*Popularity level feature importance*")
        st.dataframe(popularity_feature_importance_df, height=500)
    with col[1]:
        feature_importance_plot(popularity_feature_importance_df, 500, 600, 1)
    st.markdown("""
                We can see from the  feature importance analysis that `artist popularity` is the most influential feature, with 
                ***0.35*** or ***35%*** importance in predicting track popularity. This shows that a track's popularity is strongly 
                influenced by the popularity of the artist, making it a valuable factor in the model's decisions. This makes sense because
                a popular artist typically has a wider audience, which drives more listeners to their tracks. Other audio features such 
                as `tempo`, `duration_ms`, and `loudness` also play notable roles, however, their individual impacts are significantly 
                lower, each contributing around ***4-5%***. `Key` and `mode` have the lowest importance among the audio features, suggesting 
                that these musical qualities have less impact on a track's popularity within this dataset.
                """)
    st.markdown("---")
    popularity_forest_image = Image.open('assets/popularity_forest.png')
    st.image(popularity_forest_image, caption='Popularity Classification Forest Plot')
    st.markdown("""
           The graph shows all of the decision trees made by the `Random Forest Classifier` which contributes to the model's ability to 
           classify tracks into different popularity levels. We used `max_depth = 3` for each tree to limiting the depth and make the 
           visualizations more interpretable.
            """)
    popularity_tree_image = Image.open('assets/popularity_tree.png')
    st.image(popularity_tree_image, caption='Popularity Classification Tree Plot')
    st.markdown("""
           The graph displays a single decision tree from the `Random Forest Classifier` model trained to classify tracks into different 
           popularity levels. We used `max_depth = 3` for each tree to limiting the depth and make the visualizations more interpretable.
            """)
    st.markdown("---")#-------------------------------------------------------------------------------------------#
    st.markdown("#### Training the Genre Classification Model")
    st.code("""
            genre_model = RandomForestClassifier(random_state=42)
            genre_model.fit(X_train, y_train)
            """)
    st.markdown("#### Model Evaluation")
    #--------------------------------------------------------------------------#
    genre_y_pred = genre_model.predict(genre_X_test)
    genre_accuracy = accuracy_score(genre_y_test, genre_y_pred)
    genre_classification_rep = classification_report(genre_y_test, genre_y_pred, target_names=le_genre.classes_, output_dict=True)
    #-------------------------------------------------------------------------------------------#
    st.code("""
            y_pred = genre_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, target_names=le_genre.classes_, output_dict=True)
            """)
    genre_classification_df = pd.DataFrame(genre_classification_rep)
    genre_classification_df = genre_classification_df.applymap(lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else x)
    genre_classification_df.columns = [
        col.title().replace('Avg', 'Average').replace('Edm', 'EDM')
        for col in genre_classification_df.columns
    ]
    genre_classification_df = genre_classification_df.rename(index=str.title)   
    st.markdown(f""" 
                ##### *Classification Report*   
                **Accuracy** : ***`{genre_accuracy * 100:.2f}%`***
                """)
    st.table(genre_classification_df)
    st.markdown("""
                The genre classification model achieved an ***`accuracy of 58.65%`***, which indicates it has some ability to 
                differentiate between genres, though there is room for improvement.The model performs best classifying `Rock` genre, with
                a precision of ***0.78*** and recall of ***0.80***. For `EDM` and `Rap` genres, the model also performs reasonably well, 
                with precision values of ***0.66*** and ***0.60***, respectively, and recall scores above ***0.65***. This suggests that 
                the model generally identifies these genres correctly but may still occasionally misclassify them as other genres.
                """)
    st.markdown("#### Feature Importance")
    col = st.columns([2, 5])
    with col[0]:
        genre_feature_importance_df = pd.DataFrame({
            'Feature': genre_X_train.columns,
            'Importance': genre_model.feature_importances_
        })
        st.markdown(f"*Genre feature importance*")
        st.dataframe(genre_feature_importance_df, height=500)
    with col[1]:
        feature_importance_plot(genre_feature_importance_df, 500, 600, 2)
    st.markdown("""
                The feature importance analysis shows that `tempo` the most influential feature in classifying genre, with an importance 
                score of ***.0.09***. This suggest that the rhythm and speed of the tracks significantly vary across genres. This if 
                followed by `speechiness` and `danceability`, respectively. The `mode` feature achieved the lowest importance score with 
                ***0.01***, implying that the modality (major or minor) of a track does not strongly differentiate genres.
                """)
    genre_forest_image = Image.open('assets/genre_forest.png')
    st.image(genre_forest_image, caption='Genre Classification Forest Plot')
    st.markdown("""
                The graph shows all of the decision trees made by the `Random Forest Classifier` which contributes to the model's ability 
                to classify the genre of a track. We used `max_depth = 3` for each tree to limiting the depth and make the visualizations 
                more interpretable.
                """)
    genre_tree_image = Image.open('assets/genre_tree.png')
    st.image(genre_tree_image, caption='Genre Classification Tree Plot')
    st.markdown("""
                The graph displays a single decision tree from the `Random Forest Classifier` model trained to classify the genre of a 
                track. We used `max_depth = 3` for each tree to limiting the depth and make the visualizations more interpretable.
                """)
    
#======================================================================================================================================#  
# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")
    tab1, tab2 = st.tabs(["Popularity Level Classification", "Genre Classification"])
    
    prediction_data = preprocessed_data.copy(deep=True)
    prediction_data['playlist_genre'] = prediction_data['playlist_genre'].str.title()
    prediction_data['playlist_subgenre'] = prediction_data['playlist_subgenre'].str.title()
    
    key_names = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"]
    
    genre_to_subgenres = prediction_data.groupby('playlist_genre')['playlist_subgenre'].unique()
    with tab1:
        st.subheader("Popularity Level Classification")
        col = st.columns(4)
        
        with col[0]:
            selected_genre = st.selectbox('Genre', list(genre_to_subgenres.keys()), key='genre')
            if selected_genre:
                subgenres = genre_to_subgenres[selected_genre]
                selected_subgenre = st.selectbox('Subgenre', subgenres, key='subgenre')
            else:
                selected_subgenre = st.selectbox('Subgenre', [])
            tempo = st.number_input('Tempo (BPM)', min_value=40, max_value=200, step=1, key='tempo1', value=40 if st.session_state.clear else st.session_state.get('tempo1', 40))
            artist_popularity = st.number_input('Artist Popularity', min_value=0.0, max_value=100.0, step=5.0, key='artist_popularity1', value=0.0 if st.session_state.clear else st.session_state.get('artist_popularity1', 0.0))
            danceability = st.number_input('Danceability', min_value=0.0, max_value=1.0, step=0.1, key='danceability1', value=0.0 if st.session_state.clear else st.session_state.get('danceability1', 0.0))
            energy = st.number_input('Energy', min_value=0.0, max_value=10.0, step=0.1, key='energy1', value=0.0 if st.session_state.clear else st.session_state.get('energy', 0.0))
        with col[1]:
            selected_key = st.selectbox("Key", list(key_names), key='key1')
            loudness = st.number_input('Loudness (dB)', min_value=-60.0, max_value=0.0, step=0.1, key='loudness1', value=0.0 if st.session_state.clear else st.session_state.get('loudness1', 0.0))
            mode = st.selectbox('Mode', ['Major', 'Minor'], key='mode1')
            speechiness = st.number_input('Speechiness', min_value=0.0, max_value=10.0, step=0.1, key='speechiness1', value=0.0 if st.session_state.clear else st.session_state.get('speechiness1', 0.0))
            acousticness = st.number_input('Acousticness', min_value=0.0, max_value=1.0, step=0.1, key='acousticness1', value=0.0 if st.session_state.clear else st.session_state.get('acousticness1', 0.0))
            instrumentalness = st.number_input('Instrumentalness', min_value=0.0, max_value=1.0, step=0.1, key='instrumentalness1', value=0.0 if st.session_state.clear else st.session_state.get('instrumentalness1', 0.0))
        with col[2]:
            liveness = st.number_input('Liveness', min_value=0.0, max_value=1.0, step=0.1, key='liveness1', value=0.0 if st.session_state.clear else st.session_state.get('liveness1', 0.0))
            duration_minutes = st.number_input('Duration (minutes)', min_value=0, max_value=100, step=1, key='duration_minutes1', value=3 if st.session_state.clear else st.session_state.get('duration_minutes1', 3))
            valence = st.number_input('Valence', min_value=0.0, max_value=1.0, step=0.1, key='valence1', value=0.0 if st.session_state.clear else st.session_state.get('valence1', 0.0))
            track_year = st.number_input('Year Released', min_value=1900, max_value=2024, step=1, key='track_year1', value=2010 if st.session_state.clear else st.session_state.get('track_year1', 2010))
            track_month = st.number_input('Month Released', min_value=1, max_value=12, step=1, key='track_month1', value=1 if st.session_state.clear else st.session_state.get('track_month1', 1))
            st.write("")
            st.write("")   
            if st.button('Detect Popularity', key='detect_popularity'):
                duration_ms = duration_minutes * 60000
                track_age = 2024 - track_year
                key = key_names.index(selected_key)
                mode = 1 if mode == 'Major' else 0
                
                selected_genre_encoded = preprocessed_data[preprocessed_data['playlist_genre'].str.title() == selected_genre]['playlist_genre_encoded'].unique()
                selected_subgenre_encoded = preprocessed_data[preprocessed_data['playlist_subgenre'].str.title() == selected_subgenre]['playlist_subgenre_encoded'].unique()
                playlist_genre_encoded = int(selected_genre_encoded[0]) if len(selected_genre_encoded) > 0 else -1
                playlist_subgenre_encoded = int(selected_subgenre_encoded[0]) if len(selected_subgenre_encoded) > 0 else -1
                
                popularity_input_data = [[danceability, energy, key, loudness, mode, speechiness, 
                                          acousticness, instrumentalness, liveness, valence, tempo, 
                                          duration_ms, track_year, track_month, track_age, 
                                          playlist_genre_encoded, playlist_subgenre_encoded,artist_popularity]]
            
                popularity_prediction = popularity_model.predict(popularity_input_data)
                classes_list = ['Unpopular', 'Moderately Popular', 'Popular']
                popularity_result = classes_list[int(popularity_prediction[0])]
        
        with col[3]:
            with st.expander('Options', expanded=True):
                show_dataset1 = st.checkbox('Show Dataset', key = "show_dataset1")
                show_classes1 = st.checkbox('Show All Classes', key = "show_classes1")
                show_high = st.checkbox('Show Popular', key='show_high')
                show_medium = st.checkbox('Show Moderately Popular', key='show_medium')
                show_low = st.checkbox('Show Unpopular', key='show_low')
                clear_results = st.button('Clear Results', key=1)
                if clear_results:
                    st.session_state.clear = True
                
            if st.session_state.get('detect_popularity'):
                st.markdown("---")
                st.markdown('The song is: ')
                st.markdown(f'<p class="custom-title">{popularity_result}</p>', unsafe_allow_html=True)
        st.markdown("---")
        high_samples = preprocessed_data[preprocessed_data["popularity_level"] == "high_popularity"].head(5)
        medium_samples = preprocessed_data[preprocessed_data["popularity_level"] == "medium_popularity"].head(5)
        low_samples = preprocessed_data[preprocessed_data["popularity_level"] == "low_popularity"].head(5)

        if show_dataset1:
            st.subheader("Dataset")
            st.dataframe(preprocessed_data, use_container_width=True, hide_index=True)

        if show_classes1:
            st.subheader("High-Popularity Samples")
            st.dataframe(high_samples, use_container_width=True, hide_index=True)

            st.subheader("Medium-Popularity Samples")
            st.dataframe(medium_samples, use_container_width=True, hide_index=True)

            st.subheader("Low-Popularity Samples")
            st.dataframe(low_samples, use_container_width=True, hide_index=True)

        if show_high:
            st.subheader("High-Popularity Samples")
            st.dataframe(high_samples, use_container_width=True, hide_index=True)

        if show_medium:
            st.subheader("Medium-Popularity Samples")
            st.dataframe(medium_samples, use_container_width=True, hide_index=True)

        if show_low:
            st.subheader("Low-Popularity Samples")
            st.dataframe(low_samples, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Genre Classification")
        col = st.columns(4)
        
        with col[0]:
            track_popularity = st.number_input('Track Popularity', min_value=0.0, max_value=100.0, step=5.0, key='track_popularity', value=0.0 if st.session_state.clear else st.session_state.get('track_popularity', 0.0))
            artist_popularity = st.number_input('Artist Popularity', min_value=0.0, max_value=100.0, step=5.0, key='artist_popularity2', value=0.0 if st.session_state.clear else st.session_state.get('artist_popularity2', 0.0))
            track_year = st.number_input('Year Released', min_value=1900, max_value=2024, step=1, key='track_year2', value=2010 if st.session_state.clear else st.session_state.get('track_year2', 2010))
            track_month = st.number_input('Month Released', min_value=1, max_value=12, step=1, key='track_month2', value=1 if st.session_state.clear else st.session_state.get('track_month2', 1))
            duration_minutes = st.number_input('Duration (minutes)', min_value=0, max_value=100, step=1, key='duration_minutes2', value=3 if st.session_state.clear else st.session_state.get('duration_minutes2', 3))
            tempo = st.number_input('Tempo (BPM)', min_value=40, max_value=200, step=1, key='tempo2', value=40 if st.session_state.clear else st.session_state.get('tempo2', 40))
        with col[1]:
            selected_key = st.selectbox("Key", list(key_names), key='key2')
            mode = st.selectbox('Mode', ['Major', 'Minor'], key='mode2')
            danceability = st.number_input('Danceability', min_value=0.0, max_value=1.0, step=0.1, key='danceability2', value=0.0 if st.session_state.clear else st.session_state.get('danceability2', 0.0))
            energy = st.number_input('Energy', min_value=0.0, max_value=10.0, step=0.1, key='energy2', value=0.0 if st.session_state.clear else st.session_state.get('energy2', 0.0))
            loudness = st.number_input('Loudness (dB)', min_value=-60.0, max_value=0.0, step=0.1, key='loudness2', value=0.0 if st.session_state.clear else st.session_state.get('loudness2', 0.0))
            st.write("")
            st.write("")   
            if st.button('Detect Genre', key='detect_genre'):
                duration_ms = duration_minutes * 60000
                track_age = 2024 - track_year
                key = key_names.index(selected_key)
                mode = 1 if mode == 'Major' else 0
                track_decade = (track_year // 10) * 10
                
                genre_input_data = [[track_popularity, danceability, energy, key, loudness, mode, speechiness, 
                                     acousticness, instrumentalness, liveness, valence, tempo, duration_ms,
                                     track_year, track_month, track_decade, track_age, artist_popularity]]
            
                genre_prediction = genre_model.predict(genre_input_data)
                genre_list = ['EDM', 'Latin', 'Pop', 'R&B',' Rap', 'Rock']
                genre_result = genre_list[genre_prediction[0]]
        with col[2]:
            speechiness = st.number_input('Speechiness', min_value=0.0, max_value=10.0, step=0.1, key='speechiness2', value=0.0 if st.session_state.clear else st.session_state.get('speechiness2', 0.0))
            acousticness = st.number_input('Acousticness', min_value=0.0, max_value=1.0, step=0.1, key='acousticness2', value=0.0 if st.session_state.clear else st.session_state.get('acousticness2', 0.0))
            instrumentalness = st.number_input('Instrumentalness', min_value=0.0, max_value=1.0, step=0.1, key='instrumentalness2', value=0.0 if st.session_state.clear else st.session_state.get('instrumentalness2', 0.0))
            liveness = st.number_input('Liveness', min_value=0.0, max_value=1.0, step=0.1, key='liveness2', value=0.0 if st.session_state.clear else st.session_state.get('liveness2', 0.0))
            valence = st.number_input('Valence', min_value=0.0, max_value=1.0, step=0.1, key='valence2', value=0.0 if st.session_state.clear else st.session_state.get('valence2', 0.0))
            
        
        with col[3]:
            with st.expander('Options', expanded=True):
                show_dataset2 = st.checkbox('Show Dataset', key='show_dataset2')
                show_classes2 = st.checkbox('Show All Classes', key='show_classes2')
                show_pop = st.checkbox('Show Pop', key='show_pop')
                show_rap = st.checkbox('Show Rap', key='show_rap')
                show_rock = st.checkbox('Show Rock', key='show_rock')
                show_latin = st.checkbox('Show Latin', key='show_latin')
                show_rnb = st.checkbox('Show R&B', key='show_rnb')
                show_edm = st.checkbox('Show EDM', key='show_edm')
                clear_results = st.button('Clear Results', key=2)
                if clear_results:
                    st.session_state.clear = True
                
            if st.session_state.get('detect_genre'):
                st.markdown("---")
                st.markdown('The song is: ')
                st.markdown(f'<p class="custom-title">{genre_result}</p>', unsafe_allow_html=True)
        pop_samples = preprocessed_data[preprocessed_data["playlist_genre"] == "pop"].head(5)
        rap_samples = preprocessed_data[preprocessed_data["playlist_genre"] == "rap"].head(5)
        rock_samples = preprocessed_data[preprocessed_data["playlist_genre"] == "rock"].head(5)
        latin_samples = preprocessed_data[preprocessed_data["playlist_genre"] == "latin"].head(5)
        rnb_samples = preprocessed_data[preprocessed_data["playlist_genre"] == "r&b"].head(5)
        edm_samples = preprocessed_data[preprocessed_data["playlist_genre"] == "edm"].head(5)
        st.markdown("---")
        if show_dataset2:
            st.subheader("Dataset")
            st.dataframe(preprocessed_data, use_container_width=True, hide_index=True)

        if show_classes2:
            st.subheader("Pop Samples")
            st.dataframe(pop_samples, use_container_width=True, hide_index=True)
            st.subheader("Rap Samples")
            st.dataframe(rap_samples, use_container_width=True, hide_index=True)
            st.subheader("Rock Samples")
            st.dataframe(rock_samples, use_container_width=True, hide_index=True)
            st.subheader("Latin Samples")
            st.dataframe(latin_samples, use_container_width=True, hide_index=True)
            st.subheader("R&B Samples")
            st.dataframe(rnb_samples, use_container_width=True, hide_index=True)
            st.subheader("EDM Samples")
            st.dataframe(edm_samples, use_container_width=True, hide_index=True)

        if show_pop:
            st.subheader("Pop Samples")
            st.dataframe(pop_samples, use_container_width=True, hide_index=True)
            
        if show_rap:
            st.subheader("Rap Samples")
            st.dataframe(rap_samples, use_container_width=True, hide_index=True)
            
        if show_rock:
            st.subheader("Rock Samples")    
            st.dataframe(rock_samples, use_container_width=True, hide_index=True)
            
        if show_latin:
            st.subheader("Latin Samples")
            st.dataframe(latin_samples, use_container_width=True, hide_index=True)
            
        if show_rnb:
            st.subheader("R&B Samples")
            st.dataframe(rnb_samples, use_container_width=True, hide_index=True)
            
        if show_edm:
            st.subheader("EDM Samples")
            st.dataframe(edm_samples, use_container_width=True, hide_index=True)
            
            
            
            
# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    st.markdown("""
### **Dataset Characteristics**
 > - Among the audio features, `energy` and `loudness` show the strongest positive correlation, indicating that high-energy songs are generally louder
 > - The genre distribution is balanced, providing equal representation of each genre, which helps the model learn fairly without bias. 
 > - `Pop` and `Latin` genres have the highest average popularity, while EDM has the lowest. Subgenres also show differences, with `Post-Teen Pop` being the most popular and `Progressive Electro House` the least.
 > - Each genre shows unique characteristics. EDM tracks have the highest energy, while R&B songs are generally less intense. Rap is more speech-focused, while EDM has fewer vocals. Song speed and length also vary, with EDM tracks being faster and Rock songs tending to be longer.

### **Popularity Level Classification**
 > - Using Random Forest Classification, the model achieved ***`accuracy of 75.46%`***, indicating a that model performed relatively well in predicting popularity levels.
 > - The model achieved a ***precision of 0.77*** and ***recall of 0.76*** for `low-popularity` tracks.
 > - For `medium-popularity` tracks, the model achieved a ***precision of 0.69*** and ***recall of 0.72***.
 > - The model performs best with `high-popularity` tracks, with a ***precision of 0.80*** and ***recall of 0.78***, suggesting that it can identify most high-popularity tracks effectively.

### **Genre Classification**
 > - The genre classification model achieved an ***`accuracy of 58.65%`***, which indicates that the model has some ability to classify genres, though with room for improvement.
 > - The `Rock` genre Achieves the highest precision and recall, with ***`0.77`*** and ***`0.80***`, respectively. This suggests that the model is most effective in classifying rock tracks.
 > - For `EDM` and `Rap` genres, the model achieved a ***0.66 precision*** and ***0.72 recall*** for EDM, and ***0.59 precision*** and ***0.64 recall*** for Rap.
 > - The model struggles most with `Latin` and `Pop` genres, with lower precision and recall of ***0.40*** for Pop, and ***0.51*** and ***0.47*** for Latin.
 > - For `R&B` genre, the model achieved a ***precision of 0.54*** and ***recall of 0.47***.
    """)
