import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process
import requests
from io import BytesIO
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Movie Recommender 2025 Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom CSS for Modern Design
# ------------------------------


def apply_custom_css():
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Main Background Gradient */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
        }
        
        /* Content Container */
        .main .block-container {
            padding: 2rem 3rem;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
        
        /* Title Styling */
        h1 {
            color: #667eea;
            font-weight: 700;
            text-align: center;
            font-size: 3rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #764ba2;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        /* Input Fields */
        .stTextInput input {
            border-radius: 10px;
            border: 2px solid #667eea;
            padding: 12px;
            font-size: 1rem;
        }
        
        .stTextInput input:focus {
            border-color: #764ba2;
            box-shadow: 0 0 0 0.2rem rgba(118, 75, 162, 0.25);
        }
        
        /* Buttons */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 30px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        /* Movie Cards */
        .movie-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .movie-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.25);
        }
        
        /* Watchlist Section */
        .watchlist-item {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            margin: 5px 0;
            font-weight: 500;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #667eea;
            font-size: 2rem;
            font-weight: 700;
        }
        
        /* Dataframe */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* Success/Warning Messages */
        .stSuccess {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            border-radius: 5px;
        }
        
        .stWarning {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            border-radius: 5px;
        }
        
        /* Section Headers */
        h2, h3 {
            color: #667eea;
            font-weight: 600;
            margin-top: 2rem;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: #764ba2;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)


apply_custom_css()

# ------------------------------
# Load Data and Models
# ------------------------------


@st.cache_data
def load_data():
    try:
        movies_df = pd.read_pickle("movies_df.pkl")
        tfidf = joblib.load("tfidf_vectorizer.joblib")
        cosine_sim = joblib.load("cosine_sim_matrix.joblib")
        return movies_df, tfidf, cosine_sim
    except Exception as e:
        return None, None, None


movies_df, tfidf, cosine_sim = load_data()

if movies_df is None:
    st.error("⚠️ Error loading data files. Please ensure movies_df.pkl, tfidf_vectorizer.joblib, and cosine_sim_matrix.joblib are in the same directory.")
    st.stop()

# ------------------------------
# TMDB API Configuration
# ------------------------------
TMDB_API_KEY = "YOUR_TMDB_API_KEY"  # Replace with your TMDB API key


@st.cache_data(ttl=3600)
def get_poster(title):
    """Fetch poster URL from TMDB API with caching"""
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        data = requests.get(url, timeout=5).json()
        if data.get('results'):
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                full_path = f"https://image.tmdb.org/t/p/w300{poster_path}"
                response = requests.get(full_path, timeout=5)
                img = Image.open(BytesIO(response.content))
                return img
    except:
        pass
    return None

# ------------------------------
# Fuzzy Search Function
# ------------------------------


def fuzzy_search(query, choices, limit=5):
    results = process.extract(query, choices, limit=limit)
    return [r[0] for r in results]

# ------------------------------
# Recommendation Function
# ------------------------------


def recommend_movies(title, top_n=5, min_rating=0.0, genres_filter=[]):
    try:
        if title not in movies_df['title'].values:
            matched_titles = fuzzy_search(
                title, movies_df['title'].tolist(), limit=1)
            if not matched_titles:
                return pd.DataFrame(), None
            matched_title = matched_titles[0]
        else:
            matched_title = title

        idx = movies_df[movies_df['title'] == matched_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get more to account for filtering
        sim_scores = sim_scores[1:top_n+20]
        movie_indices = [i[0] for i in sim_scores]

        recs = movies_df.iloc[movie_indices][[
            'title', 'genres', 'rating', 'plot']].copy()

        # Apply filters
        recs = recs[recs['rating'] >= min_rating]
        if genres_filter:
            recs = recs[recs['genres'].apply(
                lambda x: any(g in x for g in genres_filter))]

        return recs.head(top_n), matched_title
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return pd.DataFrame(), None


# ------------------------------
# Initialize Session State
# ------------------------------
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []
if 'history_df' not in st.session_state:
    st.session_state['history_df'] = pd.DataFrame(
        columns=['Timestamp', 'InputMovie', 'Recommendations'])

# ------------------------------
# Header Section
# ------------------------------
st.markdown("<h1>🎬 Movie Recommender 2025</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Discover your next favorite movie with AI-powered recommendations</p>",
            unsafe_allow_html=True)

# ------------------------------
# Sidebar Configuration
# ------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Theme selector
    theme = st.radio(
        "🎨 Theme", ["Light Mode", "Dark Mode", "Purple Gradient"], index=2)

    st.markdown("---")

    # Recommendation settings
    st.markdown("### 🎯 Recommendation Settings")
    top_n = st.slider("Number of Recommendations", 3, 15, 6)
    min_rating = st.slider("Minimum Rating", 0.0, 10.0, 5.0, 0.5)

    st.markdown("---")

    # Genre filters
    st.markdown("### 🎭 Genre Filters")
    all_genres = []
    if movies_df is not None:
        try:
            all_genres = sorted(
                {g for gs in movies_df['genres'] for g in gs.split('|') if g})
        except:
            all_genres = []
    selected_genres = st.multiselect("Select Genres", all_genres)

    st.markdown("---")

    # Additional options
    st.markdown("### 📊 Data Options")
    show_stats = st.checkbox("Show Statistics", value=True)
    show_data = st.checkbox("Show Dataset Sample")

    st.markdown("---")

    # Export options
    st.markdown("### 💾 Export")
    if st.button("📥 Download History"):
        if not st.session_state['history_df'].empty:
            csv = st.session_state['history_df'].to_csv(
                index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f'movie_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        else:
            st.info("No history to download yet!")

# ------------------------------
# Main Content Area
# ------------------------------

# Statistics Section
if show_stats and movies_df is not None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", f"{len(movies_df):,}")
    with col2:
        avg_rating = movies_df['rating'].mean(
        ) if 'rating' in movies_df.columns else 0
        st.metric("Avg Rating", f"{avg_rating:.1f}/10")
    with col3:
        st.metric("In Watchlist", len(st.session_state['watchlist']))
    with col4:
        st.metric("Searches", len(st.session_state['history_df']))

st.markdown("---")

# Dataset Sample
if show_data and movies_df is not None:
    with st.expander("📁 Dataset Sample", expanded=False):
        try:
            st.dataframe(movies_df.head(
                20), use_container_width=True, height=300)
        except Exception as e:
            st.error(f"Error displaying dataset: {str(e)}")

# ------------------------------
# Search Section
# ------------------------------
st.markdown("### 🔍 Find Your Next Movie")

col_search1, col_search2 = st.columns([3, 1])
with col_search1:
    movie_input = st.text_input(
        "Enter a movie name",
        placeholder="e.g., The Dark Knight, Inception, Avatar...",
        label_visibility="collapsed"
    )
with col_search2:
    search_button = st.button("🎬 Get Recommendations",
                              use_container_width=True)

# ------------------------------
# Recommendation Results
# ------------------------------
if search_button and movie_input:
    if movies_df is None:
        st.error(
            "⚠️ Cannot generate recommendations. Data files are not loaded properly.")
    else:
        with st.spinner("🎥 Finding perfect recommendations for you..."):
            recommendations, matched_title = recommend_movies(
                movie_input, top_n, min_rating, selected_genres
            )

        if recommendations.empty:
            st.warning(
                "⚠️ No movies found matching your criteria. Try adjusting the filters!")
        else:
            if matched_title != movie_input:
                st.info(f"💡 Showing results for: **{matched_title}**")

            st.markdown(f"### 🎯 Top {len(recommendations)} Recommendations")

            # Display recommendations in grid
            cols = st.columns(min(3, len(recommendations)))

            for i, (idx, row) in enumerate(recommendations.iterrows()):
                with cols[i % 3]:
                    st.markdown("<div class='movie-card'>",
                                unsafe_allow_html=True)

                    # Poster
                    img = get_poster(row['title'])
                    if img:
                        st.image(img, use_column_width=True)
                    else:
                        st.image(
                            "https://via.placeholder.com/300x450?text=No+Poster", use_column_width=True)

                    # Movie details
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"⭐ **{row['rating']:.1f}** / 10")
                    st.markdown(f"🎭 {row['genres']}")

                    # Plot in expander
                    with st.expander("📖 Plot"):
                        st.write(
                            row['plot'][:200] + "..." if len(row['plot']) > 200 else row['plot'])

                    # Watchlist button
                    if st.button(f"➕ Add to Watchlist", key=f"add_{idx}"):
                        if row['title'] not in st.session_state['watchlist']:
                            st.session_state['watchlist'].append(row['title'])
                            st.success(f"Added to watchlist!")
                        else:
                            st.info("Already in watchlist!")

                    st.markdown("</div>", unsafe_allow_html=True)

            # Save to history
            hist_row = {
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'InputMovie': matched_title,
                'Recommendations': ", ".join(recommendations['title'].tolist())
            }
            st.session_state['history_df'] = pd.concat(
                [st.session_state['history_df'], pd.DataFrame([hist_row])],
                ignore_index=True
            )

            # Genre distribution chart
            st.markdown("---")
            st.markdown("### 📊 Genre Distribution")

            genre_counts = {}
            for g in recommendations['genres']:
                for genre in g.split('|'):
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1

            if genre_counts:
                fig = px.bar(
                    x=list(genre_counts.keys()),
                    y=list(genre_counts.values()),
                    labels={'x': 'Genre', 'y': 'Count'},
                    color=list(genre_counts.values()),
                    color_continuous_scale='Purples'
                )
                fig.update_layout(
                    showlegend=False,
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Watchlist Section
# ------------------------------
if st.session_state['watchlist']:
    st.markdown("---")
    st.markdown("### 🎯 Your Watchlist")

    watchlist_cols = st.columns(4)
    for i, movie in enumerate(st.session_state['watchlist']):
        with watchlist_cols[i % 4]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"<div class='watchlist-item'>{movie}</div>", unsafe_allow_html=True)
            with col2:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state['watchlist'].remove(movie)
                    st.rerun()

# ------------------------------
# History Section
# ------------------------------
if not st.session_state['history_df'].empty:
    st.markdown("---")
    st.markdown("### 📜 Recommendation History")
    st.dataframe(
        st.session_state['history_df'].tail(10),
        use_container_width=True,
        height=300
    )

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("<div class='footer'>Made with ❤️ by Dheeraj Muley — 2025</div>",
            unsafe_allow_html=True)
