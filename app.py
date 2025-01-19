import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load TF-IDF model and dataset
try:
    tfidf = joblib.load('vectorizer.pkl')
    df = pd.read_csv('Indonesia_food_dataset.csv')
    df['Ingredients'] = df['Ingredients'].fillna('')
    model = tfidf.transform(df['Ingredients'])
except FileNotFoundError as e:
    raise RuntimeError(f"File tidak ditemukan: {e}")
except Exception as e:
    raise RuntimeError(f"Kesalahan saat memuat model atau data: {e}")

# Function to add numbered steps
# Function to add numbered steps and clean formatting
def format_steps(steps):
    """
    Format the steps by:
    1. Splitting them using '--' as a separator.
    2. Adding sequential numbering.
    3. Returning as HTML-compatible string with line breaks.
    """
    # Split steps by '--' and clean up each step
    step_list = [step.strip() for step in steps.split('--') if step.strip()]
    # Add numbering
    numbered_steps = [f"{i+1}. {step}" for i, step in enumerate(step_list)]
    return "<br>".join(numbered_steps)

# Function to format ingredients into a list
def format_ingredients(ingredients):
    # Split ingredients by '--' and remove extra spaces
    ingredients_list = [ingredient.strip() for ingredient in ingredients.split('--') if ingredient.strip()]
    formatted_ingredients = "\n".join([f"• {ingredient}" for ingredient in ingredients_list])
    return formatted_ingredients

# Menambahkan CSS untuk penataan halaman
st.markdown("""
    <style>

.recipe-card:hover {
        background-color: #ff8c1a;
        color: white;
        transition: 0.3s;
}
    html {
        scroll-behavior: smooth;
        margin: 0;
        padding: 0;
    }

    body {
        background-color: #ffeae0;
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        margin: 0;
        padding: 0;
    }

    /* Navbar Styles */
    .navbar {
        background-color: #ff511c;
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
    }

    .navbar .navbar-brand b {
        color: #fff;
        font-size: 24px;
    }

    .navbar-nav .nav-link {
        color: #fff;
        padding: 10px 15px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }

    .navbar-nav .nav-link:hover {
        background-color: #ff8c1a;
    }

    /* Form Styles */
    .form-container {
        max-width: 800px;
        margin: 30px auto;
        padding: 20px;
        background-color: #fff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .input-field {
        width: 100%;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        border: 1px solid #ff511c;
    }

    /* Recipe Cards */
    .recipe-card {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 10px;
        transition: transform 0.3s ease-in-out, background-color 0.3s;
    }

    .recipe-card:hover {
        transform: translateY(-5px);
        background-color: #ffdcc2; /* Ganti warna hover sesuai preferensi */
        color: #333; /* Pastikan teks tetap terlihat jelas */
    }

    .recipe-card .card-title {
        font-size: 20px;
        font-weight: bold;
        color: #ff511c;
    }

    .recipe-card .card-text {
        font-size: 14px;
        color: #555;
    }

    .recipe-card .card-subtitle {
        font-size: 16px;
        color: #ff8c1a;
    }

    .btn-signup {
        background-color: #ff511c;
        color: #fff;
        border-radius: 5px;
        padding: 10px 20px;
        text-decoration: none;
        display: inline-block;
    }

    .btn-signup:hover {
        background-color: #ffffff;
        transition: 0.3s;
    }

    /* Footer */
    .footer {
        background-color: #ff511c;
        color: #fff;
        text-align: center;
        padding: 15px 0;
        position: fixed;
        width: 100%;
        bottom: 0;
    }

    /* Media Queries for Responsiveness */
    @media screen and (max-width: 768px) {
        body {
            font-size: 14px;
        }

        .form-container {
            padding: 15px;
        }

        .recipe-card {
            margin: 5px;
        }

        .navbar {
            flex-direction: column;
            align-items: flex-start;
        }

        .navbar-nav .nav-link {
            padding: 8px 12px;
        }

        .footer {
            padding: 10px 0;
        }
    }

    @media screen and (max-width: 480px) {
        .recipe-card {
            padding: 15px;
        }

        .btn-signup {
            font-size: 14px;
            padding: 8px 15px;
        }
    }

    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Rekomendasi Resep Masakan")
st.write("Masukkan bahan yang Anda miliki untuk mendapatkan rekomendasi resep.")

# Input fields for ingredients in a responsive form container
with st.form("ingredient_form"):
    bahan1 = st.text_input("Bahan 1:", key="bahan1", label_visibility="collapsed")
    bahan2 = st.text_input("Bahan 2:", key="bahan2", label_visibility="collapsed")
    bahan3 = st.text_input("Bahan 3:", key="bahan3", label_visibility="collapsed")
    submit_button = st.form_submit_button("Cari Resep")

if submit_button:
    query_all = ' '.join([bahan1, bahan2, bahan3]).strip()

    if not query_all:
        st.error("Harap masukkan minimal satu bahan untuk mencari resep")
    else:
        query_vector = tfidf.transform([query_all])

        # Calculate similarities
        cosine_similarities = cosine_similarity(query_vector, model).flatten()
        top_indices = np.argpartition(-cosine_similarities, 3)[:3]
        top_indices = top_indices[np.argsort(-cosine_similarities[top_indices])]

        results = []
        for idx in top_indices:
            recipe = df.iloc[idx]
            # Format ingredients and steps
            bahan_bersih = format_ingredients(recipe['Ingredients'])  # Use the format_ingredients function
            langkah_bersih = format_steps(recipe['Steps'])  # Use the format_steps function
            results.append({
                "Masakan": recipe['Title'], 
                "Bahan": bahan_bersih, 
                "Langkah": langkah_bersih
            })

        # Display results in recipe card format
        if results:
            for recipe in results:
                st.markdown(f"""
                    <div class="recipe-card">
                        <div class="card-title">{recipe["Masakan"]}</div>
                        <div class="card-subtitle">Bahan:</div>
                        <div class="card-text">{recipe["Bahan"]}</div>
                        <div class="card-subtitle">Langkah:</div>
                        <div class="card-text">{recipe["Langkah"]}</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.write("Tidak ada resep yang ditemukan.")

st.markdown("""
    <style>
    .footer {
        background-color: #ff511c;
        color: #fff;
        text-align: center;
        padding: 15px 0;
        position: relative;
        bottom: 0;
        width: 100%;
    }

    @media screen and (max-width: 768px) {
        .footer {
            padding: 10px 0;
        }
    }

    @media screen and (max-width: 480px) {
        .footer {
            padding: 8px 0;
            font-size: 14px;
        }
    }
    </style>
    <div class="footer">
        <p>© 2025 Rekomendasi Resep Masakan | All Rights Reserved</p>
    </div>
""", unsafe_allow_html=True)