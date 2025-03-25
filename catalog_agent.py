import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import warnings
import nltk
import os
import re
import difflib
from rapidfuzz import fuzz, process

nltk.download('punkt')

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load data from CSV file
@st.cache_data
def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Currency conversion rates to USD
def get_currency_conversion_rates():
    return {
        'USD': 1.0,
        'INR': 0.012,    # 1 INR = 0.012 USD
        'EUR': 1.09      # 1 EUR = 1.09 USD
    }

# Process data
def preprocess_data(df):
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Handle NaN values
    df['product_name'] = df['product_name'].fillna('')
    df['description'] = df['description'].fillna('')
    df['product_name'] = df['product_name'].astype(str).str.strip().str.upper()
    df['description'] = df['description'].astype(str).str.strip()
    
    # Convert numeric columns to appropriate types
    numeric_columns = ['unit_price', 'quantity', 'lead_time', 'shipping_charges', 'specifications.weight.value']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Handle currency
    df['currency'] = df['currency'].fillna('USD').astype(str).str.strip().str.upper()
    
    # Convert all prices to USD
    conversion_rates = get_currency_conversion_rates()
    df['currency_rate'] = df['currency'].map(conversion_rates)
    df['currency_rate'] = df['currency_rate'].fillna(1.0)  # Default to 1.0 if currency not found
    
    # Convert prices to USD
    df['unit_price_usd'] = df['unit_price'] * df['currency_rate']
    df['shipping_charges_usd'] = df['shipping_charges'] * df['currency_rate']
    
    # Extract discount as a numeric value
    df['discount_value'] = df['discounts'].apply(lambda x: float(str(x).replace('%', '')) / 100 if pd.notna(x) and '%' in str(x) else 0.0)
    
    # Calculate total price (unit price + shipping - discount)
    df['total_price_usd'] = df['unit_price_usd'] + df['shipping_charges_usd']
    df['total_price_usd'] = df['total_price_usd'] * (1 - df['discount_value'])
    
    # Keep original currency for reference
    df['original_currency'] = df['currency']
    df['original_unit_price'] = df['unit_price']
    
    return df

# Define the stemmed vectorizer
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        stemmer = PorterStemmer()
        return lambda doc: [stemmer.stem(word) for word in analyzer(doc)]

# Function to handle spelling corrections
def correct_spelling(query, reference_words, threshold=90):
    """
    Correct misspelled words in a query using fuzzy matching against reference words.
    
    Args:
        query (str): The user query
        reference_words (list): List of valid words to check against
        threshold (int): Minimum similarity score (0-100) to accept a correction
        
    Returns:
        str: Corrected query
    """
    words = query.lower().split()
    corrected_words = []
    
    for word in words:
        # Skip very short words, stop words, or numbers
        if len(word) <= 2 or word.isdigit():
            corrected_words.append(word)
            continue
            
        # Use rapidfuzz to find closest matches
        matches = process.extract(word, reference_words, scorer=fuzz.ratio, limit=1)
        
        if matches and matches[0][1] >= threshold:
            corrected_words.append(matches[0][0])
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

def find_similar_products(df, query, top_n=6):
    """
    Find products similar to the user's query using TF-IDF + Cosine Similarity.
    Enhanced to handle typos and spelling mistakes using fuzzy matching.
    """
    # Define category mappings for common search terms
    category_mappings = {
        "stationery": ["pen", "pencil", "eraser", "notebook", "paper", "glue", "scissors", "marker", "highlighter", "stapler"],
        "laptop": ["laptop", "notebook computer", "macbook", "thinkpad", "chromebook", "dell", "hp", "lenovo", "asus"],
        "desktop": ["desktop", "computer", "tower", "workstation"],
        "computer": ["laptop", "desktop", "notebook computer", "macbook", "thinkpad", "chromebook", "dell", "hp", "lenovo", "asus"],
        "phone": ["phone", "smartphone", "iphone", "android", "samsung", "pixel", "mobile"],
        "tablet": ["tablet", "ipad", "samsung tab", "surface"],
        "ball": ["scoccer","football"],
        "kitchen": ["cookware", "utensil", "knife", "pot", "pan", "blender", "mixer"],
        "furniture": ["chair", "table", "desk", "sofa", "cabinet", "bookshelf"],
        "clothing": ["shirt", "pant", "dress", "jacket", "sweater", "coat"],
        'metal': ['bar', 'round', 'square', 'stainless', 'steel', 'aluminum', 'metal', 'mm', 'inch'],
        'tool': ['wrench', 'screwdriver', 'caliper', 'tool', 'pneumatic', 'impact'],
        'adhesive': ['glue', 'adhesive', 'paste', 'all purpose'],
        'mechanical': ['belt', 'hose', 'assembly', 'hydraulic', 'v-belt'],
        'bar': ['metal bar', 'steel bar', 'aluminum bar', 'round bar', 'square bar', 'mm']
    }
    
    # Add plural forms to category mappings
    expanded_mappings = {}
    for category, terms in category_mappings.items():
        expanded_terms = terms.copy()
        # Add plural forms for category terms
        plural_category = category + 's' if not category.endswith('s') else category
        expanded_mappings[category] = terms
        expanded_mappings[plural_category] = terms
        
        # Also add plural forms for each term
        for term in terms:
            plural_term = term + 's' if not term.endswith('s') else term
            if plural_term not in expanded_terms:
                expanded_terms.append(plural_term)
        expanded_mappings[category] = expanded_terms
        expanded_mappings[plural_category] = expanded_terms
    
    # Create a reference vocabulary for spelling correction
    reference_vocabulary = []
    for category, terms in expanded_mappings.items():
        reference_vocabulary.append(category)
        reference_vocabulary.extend(terms)
    
    # Get unique product names for spelling correction
    all_product_words = set()
    for product_name in df['product_name'].dropna():
        words = re.findall(r'\b\w+\b', product_name.lower())
        all_product_words.update(words)
    
    reference_vocabulary.extend(list(all_product_words))
    reference_vocabulary = list(set(reference_vocabulary))  # Remove duplicates
    
    # Apply spelling correction to the query
    query_lower = query.lower().strip()
    corrected_query = correct_spelling(query_lower, reference_vocabulary)
    
    # Check if query was corrected
    if corrected_query != query_lower:
        # Only log this during development
        # In production, you might want to show this to the user
        print(f"Corrected query: '{query_lower}' -> '{corrected_query}'")
    
    # Common intent phrases to filter out
    intent_phrases = [
        "i want", "i need", "looking for", "searching for", "where can i find", 
        "i would like", "can i get", "do you have", "show me", "buy", "purchase",
        "to buy", "to get", "to find", "to purchase"
    ]
    
    # Remove intent phrases to focus on product terms
    cleaned_query = corrected_query
    for phrase in intent_phrases:
        if phrase in cleaned_query:
            cleaned_query = cleaned_query.replace(phrase, "")
    
    # Strip extra spaces and articles
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
    cleaned_query = re.sub(r'\b(a|an|the)\b', '', cleaned_query).strip()
    
    # Extract product keywords - remove common filler words
    filler_words = ['product', 'products', 'item', 'items', 'some']
    product_terms = []
    for word in cleaned_query.split():
        if word not in filler_words:
            product_terms.append(word)
    
    product_term = ' '.join(product_terms)
    
    # If product term is empty after cleaning, use original query
    if not product_term:
        product_term = corrected_query
    
    # Filter out rows with empty product_name before processing
    df = df[df['product_name'].notna() & (df['product_name'] != '')].copy()
    
    # Preprocess the dataframe columns for search
    df['search_text'] = df.apply(lambda row: 
                               (row['product_name'] if pd.notna(row['product_name']) else '') + ' ' + 
                               (row['category'] if pd.notna(row['category']) else '') + ' ' + 
                               (row['description'] if pd.notna(row['description']) else ''),
                               axis=1)
    
    # Clean the query and product texts
    clean_query = re.sub(r'[^\w\s\'"\-]', ' ', product_term)
    
    # Handle singular/plural forms
    product_term_lower = clean_query.lower()
    if product_term_lower.endswith('s'):
        singular_form = product_term_lower[:-1]  # Remove 's' for singular form
        expanded_query = f"{product_term_lower} {singular_form}"  # Include both forms
    else:
        plural_form = product_term_lower + 's'  # Add 's' for plural form
        expanded_query = f"{product_term_lower} {plural_form}"  # Include both forms
    
    # Use the expanded query for processing
    clean_query = expanded_query
    
    # Get unique product texts
    product_texts = df['search_text'].tolist()
    
    # Create a combined product name field that uses service_name as fallback
    df['display_name'] = df.apply(lambda row: 
                                row['product_name'] if pd.notna(row['product_name']) 
                                else 'Unknown', axis=1)
    
    # First handle direct category mapping for any product category
    known_category_matches = []
    product_term_words = product_term.split()
    
    # Check if any of the words in the product term match our categories
    matching_categories = []
    for word in product_term_words:
        # Use fuzzy matching for category identification
        best_category_match = None
        best_score = 0
        
        for category in expanded_mappings:
            score = fuzz.ratio(word.lower(), category.lower())
            if score > 80 and score > best_score:  # High threshold for categories
                best_score = score
                best_category_match = category
        
        if best_category_match:
            matching_categories.append(best_category_match)
    
    # If we have matching categories, prioritize products from those categories
    if matching_categories:
        for category in matching_categories:
            related_products = expanded_mappings[category]
            
            # Find products that match any of the related terms
            for i, row in df.iterrows():
                product_name = row['display_name'].lower()
                search_text = row['search_text'].lower()
                
                # Direct category match in the product name gets highest priority
                if category.lower() in product_name:
                    known_category_matches.append((row['display_name'], 1.0))
                    continue
                
                # Check if any related terms appear in the product name with fuzzy matching
                for related_term in related_products:
                    # Direct match
                    if related_term.lower() in product_name:
                        known_category_matches.append((row['display_name'], 0.9))
                        break
                    
                    # Fuzzy match for product names
                    if fuzz.partial_ratio(related_term.lower(), product_name) > 85:
                        known_category_matches.append((row['display_name'], 0.85))
                        break
                
                # If not found in product name, check entire search text
                if not any(related_term.lower() in product_name for related_term in related_products):
                    for related_term in related_products:
                        if related_term.lower() in search_text:
                            known_category_matches.append((row['display_name'], 0.7))
                            break
                        
                        # Fuzzy match for search text
                        if fuzz.partial_ratio(related_term.lower(), search_text) > 80:
                            known_category_matches.append((row['display_name'], 0.65))
                            break
    
    # Check for direct matches even if not in category mappings
    # This handles specific product searches with fuzzy matching
    if product_term and not known_category_matches:
        for i, row in df.iterrows():
            product_name = row['display_name'].lower()
            search_text = row['search_text'].lower()
            
            # Exact match in product name
            if product_term.lower() in product_name:
                known_category_matches.append((row['display_name'], 0.95))
                continue
            
            # Fuzzy match for whole product term
            if fuzz.partial_ratio(product_term.lower(), product_name) > 80:
                known_category_matches.append((row['display_name'], 0.9))
                continue
                
            # Check for all words appearing in product name with fuzzy matching
            all_words_match = True
            for word in product_term_words:
                # Check if any word in product name is similar to the query word
                max_word_score = 0
                for prod_word in product_name.split():
                    word_score = fuzz.ratio(word.lower(), prod_word.lower())
                    max_word_score = max(max_word_score, word_score)
                
                if max_word_score < 75:  # Word not found in product name with sufficient similarity
                    all_words_match = False
                    break
            
            if all_words_match:
                known_category_matches.append((row['display_name'], 0.85))
                continue
                
            # Check for all words appearing in search text
            if all(word in search_text for word in product_term_words):
                known_category_matches.append((row['display_name'], 0.8))
                continue
            
            # Fuzzy match for words in search text
            all_words_match_fuzzy = True
            for word in product_term_words:
                # Check if any word in search text is similar to the query word
                max_word_score = 0
                for search_word in search_text.split():
                    word_score = fuzz.ratio(word.lower(), search_word.lower())
                    max_word_score = max(max_word_score, word_score)
                
                if max_word_score < 70:  # Word not found in search text with sufficient similarity
                    all_words_match_fuzzy = False
                    break
            
            if all_words_match_fuzzy:
                known_category_matches.append((row['display_name'], 0.75))
                continue
    
    # If we have category or direct matches, sort by score and return top results
    if known_category_matches:
        # Remove duplicates while preserving order
        unique_results = []
        seen_products = set()
        for product, score in sorted(known_category_matches, key=lambda x: x[1], reverse=True):
            if product not in seen_products:
                unique_results.append((product, score))
                seen_products.add(product)
                if len(unique_results) >= top_n:
                    break
        
        if unique_results:
            return unique_results[:top_n]
    
    # If no matches found through direct matching, fall back to TF-IDF
    # Define the StemmedTfidfVectorizer
    class StemmedTfidfVectorizer(TfidfVectorizer):
        def build_analyzer(self):
            analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
            stemmer = PorterStemmer()
            return lambda doc: [stemmer.stem(word) for word in analyzer(doc)]
    
    # Use the cleaned query that focuses on product terms
    clean_query = re.sub(r'[^\w\s\'"\-]', ' ', product_term)
    
    # Get unique product texts
    product_texts = df['search_text'].tolist()
    product_names = df['display_name'].tolist()
    
    # Use TF-IDF with stemming
    vectorizer = StemmedTfidfVectorizer(
        analyzer='word',
        token_pattern=r'(?u)\b\w+[\'"\-\w]*\b|\d+[\'"\"]',
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(product_texts)
    
    # Transform the query using the same vectorizer
    query_vector = vectorizer.transform([clean_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get indices sorted by similarity score in descending order
    sorted_indices = np.argsort(cosine_similarities)[::-1]
    
    # Create a list of tuples (product_name, similarity_score)
    results = []
    seen_products = set()
    
    for i in sorted_indices:
        if cosine_similarities[i] > 0.1:  # Lower threshold for fallback
            product_name = product_names[i]
            
            # Skip if we've already seen this product
            if product_name in seen_products:
                continue
            
            results.append((product_name, cosine_similarities[i]))
            seen_products.add(product_name)
            
            if len(results) >= top_n:
                break
    
    # If still no results, use fuzzy matching as last resort
    if not results:
        fuzzy_results = []
        for product_name in product_names:
            # Calculate fuzzy match score
            score = fuzz.token_sort_ratio(clean_query, product_name.lower())
            if score > 60:  # Only consider reasonably good matches
                if product_name not in seen_products:
                    fuzzy_results.append((product_name, score / 100))  # Normalize score to 0-1 range
                    seen_products.add(product_name)
        
        # Sort by score and return top matches
        if fuzzy_results:
            results = sorted(fuzzy_results, key=lambda x: x[1], reverse=True)[:top_n]
    
    return results

# Find and analyze suppliers
def find_and_analyze_suppliers(df, product_name, top_n=5):
    product_name = product_name.strip().upper()
    
    # Try exact match first
    matched_df = df[df['product_name'] == product_name].copy()
    
    # If no exact match, try partial match
    if matched_df.empty:
        matched_df = df[df['product_name'].str.contains(product_name, na=False)].copy()
        if not matched_df.empty:
            st.info(f"No exact match found for '{product_name}'. Showing partial matches.")
    
    if matched_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Store raw data for display
    all_products = matched_df.copy()
    
    # Aggregate by supplier
    aggregated = []
    
    # Group by supplier
    supplier_groups = matched_df.groupby('supplier_name')
    
    for supplier, group in supplier_groups:
        # Basic aggregation
        total_quantity = group['quantity'].sum()
        avg_price_usd = group['unit_price_usd'].mean()
        avg_lead_time = group['lead_time'].mean()
        avg_shipping = group['shipping_charges_usd'].mean()
        product_count = len(group)
        
        # Calculate original currency values
        original_currency = group['original_currency'].mode().iloc[0]  # Most frequent currency
        avg_price_original = group['original_unit_price'].mean()
        
        # Get the most common measurement unit
        unit_of_measure = group['unit_of_measure'].mode().iloc[0] if not group['unit_of_measure'].mode().empty else ''
        
        # Calculate price consistency
        price_std = group['unit_price_usd'].std()
        price_consistency = 1 - (price_std / avg_price_usd if avg_price_usd > 0 else 0)
        price_consistency = max(0, min(1, price_consistency))  # Clamp between 0 and 1
        
        # Collect delivery terms
        delivery_terms = group['delivery_terms'].mode().iloc[0] if not group['delivery_terms'].mode().empty else ''
        
        # Store in aggregated results
        aggregated.append({
            'Supplier Name': supplier,
            'Supplier ID': group['supplier_id'].iloc[0],
            'Total Quantity': total_quantity,
            'Unit of Measure': unit_of_measure,
            'Avg Price (USD)': avg_price_usd,
            'Avg Price (Original)': avg_price_original,
            'Original Currency': original_currency,
            'Avg Lead Time (days)': avg_lead_time,
            'Avg Shipping (USD)': avg_shipping,
            'Product Count': product_count,
            'Price Consistency': price_consistency * 100,  # Convert to percentage
            'Delivery Terms': delivery_terms,
            'Discount(%)': group['discount_value'].mean() * 100  # Convert to percentage
        })
    
    # Convert to DataFrame
    agg_df = pd.DataFrame(aggregated)
    
    # Calculate scores for comparison across suppliers
    if not agg_df.empty and len(agg_df) > 0:
        # Base metrics normalization
        price_min = agg_df['Avg Price (USD)'].min() if not agg_df['Avg Price (USD)'].empty else 0  # Lower price is better
        lead_time_min = agg_df['Avg Lead Time (days)'].min() if not agg_df['Avg Lead Time (days)'].empty else 0  # Lower days is better
        shipping_min = agg_df['Avg Shipping (USD)'].min() if not agg_df['Avg Shipping (USD)'].empty else 0  # Lower shipping is better
        
        # Price score - lower is better
        if price_min > 0:
            agg_df['Price Score'] = price_min / agg_df['Avg Price (USD)']
        else:
            agg_df['Price Score'] = 0
            
        # Lead time score - lower is better
        if lead_time_min > 0:
            agg_df['Lead Time Score'] = lead_time_min / agg_df['Avg Lead Time (days)']
        else:
            agg_df['Lead Time Score'] = 1  # Perfect score if zero days
            
        # Shipping score - lower is better
        if shipping_min > 0:
            agg_df['Shipping Score'] = shipping_min / agg_df['Avg Shipping (USD)']
        else:
            agg_df['Shipping Score'] = 1  # Perfect score if zero shipping
        
        # Consistency score
        agg_df['Consistency Score'] = agg_df['Price Consistency'] / 100  # Convert back to 0-1 scale
        
        # Discount score - higher is better
        max_discount = agg_df['Discount(%)'].max()
        if max_discount > 0:
            agg_df['Discount Score'] = agg_df['Discount(%)'] / max_discount
        else:
            agg_df['Discount Score'] = 0
        
        # Calculate weighted final score
        agg_df['Score(%)'] = ((agg_df['Price Score'] * 0.35 +
                          agg_df['Lead Time Score'] * 0.25 +
                          agg_df['Shipping Score'] * 0.20 +
                          agg_df['Consistency Score'] * 0.10 +
                          agg_df['Discount Score'] * 0.10) * 100).clip(0, 100)
        
        # Round score to 2 decimal places
        agg_df['Score(%)'] = agg_df['Score(%)'].round(2)
    
    # Return top n suppliers by score
    if not agg_df.empty and 'Score(%)' in agg_df.columns:
        return agg_df.nlargest(top_n, 'Score(%)'), all_products
    else:
        return agg_df, all_products

def generate_supplier_insights(supplier_data, all_suppliers_data):
    """Generate insights about a specific supplier compared to others"""
    insights = {}
    
    # Extract data
    supplier_name = supplier_data['Supplier Name']
    avg_price = supplier_data['Avg Price (USD)']
    lead_time = supplier_data['Avg Lead Time (days)']
    shipping = supplier_data['Avg Shipping (USD)']
    price_consistency = supplier_data['Price Consistency']
    discount = supplier_data['Discount(%)']
    delivery_terms = supplier_data['Delivery Terms']
    
    # Calculate averages across all suppliers
    avg_price_all = all_suppliers_data['Avg Price (USD)'].mean()
    avg_lead_time_all = all_suppliers_data['Avg Lead Time (days)'].mean()
    avg_shipping_all = all_suppliers_data['Avg Shipping (USD)'].mean()
    
    # Price insight
    price_diff_pct = ((avg_price / avg_price_all) - 1) * 100 if avg_price_all > 0 else 0
    if abs(price_diff_pct) < 3:
        insights['price'] = f"This supplier offers purchase competitive pricing at around ${avg_price:.2f}."
    elif price_diff_pct < 0:
        insights['price'] = f"This supplier offers pricing {abs(price_diff_pct):.1f}% lower than the purchase average of ${avg_price_all:.2f}."
    else:
        insights['price'] = f"This supplier's pricing is {price_diff_pct:.1f}% higher than the purchase average of ${avg_price_all:.2f}."
    
    # Lead time insight
    lead_diff = lead_time - avg_lead_time_all
    if abs(lead_diff) < 1:
        insights['lead_time'] = f"Lead time of {lead_time:.1f} days is on par with lead time average."
    elif lead_diff < 0:
        insights['lead_time'] = f"Lead time of {lead_time:.1f} days is {abs(lead_diff):.1f} days faster than lead time average."
    else:
        insights['lead_time'] = f"Lead time of {lead_time:.1f} days is {lead_diff:.1f} days slower than lead time average."
    
    # Shipping insight
    shipping_diff_pct = ((shipping / avg_shipping_all) - 1) * 100 if avg_shipping_all > 0 else 0
    if abs(shipping_diff_pct) < 5:
        insights['shipping'] = f"Shipping charges of ${shipping:.2f} are comparable to shipping average."
    elif shipping_diff_pct < 0:
        insights['shipping'] = f"Shipping charges are {abs(shipping_diff_pct):.1f}% lower than shipping average of ${avg_shipping_all:.2f}."
    else:
        insights['shipping'] = f"Shipping charges are {shipping_diff_pct:.1f}% higher than shipping average of ${avg_shipping_all:.2f}."
    
    # Consistency insight
    if price_consistency > 90:
        insights['consistency'] = "Excellent price consistency across products, indicating reliable pricing."
    elif price_consistency > 75:
        insights['consistency'] = "Good price consistency across products."
    else:
        insights['consistency'] = "Variable pricing across products, may require negotiation."
    
    # Discount insight
    if discount > 15:
        insights['discount'] = f"Offers excellent discounts of {discount:.1f}%."
    elif discount > 5:
        insights['discount'] = f"Offers standard market discounts of {discount:.1f}%."
    else:
        insights['discount'] = f"Limited discounts of only {discount:.1f}%."
    
    # Delivery terms insight
    insights['delivery'] = f"Standard delivery terms: {delivery_terms}"
    
    return insights


# Set up the Streamlit app
def main():
    st.title("Catalog Recommendation Agent")
    
    # Input for CSV file path
    default_path = r"Catalogs2.csv"
    
    # Load and preprocess data
    df = load_data(default_path)
    if df.empty:
        st.error("Failed to load data. Please check your file path.")
        return
    
    processed_df = preprocess_data(df)
    
    # Sidebar for search
    st.header("Search Products")
    search_query = st.text_input("Enter search terms:")

    # Reset selected product if a new search query is entered
    if search_query and 'selected_product' in st.session_state:
        if st.session_state.get('last_search_query', '') != search_query:
            del st.session_state.selected_product
            if 'selected_supplier' in st.session_state:
                del st.session_state.selected_supplier
    st.session_state.last_search_query = search_query
    
    if search_query:
        # Find similar products
        similar_products = find_similar_products(processed_df, search_query)
        
        if not similar_products:
            st.warning(f"No products found matching '{search_query}'.")
        else:
            st.header(f"Products matching: '{search_query}'")
            
            # Display matched products
            st.subheader("Matching Products")
            for i, (product, _) in enumerate(similar_products):
                if st.button(f"{product}", key=f"product_button_{i}"):
                    st.session_state.selected_product = product
                    if 'selected_supplier' in st.session_state:
                        del st.session_state.selected_supplier
            
            # If a product is selected, show analysis
            if 'selected_product' in st.session_state:
                selected_product = st.session_state.selected_product
                st.header(f"Analysis for: {selected_product}")
                
                # Get supplier data for the selected product
                supplier_data, all_products = find_and_analyze_suppliers(processed_df, selected_product)
                
                if supplier_data.empty:
                    st.warning(f"No suppliers found for {selected_product}.")
                else:
                    # Display supplier comparison
                    st.subheader("Key Matrics Comparison")
                    display_columns = ['Supplier Name', 'Avg Price (USD)', 'Avg Lead Time (days)', 
                                      'Avg Shipping (USD)', 'Price Consistency', 'Discount(%)', 'Score(%)']
                    st.dataframe(supplier_data[display_columns],hide_index=True)
                    
                    # Allow user to select a supplier from dropdown, default to top supplier
                    supplier_list = supplier_data['Supplier Name'].tolist()
                    selected_supplier_name = st.selectbox(
                        "Select a supplier for detailed insights:", 
                        supplier_list,
                        index=0  # Default to top supplier (index 0)
                    )
                    
                    # Store the selected supplier in session state
                    if 'selected_supplier' not in st.session_state or st.session_state.selected_supplier != selected_supplier_name:
                        st.session_state.selected_supplier = selected_supplier_name
                    
                    # Get the selected supplier data
                    selected_supplier = supplier_data[supplier_data['Supplier Name'] == selected_supplier_name].iloc[0]
                    
                    st.subheader(f"Supplier Insights: {selected_supplier_name}")
                    
                    # Generate and display insights
                    insights = generate_supplier_insights(selected_supplier, supplier_data)
                    
                    for key, insight in insights.items():
                        st.write(f"• {insight}")
                    
                    # Show all products from this supplier
                    supplier_products = all_products[all_products['supplier_name'] == selected_supplier_name]
                    
                    # Display all columns for the selected supplier
                    
                    columns_to_display=[
                        'supplier_name', 'supplier_id', 'product_name', 'parent_category', 'category', 
                        'description', 'catalog_id', 'unit_price', 'quantity', 'unit_of_measure', 'lead_time', 
                        'currency', 'specifications.dimensions', 'specifications.manufacturer', 
                        'specifications.weight.value', 'specifications.weight.type', 'specifications.color', 
                        'delivery_terms', 'discounts', 'shipping_charges', 'tax', 'vendor_part_number', 
                        'category_type', 'sku_id', 'service_name', 'route', 'equipment', 'distance_miles', 
                        'base_rate_per_mile', 'fuel_surcharge', 'detention_rate_per_hour', 'liftgate_service_rate', 
                        'special_instructions', 'additional_terms_and_conditions', 'additional_services'
                    ]

                    # Filter out columns where all values are missing (NaN)
                    available_columns = [col for col in columns_to_display if not supplier_products[col].isna().all()]

                    # Display only the available columns
                    st.subheader(f"Catalog Details for {selected_supplier_name}")
                    st.dataframe(supplier_products[available_columns],hide_index=True)

# Run the app
if __name__ == "__main__":
    main()
