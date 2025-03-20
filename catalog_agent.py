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
    df['discount_value'] = df['discounts'].str.replace('%', '').astype(float) / 100
    
    # Calculate total price (unit price + shipping - discount)
    df['total_price_usd'] = df['unit_price_usd'] + df['shipping_charges_usd']
    df['total_price_usd'] = df['total_price_usd'] * (1 - df['discount_value'])
    
    # Keep original currency for reference
    df['original_currency'] = df['currency']
    df['original_unit_price'] = df['unit_price']
    
    return df

# Define the stemmed vectorizer first
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        stemmer = PorterStemmer()
        return lambda doc: [stemmer.stem(word) for word in analyzer(doc)]

def find_similar_products(df, query, top_n=5):
    """
    Find products similar to the user's query using TF-IDF + Cosine Similarity.
    """
    # Filter out rows with empty product_name before processing
    df = df[df['product_name'].notna() & (df['product_name'] != '')].copy()
    
    # Preprocess the dataframe columns for search - handle empty product_name
    df['search_text'] = df.apply(lambda row: 
                                 (row['product_name'] if pd.notna(row['product_name']) else '') + ' ' + 
                                 (row['description'] if pd.notna(row['description']) else '') ,
                                 axis=1)
    
    # Clean the query and product texts
    clean_query = re.sub(r'[^\w\s\'"\-]', ' ', query)
    
    # Handle singular/plural forms
    query_lower = clean_query.lower()
    if query_lower.endswith('s'):
        singular_form = query_lower[:-1]  # Remove 's' for singular form
        expanded_query = f"{query_lower} {singular_form}"  # Include both forms
    else:
        plural_form = query_lower + 's'  # Add 's' for plural form
        expanded_query = f"{query_lower} {plural_form}"  # Include both forms
    
    # Use the expanded query for processing
    clean_query = expanded_query
    
    # Get unique product texts
    product_texts = df['search_text'].tolist()
    
    # Create a combined product name field that uses service_name as fallback
    df['display_name'] = df.apply(lambda row: 
                                 row['product_name'] if pd.notna(row['product_name']) 
                                 else 'Unknown', axis=1)
    product_names = df['display_name'].tolist()
    
    # Extract key terms from query for exact phrase matching
    query_terms = clean_query.strip().lower().split()
    query_length = len(query_terms)
    
    # First try to match the exact phrase with higher priority
    exact_matches = []
    partial_phrase_matches = []
    
    # Original query for exact matches
    original_query = query.lower()
    
    # Look for exact phrase matches and consecutive word matches first
    for i, text in enumerate(product_texts):
        text_lower = text.lower()
        product_name_lower = product_names[i].lower()
        
        # Skip entries without a product name or service name
        if product_name_lower == 'unknown':
            continue
            
        # Check for exact phrase match in product name (highest priority)
        if original_query in product_name_lower:
            exact_matches.append((product_names[i], 1.0))
            continue
            
        # Check for exact phrase match in search text (high priority)
        if original_query in text_lower:
            exact_matches.append((product_names[i], 0.9))
            continue
            
        # Check for all terms appearing in the product name, even if not consecutive
        if query_length > 1 and all(term in product_name_lower for term in query_terms):
            partial_phrase_matches.append((product_names[i], 0.8))
            continue
            
        # Check for most terms appearing in the product name (more flexible)
        if query_length > 2:  # Only for queries with 3+ words
            matches = sum(1 for term in query_terms if term in product_name_lower)
            if matches >= query_length - 1:  # Allow missing one term
                partial_phrase_matches.append((product_names[i], 0.7))
                continue
                
        # Check for most terms appearing in the search text
        if query_length > 2:  # Only for queries with 3+ words
            matches = sum(1 for term in query_terms if term in text_lower)
            if matches >= query_length - 1:  # Allow missing one term
                partial_phrase_matches.append((product_names[i], 0.6))
                continue
    
    # If we have exact or good partial matches, return those
    if exact_matches or partial_phrase_matches:
        # Remove duplicates while preserving order
        unique_results = []
        seen_products = set()
        for product, score in exact_matches + partial_phrase_matches:
            if product not in seen_products:
                unique_results.append((product, score))
                seen_products.add(product)
                
                if len(unique_results) >= top_n:
                    break
                    
        return unique_results[:top_n]
    
    # Fall back to TF-IDF + semantic context scoring if no exact matches
    vectorizer = StemmedTfidfVectorizer(
        analyzer='word',
        token_pattern=r'(?u)\b\w+[\'"\-\w]*\b|\d+[\'"\"]',
        ngram_range=(1, 2),  # Consider both unigrams and bigrams
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
    seen_products = set()  # To avoid duplicates
    
    # Relaxed threshold for longer queries
    min_similarity_threshold = 0.2 if query_length <= 2 else 0.15
    
    # Add semantic context filtering to prevent partial word matches from irrelevant categories
    for i in sorted_indices:
        if cosine_similarities[i] > min_similarity_threshold:  # Lower threshold for longer queries
            product_name = product_names[i]
            
            # Skip if we've already seen this product
            if product_name in seen_products:
                continue
                
            product_text = product_texts[i].lower()
            
            if query_length > 1:
                # Enhanced check for multi-word queries to avoid cross-category matches
                query_match_count = sum(1 for term in query_terms if term in product_text)
                
                # Relaxed matching criteria for longer queries
                min_match_ratio = 0.7 if query_length <= 2 else 0.6
                
                # Only include if most query terms appear, or the similarity is very high
                if query_match_count / query_length >= min_match_ratio or cosine_similarities[i] > 0.5:
                    # Check if the product name or description suggests a different category
                    if not all(term in product_text for term in query_terms) and cosine_similarities[i] < 0.6:
                        # Apply a penalty to similarity score
                        adjusted_score = cosine_similarities[i] * 0.7  # Less penalty
                    else:
                        adjusted_score = cosine_similarities[i]
                    
                    results.append((product_name, adjusted_score))
                    seen_products.add(product_name)
            else:
                # For single word queries, still apply some filtering to avoid cross-category matches
                results.append((product_name, cosine_similarities[i]))
                seen_products.add(product_name)
            
            # Stop when we have enough results
            if len(results) >= top_n:
                break
    
    # If still no results found, use direct substring match as fallback
    if not results:
        # First try inch pattern
        inch_pattern = r'(\d+)["\'"]'
        query_has_inches = re.search(inch_pattern, query)
        
        if query_has_inches:
            # Extract the numeric part
            inch_value = query_has_inches.group(1)
            
            # Look for products containing this inch value with quotes
            for i, name in enumerate(product_names):
                if f"{inch_value}\"" in name or f"{inch_value} inch" in name.lower():
                    if name not in seen_products:
                        results.append((name, 0.5))
                        seen_products.add(name)
                        if len(results) >= top_n:
                            break
        
        # Then try simple substring match for individual words
        elif not results:
            for i, name in enumerate(product_names):
                name_lower = name.lower()
                # Check if any of the query terms appear in the name
                if any(term in name_lower for term in query_terms):
                    if name not in seen_products:
                        results.append((name, 0.4))
                        seen_products.add(name)
                        if len(results) >= top_n:
                            break
    
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
