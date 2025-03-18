import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os

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

# Find similar products using TF-IDF + Cosine Similarity
def find_similar_products(df, query, top_n=5):
    """
    Find products similar to the user's query using TF-IDF + Cosine Similarity.
    """
    # Combine product name and description for better matching
    df['search_text'] = df['product_name'] + ' ' + df['description']
    
    # Get unique product descriptions
    product_texts = df['search_text'].unique()
    product_names = df['product_name'].unique()
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the product texts
    tfidf_matrix = vectorizer.fit_transform(product_texts)
    
    # Transform the query
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity between query and product descriptions
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get the indices of the top N most similar products
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    
    # Return the matched products and their similarity scores
    matches = [(product_texts[i], cosine_similarities[i]) for i in top_indices]
    
    # Map back to product names
    product_matches = []
    for text, score in matches:
        for name in product_names:
            if name in text:
                product_matches.append((name, score))
                break
    
    return product_matches

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
            'Discount': group['discount_value'].mean() * 100  # Convert to percentage
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
        max_discount = agg_df['Discount'].max()
        if max_discount > 0:
            agg_df['Discount Score'] = agg_df['Discount'] / max_discount
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
    discount = supplier_data['Discount']
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
        insights['lead_time'] = f"Lead time of {lead_time:.1f} days is on per with industry average."
    elif lead_diff < 0:
        insights['lead_time'] = f"Lead time of {lead_time:.1f} days is {abs(lead_diff):.1f} days faster than industry average."
    else:
        insights['lead_time'] = f"Lead time of {lead_time:.1f} days is {lead_diff:.1f} days slower than industry average."
    
    # Shipping insight
    shipping_diff_pct = ((shipping / avg_shipping_all) - 1) * 100 if avg_shipping_all > 0 else 0
    if abs(shipping_diff_pct) < 5:
        insights['shipping'] = f"Shipping charges of ${shipping:.2f} are comparable to purchase average."
    elif shipping_diff_pct < 0:
        insights['shipping'] = f"Shipping charges are {abs(shipping_diff_pct):.1f}% lower than purchase average of ${avg_shipping_all:.2f}."
    else:
        insights['shipping'] = f"Shipping charges are {shipping_diff_pct:.1f}% higher than purchase average of ${avg_shipping_all:.2f}."
    
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
    st.title("Supplier Analyzer")
    
    # Input for CSV file path
    default_path = "C:\\Users\\shaik\\Downloads\\Catalogs2.csv"
    file_path = st.text_input("Enter the path to the CSV file:", default_path)
    
    if not file_path:
        st.warning("Please enter a file path.")
        return
    
    # Load and preprocess data
    df = load_data(file_path)
    if df.empty:
        st.error("Failed to load data. Please check your file path.")
        return
    
    processed_df = preprocess_data(df)
    
    # Sidebar for search
    st.sidebar.header("Search Products")
    search_query = st.sidebar.text_input("Enter search terms:")
    
    if search_query:
        # Find similar products
        similar_products = find_similar_products(processed_df, search_query)
        
        if not similar_products:
            st.warning(f"No products found matching '{search_query}'.")
        else:
            st.header(f"Products matching: '{search_query}'")
            
            # Display matched products
            st.subheader("Matching Products")
            for i, (product, score) in enumerate(similar_products):
                if st.button(f"{product} (Score: {score:.2f})", key=f"product_button_{i}"):
                    st.session_state.selected_product = product
            
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
                    st.subheader("Supplier Comparison")
                    display_columns = ['Supplier Name', 'Avg Price (USD)', 'Avg Lead Time (days)', 
                                      'Avg Shipping (USD)', 'Price Consistency', 'Discount', 'Score(%)']
                    st.dataframe(supplier_data[display_columns])
                    
                    # Allow user to select a supplier from dropdown, default to top supplier
                    supplier_list = supplier_data['Supplier Name'].tolist()
                    selected_supplier_name = st.selectbox(
                        "Select a supplier for detailed insights:", 
                        supplier_list,
                        index=0  # Default to top supplier (index 0)
                    )
                    
                    # Get the selected supplier data
                    selected_supplier = supplier_data[supplier_data['Supplier Name'] == selected_supplier_name].iloc[0]
                    
                    st.subheader(f"Supplier Insights: {selected_supplier_name}")
                    
                    # Generate and display insights
                    insights = generate_supplier_insights(selected_supplier, supplier_data)
                    
                    for key, insight in insights.items():
                        st.write(f"• {insight}")
                    
                    # Show all products from this supplier
                    #st.subheader(f"Products from {selected_supplier_name}")
                    supplier_products = all_products[all_products['supplier_name'] == selected_supplier_name]
                    
                    # Display all columns for the selected supplier
                    st.subheader(f"Full Details for {selected_supplier_name}")
                    st.dataframe(supplier_products[[
                        '_id', 'supplier_name', 'supplier_id', 'product_name', 'parent_category', 'category', 
                        'description', 'catalog_id', 'unit_price', 'quantity', 'unit_of_measure', 'lead_time', 
                        'currency', 'specifications.dimensions', 'specifications.manufacturer', 
                        'specifications.weight.value', 'specifications.weight.type', 'specifications.color', 
                        'delivery_terms', 'discounts', 'shipping_charges', 'tax', 'vendor_part_number', 
                        'category_type', 'sku_id', 'service_name', 'route', 'equipment', 'distance_miles', 
                        'base_rate_per_mile', 'fuel_surcharge', 'detention_rate_per_hour', 'liftgate_service_rate', 
                        'special_instructions', 'additional_terms_and_conditions', 'additional_services'
                    ]])

# Run the app
if __name__ == "__main__":
    main()