import streamlit as st
import time
import logging
from datetime import datetime
from typing import Dict, Any, List
import plotly.graph_objects as go
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Gov Data Q&A Chatbot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .suggestion-box {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Gemini setup helpers ---
_GEMINI_MODEL = None

def _init_gemini_model():
    global _GEMINI_MODEL
    if _GEMINI_MODEL is not None:
        return _GEMINI_MODEL
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        _GEMINI_MODEL = genai.GenerativeModel('gemini-2.5-flash')
        return _GEMINI_MODEL
    except Exception:
        return None

def create_dynamic_chart(x_data, y_data, user_query, x_title, y_title):
    """Create dynamic chart based on query context"""
    fig = go.Figure()
    
    # Determine chart type based on query keywords
    query_lower = user_query.lower()
    
    if any(word in query_lower for word in ['trend', 'over time', 'yearly', 'monthly', 'progression']):
        # Line chart for trends
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            name=y_title,
            line=dict(width=3)
        ))
        chart_title = f"{y_title} Trends"
    elif any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus']):
        # Bar chart for comparisons
        fig.add_trace(go.Bar(
            x=x_data,
            y=y_data,
            name=y_title,
            marker_color='lightblue'
        ))
        chart_title = f"{y_title} Comparison"
    elif any(word in query_lower for word in ['distribution', 'share', 'percentage', 'proportion']):
        # Pie chart for distributions
        fig.add_trace(go.Pie(
            labels=x_data,
            values=y_data,
            name=y_title
        ))
        chart_title = f"{y_title} Distribution"
    else:
        # Default bar chart
        fig.add_trace(go.Bar(
            x=x_data,
            y=y_data,
            name=y_title,
            marker_color='lightblue'
        ))
        chart_title = f"{y_title} Analysis"
    
    fig.update_layout(
        title=chart_title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=500,
        showlegend=True
    )
    
    return fig

def parse_query(user_query: str) -> dict:
    """Very lightweight intent extraction for commodity queries.
    Returns keys: mode in {top, compare, trend}, n (int), commodities (list[str])."""
    text = (user_query or "").lower()
    result = {"mode": "top", "n": 5, "commodities": []}
    # top-N
    import re
    m = re.search(r"top\s+(\d+)", text)
    if m:
        try:
            result["n"] = max(1, min(10, int(m.group(1))))
            result["mode"] = "top"
        except ValueError:
            pass
    # trend
    if any(k in text for k in ["trend", "over time", "monthly", "yearly"]):
        result["mode"] = "trend"
    # compare
    if any(k in text for k in ["compare", "vs", "versus"]):
        result["mode"] = "compare"
    # crude commodity name capture (split by commas and 'and')
    if any(k in text for k in [",", " and "]):
        # extract words between separators
        candidates = re.split(r",| and ", text)
        # keep tokens that look like commodity words (letters/spaces)
        picks = []
        for token in candidates:
            t = token.strip()
            if any(w in t for w in ["ginger", "onion", "potato", "ragi", "rajma", "guava", "rice", "wheat", "maize", "paddy"]):
                picks.append(t)
        if picks:
            result["commodities"] = list(dict.fromkeys(picks))[:5]
    return result

def generate_gemini_summary(user_query: str, data_summary: dict, citations: list) -> tuple[str | None, str | None]:
    model = _init_gemini_model()
    if model is None:
        return None, None
    try:
        prompt = (
            "You are a data analyst. Answer the user's question using the provided data summary. "
            "Be concise, include key numbers, trends, and comparisons. If applicable, mention sources.\n\n"
            f"User question: {user_query}\n\n"
            f"Data summary (JSON): {data_summary}\n\n"
            f"Citations: {citations}\n\n"
            "Return a clear, well-structured paragraph followed by 3-5 bullet points of key facts.\n\n"
            "Also suggest the best chart type for this query from: bar, line, pie, scatter. "
            "Respond with format: CHART_TYPE: [type] at the end."
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, 'text', None)
        if not text:
            return None, None
        
        # Extract chart type suggestion
        chart_type = None
        if "CHART_TYPE:" in text:
            chart_type = text.split("CHART_TYPE:")[-1].strip().lower()
            text = text.split("CHART_TYPE:")[0].strip()
        
        return text.strip(), chart_type
    except Exception:
        return None, None

# Dataset configuration
DATASETS = {
    'rainfall': {
        'name': 'Area-weighted Monthly Rainfall Data',
        'description': 'Monthly, seasonal and annual rainfall data from 1901',
        'url': 'https://www.data.gov.in/resource/area-weighted-monthly-seasonal-and-annual-rainfall-mm-36-meteorological-subdivisions-1901',
        'resource_id': 'abfd2d50-0d73-4a3e-9027-10edb3d21940',  # Working resource ID for testing
        'fields': ['commodities', 'weight', 'apr_11', 'may_11', 'jun_11', 'jul_11', 'aug_11', 'sep_11', 'oct_11', 'nov_11', 'dec_11'],
        'ministry': 'Ministry of Commerce and Industry'
    },
    'crop_cost': {
        'name': 'Crop Production Cost Data',
        'description': 'Crop-wise all India weighted average cost of production for 23 mandated crops 2018-19',
        'url': 'https://www.data.gov.in/resource/crop-wise-all-india-weighted-average-cost-production-respect-mandated-23-crops-2018-19',
        'resource_id': 'CROP_COST_RESOURCE_ID_HERE',  # Replace with actual UUID
        'fields': ['crop', 'state', 'cost_per_quintal', 'year'],
        'ministry': 'Ministry of Agriculture & Farmers Welfare'
    },
    'vegetables': {
        'name': 'Vegetable Crops Data',
        'description': 'District-wise area, production, yield and value of vegetable crops 2021',
        'url': 'https://www.data.gov.in/resource/district-wise-area-production-yield-value-vegetable-crops-2021',
        'resource_id': 'VEGETABLES_RESOURCE_ID_HERE',  # Replace with actual UUID
        'fields': ['district', 'state', 'crop', 'area', 'production', 'yield'],
        'ministry': 'Ministry of Agriculture & Farmers Welfare'
    }
}

def get_api_key():
    """Get API key from environment"""
    return os.getenv('DATA_GOV_IN_API_KEY')

def fetch_dataset_data(resource_id, limit=100, filters=None):
    """Fetch data from a specific dataset"""
    api_key = get_api_key()
    if not api_key:
        return None, "API key not found"
    
    try:
        url = f"https://api.data.gov.in/resource/{resource_id}"
        params = {
            'api-key': api_key,
            'format': 'json',
            'limit': limit
        }
        
        if filters:
            params.update(filters)
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data, None
        else:
            return None, f"API error: {response.status_code}"
            
    except Exception as e:
        return None, str(e)

def analyze_rainfall_data(data, user_query=""):
    """Analyze rainfall data and create visualizations"""
    if not data or 'records' not in data:
        return None, "No rainfall data available"
    
    records = data['records']
    if not records:
        return None, "No rainfall records found"
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(records)
    
    # Create analysis
    analysis = {
        'total_records': len(records),
        'fields': list(df.columns),
        'sample_data': records[:3] if records else []
    }
    
    # For commodity data (which is what we have working)
    if 'commodities' in df.columns:
        intent = parse_query(user_query)

        # Build monthly numeric columns map
        numeric_cols = [c for c in df.columns if c not in ['commodities', 'weight']]
        # Convert numeric columns to float where possible
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        if intent['mode'] == 'trend':
            # Trend: pick up to 3 commodities and plot monthly lines
            pick_commodities = intent['commodities'][:3]
            if not pick_commodities:
                # choose top by overall mean
                means = df[numeric_cols].mean(axis=1)
                top_idx = means.nlargest(3).index
                pick_commodities = df.loc[top_idx, 'commodities'].tolist()

            fig = go.Figure()
            # Use last year columns if many; otherwise all numeric
            cols_for_trend = numeric_cols[-12:] if len(numeric_cols) >= 12 else numeric_cols
            x_vals = cols_for_trend
            for name in pick_commodities:
                series = df[df['commodities'].str.lower() == name.lower()][cols_for_trend].mean()
                fig.add_trace(go.Scatter(x=x_vals, y=series.values, mode='lines+markers', name=name))
            fig.update_layout(title='Monthly Price Index Trend', xaxis_title='Month', yaxis_title='Price Index', height=500)
            analysis['chart'] = fig

        elif intent['mode'] == 'compare':
            # Compare: bar chart for specified commodities (or top N)
            if intent['commodities']:
                subset = df[df['commodities'].str.lower().isin([c.lower() for c in intent['commodities']])]
            else:
                # top by overall mean
                df['_mean'] = df[numeric_cols].mean(axis=1)
                subset = df.sort_values('_mean', ascending=False).head(intent['n'])
            x_vals = subset['commodities'].tolist()
            y_vals = subset[numeric_cols].mean(axis=1).tolist()
            fig = create_dynamic_chart(x_vals, y_vals, user_query, "Commodity", "Average Price Index")
            analysis['chart'] = fig
            analysis['top_commodities'] = dict(zip(x_vals, y_vals))

        else:  # top mode
            # Top-N commodities by mean index
            df['_mean'] = df[numeric_cols].mean(axis=1)
            top = df.sort_values('_mean', ascending=False).head(intent['n'])
            x_vals = top['commodities'].tolist()
            y_vals = top['_mean'].tolist()
            fig = create_dynamic_chart(x_vals, y_vals, user_query, "Commodity", "Average Price Index")
            analysis['chart'] = fig
            analysis['top_commodities'] = dict(zip(x_vals, y_vals))
    
    return analysis, None

def analyze_crop_cost_data(data):
    """Analyze crop cost data and create visualizations"""
    if not data or 'records' not in data:
        return None, "No crop cost data available"
    
    records = data['records']
    if not records:
        return None, "No crop cost records found"
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    analysis = {
        'total_records': len(records),
        'fields': list(df.columns),
        'sample_data': records[:3] if records else []
    }
    
    # Create visualization
    fig = go.Figure()
    
    # If we have crop and cost data
    if 'crop' in df.columns and 'cost_per_quintal' in df.columns:
        # Group by crop and calculate average cost
        crop_costs = df.groupby('crop')['cost_per_quintal'].mean().sort_values(ascending=False)
        
        fig.add_trace(go.Bar(
            x=crop_costs.index,
            y=crop_costs.values,
            name='Average Cost per Quintal'
        ))
        
        fig.update_layout(
            title="Average Production Cost by Crop",
            xaxis_title="Crop",
            yaxis_title="Cost per Quintal",
            height=500
        )
        
        analysis['chart'] = fig
        analysis['top_crops'] = crop_costs.head(5).to_dict()
    
    return analysis, None

def analyze_vegetable_data(data):
    """Analyze vegetable crop data and create visualizations"""
    if not data or 'records' not in data:
        return None, "No vegetable data available"
    
    records = data['records']
    if not records:
        return None, "No vegetable records found"
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    analysis = {
        'total_records': len(records),
        'fields': list(df.columns),
        'sample_data': records[:3] if records else []
    }
    
    # Create visualization
    fig = go.Figure()
    
    # If we have crop and production data
    if 'crop' in df.columns and 'production' in df.columns:
        # Group by crop and calculate total production
        crop_production = df.groupby('crop')['production'].sum().sort_values(ascending=False)
        
        fig.add_trace(go.Bar(
            x=crop_production.index,
            y=crop_production.values,
            name='Total Production'
        ))
        
        fig.update_layout(
            title="Total Production by Vegetable Crop",
            xaxis_title="Crop",
            yaxis_title="Production",
            height=500
        )
        
        analysis['chart'] = fig
        analysis['top_crops'] = crop_production.head(5).to_dict()
    
    return analysis, None

def display_header():
    """Display the main header"""
    st.markdown('<h1 class="main-header">🌾 Gov Data Q&A Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        Ask questions about Indian agriculture, rainfall, and vegetable crop data from government datasets
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with information and examples"""
    with st.sidebar:
        st.header("📊 Available Datasets")
        for key, dataset in DATASETS.items():
            with st.expander(f"🌾 {dataset['name']}"):
                st.write(f"**Description:** {dataset['description']}")
                st.write(f"**Ministry:** {dataset['ministry']}")
                st.write(f"**Fields:** {', '.join(dataset['fields'])}")
        
        st.header("💡 Example Queries")
        example_queries = [
            "Show me the top 5 commodities by price index",
            "Compare commodity price trends",
            "What is the average price index for all commodities?",
            "List top 3 commodities with highest price index",
            "Show commodity price analysis"
        ]
        
        for i, query in enumerate(example_queries, 1):
            if st.button(f"{i}. {query}", key=f"example_{i}"):
                st.session_state.user_input = query
                st.rerun()
        
        st.header("🔧 Setup Status")
        api_key = get_api_key()
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.error("❌ API Key missing")
            st.write("Add DATA_GOV_IN_API_KEY to your .env file")

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

def display_chat_history():
    """Display chat message history"""
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display charts if available
            if "chart" in message and message["chart"]:
                st.plotly_chart(message["chart"], use_container_width=True, key=f"history_chart_{i}")
            
            # Display data summary if available
            if "data_summary" in message and message["data_summary"]:
                st.json(message["data_summary"])

def process_query(user_input: str):
    """Process user query and return response"""
    start_time = time.time()
    
    try:
        # Determine which dataset to query based on keywords
        dataset_key = None
        if any(word in user_input.lower() for word in ['rainfall', 'rain', 'precipitation']):
            dataset_key = 'rainfall'
        elif any(word in user_input.lower() for word in ['cost', 'production cost', 'crop cost']):
            dataset_key = 'crop_cost'
        elif any(word in user_input.lower() for word in ['vegetable', 'crop', 'production']):
            dataset_key = 'vegetables'
        else:
            # Default to rainfall if no specific keywords
            dataset_key = 'rainfall'
        
        dataset = DATASETS[dataset_key]
        
        # Fetch data
        with st.spinner(f"Fetching {dataset['name']}..."):
            data, error = fetch_dataset_data(dataset['resource_id'], limit=50)
            
            if error:
                return {
                    'success': False,
                    'error': f"Failed to fetch data: {error}",
                    'dataset': dataset['name']
                }
        
        # Analyze data based on dataset type
        with st.spinner("Analyzing data..."):
            if dataset_key == 'rainfall':
                analysis, analysis_error = analyze_rainfall_data(data, user_input)
            elif dataset_key == 'crop_cost':
                analysis, analysis_error = analyze_crop_cost_data(data)
            else:  # vegetables
                analysis, analysis_error = analyze_vegetable_data(data)
            
            if analysis_error:
                return {
                    'success': False,
                    'error': f"Analysis failed: {analysis_error}",
                    'dataset': dataset['name']
                }
        
        # Prepare shared metadata
        response_time = int((time.time() - start_time) * 1000)

        # Build local fallback summary
        local_summary = f"**Analysis of {dataset['name']}**\n\n"
        local_summary += f"📊 **Dataset Info:**\n"
        local_summary += f"- Total records: {analysis.get('total_records', 0)}\n"
        local_summary += f"- Available fields: {', '.join(analysis.get('fields', []))}\n"
        local_summary += f"- Ministry: {dataset['ministry']}\n\n"
        if 'top_commodities' in analysis:
            local_summary += f"🌾 **Top Commodities by Price Index:**\n"
            for commodity, value in list(analysis['top_commodities'].items())[:3]:
                local_summary += f"- {commodity}: {value:.2f}\n"
        elif 'top_subdivisions' in analysis:
            local_summary += f"🌧️ **Top Subdivisions by Rainfall:**\n"
            for sub, rainfall in list(analysis['top_subdivisions'].items())[:3]:
                local_summary += f"- {sub}: {rainfall:.2f} mm\n"
        elif 'top_crops' in analysis:
            local_summary += f"🌾 **Top Crops:**\n"
            for crop, value in list(analysis['top_crops'].items())[:3]:
                local_summary += f"- {crop}: {value:.2f}\n"
        local_summary += f"\n📈 **Response Time:** {response_time}ms\n"
        local_summary += f"🔗 **Data Source:** {dataset['url']}"

        # Try Gemini summary (inline)
        with st.spinner("Generating AI response..."):
            data_summary = {
                'dataset_name': dataset['name'],
                'total_records': analysis.get('total_records', 0),
                'fields': analysis.get('fields', []),
                'ministry': dataset['ministry'],
                'top_commodities': analysis.get('top_commodities', {}),
                'top_subdivisions': analysis.get('top_subdivisions', {}),
                'top_crops': analysis.get('top_crops', {}),
                'sample_data': analysis.get('sample_data', [])
            }
            ai_summary, suggested_chart_type = generate_gemini_summary(user_input, data_summary, [dataset['url']])

        return {
            'success': True,
            'summary': ai_summary or local_summary,
            'chart': analysis.get('chart'),
            'data_summary': analysis.get('sample_data', []),
            'dataset': dataset['name'],
            'response_time': response_time
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            'success': False,
            'error': str(e),
            'dataset': 'Unknown'
        }

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Display header and sidebar
    display_header()
    display_sidebar()
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    user_input = st.chat_input("Ask a question about rainfall, crop costs, or vegetable production...")
    
    if user_input or st.session_state.user_input:
        # Use either new input or example query
        query = user_input if user_input else st.session_state.user_input
        st.session_state.user_input = ""  # Clear the example query
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process query and display response
        with st.chat_message("assistant"):
            response = process_query(query)
            
            if response['success']:
                # Display text summary
                st.markdown(response['summary'])
                
                # Display chart if available
                if response.get('chart'):
                    st.plotly_chart(response['chart'], use_container_width=True, key=f"chart_{len(st.session_state.messages)}")
                
                # Display data summary
                if response.get('data_summary'):
                    with st.expander("📋 Sample Data"):
                        st.json(response['data_summary'])
                
                # Add bot message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response['summary'],
                    "chart": response.get('chart'),
                    "data_summary": response.get('data_summary', [])
                })
            else:
                # Display error
                st.error(f"❌ Error: {response['error']}")
                
                # Add error message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"❌ Error: {response['error']}"
                })
    
    # Display footer
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #eee;">
        <p>🌾 Gov Data Q&A Chatbot | Powered by Indian Government Datasets</p>
        <p><small>Data sources: data.gov.in | Built with Streamlit</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()