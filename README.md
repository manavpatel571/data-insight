# 🌾 Gov Data Q&A Chatbot

A Streamlit-based chatbot that answers questions about Indian government datasets from data.gov.in using real-time data fetching and AI-powered analysis.

## ✨ Features

- **Real-time Data Fetching**: Fetches live data from data.gov.in API
- **AI-Powered Responses**: Uses Google Gemini 2.5 Flash for natural language responses
- **Dynamic Visualizations**: Creates interactive charts that adapt to your queries
- **Smart Query Parsing**: Understands different types of queries (trends, comparisons, top-N)
- **Source Citations**: Always shows where the data comes from

## 🎯 Current Dataset

The app currently works with **Commodity Price Index Data** from the Ministry of Commerce and Industry:
- **869 records** covering 2011-2017
- **Monthly price index data** for various commodities
- **Resource ID**: `abfd2d50-0d73-4a3e-9027-10edb3d21940`

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv data-env
data-env\Scripts\activate  # Windows
# source data-env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
DATA_GOV_IN_API_KEY=your_data_gov_in_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

**Get API Keys:**
- **data.gov.in**: Register at https://data.gov.in/ and get your API key from the dashboard
- **Google Gemini**: Get your API key from https://aistudio.google.com/app/apikey

### 3. Run the Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## 💡 Example Queries

### Bar Charts (Comparisons & Top-N)
- "Top 5 commodities by price index"
- "Top 3 commodities by price index"
- "Compare Ginger and Onion"
- "Compare Ginger, Onion and Potato"

### Line Charts (Trends)
- "Show monthly trend for Onion and Potato"
- "Price index trend over time for Ginger (Fresh)"
- "Monthly price trend for Ragi and Rajma"

### Pie Charts (Distributions)
- "Show distribution of top 5 commodities"
- "Distribution of price index among Ginger (Fresh), Onion, Potato"

## 🏗️ Project Structure

```
gov_data/
├── app.py                    # Main Streamlit application
├── data_fetcher.py           # Data.gov.in API integration
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .env                     # API keys (create this)
```

## 🔧 How It Works

1. **Query Processing**: The app analyzes your natural language query to determine intent
2. **Data Fetching**: Fetches real-time data from data.gov.in API using your resource ID
3. **Data Analysis**: Processes the data and creates appropriate visualizations
4. **AI Response**: Google Gemini generates natural language responses based on the data
5. **Dynamic Charts**: Creates interactive charts that adapt to your specific query type

## 📊 Supported Chart Types

- **Bar Charts**: For comparisons and top-N queries
- **Line Charts**: For trend analysis over time
- **Pie Charts**: For distribution and proportion analysis

## 🎨 Query Types

The app intelligently detects different query types:

- **Top-N Queries**: "Show me the top 5 commodities"
- **Comparison Queries**: "Compare commodity A vs B"
- **Trend Queries**: "Show monthly trends for..."
- **Distribution Queries**: "Show distribution of..."

## 🔍 Adding More Datasets

To add more datasets from data.gov.in:

1. Find the dataset on https://data.gov.in/catalog
2. Get the Resource ID (UUID format) from the API tab
3. Update the `DATASETS` configuration in `app.py`
4. Add corresponding analysis functions

## 🛠️ Dependencies

- **Streamlit**: Web application framework
- **Google Generative AI**: For AI-powered responses
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **Requests**: HTTP requests to data.gov.in API

## 📝 Notes

- The app currently works with commodity price index data (869 records, 2011-2017)
- ALTS credentials warnings in the console are harmless Google Cloud warnings
- Charts adapt dynamically based on your query keywords
- All responses include source citations from data.gov.in

## 🤝 Contributing

Feel free to extend the app with additional datasets or improve the query parsing logic!

## 📄 License

This project is for educational and research purposes.