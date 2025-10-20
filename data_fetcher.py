import requests
import json
import time
import logging
from typing import List, Dict, Any, Optional, Generator
from urllib.parse import urlencode
from config import Config
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class DataGovInFetcher:
    """Fetches data from data.gov.in API"""
    
    def __init__(self, api_key: str = None, db_manager: DatabaseManager = None):
        self.api_key = api_key or Config.DATA_GOV_IN_API_KEY
        self.base_url = Config.DATA_GOV_BASE_URL
        self.db_manager = db_manager or DatabaseManager()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Gov-Data-QA-Chatbot/1.0',
            'Accept': 'application/json'
        })
        
        if not self.api_key:
            raise ValueError("DATA_GOV_IN_API_KEY is required")
    
    def _make_request(self, resource_id: str, params: Dict[str, Any] = None, 
                     use_cache: bool = True) -> Dict[str, Any]:
        """Make API request with caching and retry logic"""
        params = params or {}
        params['api-key'] = self.api_key
        params['format'] = 'json'
        
        # Create cache key from params
        cache_key = f"{resource_id}_{urlencode(sorted(params.items()))}"
        
        # Check cache first
        if use_cache:
            cached_data = self.db_manager.get_cached_response(resource_id, cache_key)
            if cached_data:
                logger.info(f"Using cached data for resource {resource_id}")
                return cached_data
        
        url = f"{self.base_url}/{resource_id}"
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=Config.API_TIMEOUT)
                response.raise_for_status()
                
                data = response.json()
                
                # Cache the response
                if use_cache:
                    self.db_manager.cache_api_response(resource_id, cache_key, data)
                
                logger.info(f"Successfully fetched data for resource {resource_id}")
                return data
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for resource {resource_id}: {e}")
                if attempt == Config.MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def fetch_dataset(self, resource_id: str, limit: int = 1000, 
                     offset: int = 0, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch dataset with pagination support"""
        params = {
            'limit': limit,
            'offset': offset
        }
        
        if filters:
            for key, value in filters.items():
                params[f'filters[{key}]'] = value
        
        return self._make_request(resource_id, params)
    
    def fetch_all_data(self, resource_id: str, batch_size: int = 1000,
                      filters: Dict[str, Any] = None) -> Generator[List[Dict], None, None]:
        """Fetch all data from a resource with pagination"""
        offset = 0
        
        while True:
            data = self.fetch_dataset(resource_id, limit=batch_size, offset=offset, filters=filters)
            
            records = data.get('records', [])
            if not records:
                break
            
            yield records
            offset += len(records)
            
            # Check if we've reached the end
            if len(records) < batch_size:
                break
    
    def get_available_datasets(self, ministry: str = None, category: str = None) -> List[Dict]:
        """Get list of available datasets"""
        # Real working dataset from data.gov.in
        important_datasets = [
            {
                'resource_id': 'abfd2d50-0d73-4a3e-9027-10edb3d21940',
                'name': 'Commodity Price Index Data',
                'ministry': 'Ministry of Commerce and Industry',
                'category': 'economics',
                'description': 'Monthly commodity price index data from 2011-2017 (869 records)',
                'url': 'https://www.data.gov.in/apis/abfd2d50-0d73-4a3e-9027-10edb3d21940'
            }
        ]
        
        if ministry:
            important_datasets = [d for d in important_datasets if ministry.lower() in d['ministry'].lower()]
        
        if category:
            important_datasets = [d for d in important_datasets if category.lower() in d['category'].lower()]
        
        return important_datasets
    
    def search_datasets(self, query: str, ministry: str = None) -> List[Dict]:
        """Search for datasets by query"""
        # In a real implementation, this would use the search API
        # For now, we'll filter our known datasets
        
        all_datasets = self.get_available_datasets(ministry)
        
        query_lower = query.lower()
        matching_datasets = []
        
        for dataset in all_datasets:
            if (query_lower in dataset['name'].lower() or 
                query_lower in dataset['description'].lower() or
                query_lower in dataset['category'].lower()):
                matching_datasets.append(dataset)
        
        return matching_datasets
    
    def get_dataset_info(self, resource_id: str) -> Dict[str, Any]:
        """Get detailed information about a dataset"""
        # Fetch a small sample to get metadata
        data = self.fetch_dataset(resource_id, limit=1)
        
        return {
            'resource_id': resource_id,
            'total_records': data.get('total', 0),
            'fields': list(data.get('records', [{}])[0].keys()) if data.get('records') else [],
            'last_updated': data.get('last_updated'),
            'source': data.get('source')
        }
    
    def validate_resource_id(self, resource_id: str) -> bool:
        """Validate if a resource ID exists and is accessible"""
        try:
            data = self.fetch_dataset(resource_id, limit=1)
            return 'records' in data
        except Exception as e:
            logger.warning(f"Resource ID {resource_id} validation failed: {e}")
            return False
    
    def get_rainfall_data(self, states: List[str] = None, years: List[int] = None,
                         months: List[int] = None) -> List[Dict]:
        """Fetch rainfall data with filters"""
        resource_id = 'rainfall-district-wise-monthly'
        
        # Build filters
        filters = {}
        if states:
            filters['state'] = '|'.join(states)
        if years:
            filters['year'] = '|'.join(map(str, years))
        if months:
            filters['month'] = '|'.join(map(str, months))
        
        all_records = []
        for batch in self.fetch_all_data(resource_id, filters=filters):
            all_records.extend(batch)
        
        return all_records
    
    def get_crop_production_data(self, states: List[str] = None, years: List[int] = None,
                                crops: List[str] = None, seasons: List[str] = None) -> List[Dict]:
        """Fetch crop production data with filters"""
        resource_id = 'area-production-statistics'
        
        # Build filters
        filters = {}
        if states:
            filters['state'] = '|'.join(states)
        if years:
            filters['year'] = '|'.join(map(str, years))
        if crops:
            filters['crop'] = '|'.join(crops)
        if seasons:
            filters['season'] = '|'.join(seasons)
        
        all_records = []
        for batch in self.fetch_all_data(resource_id, filters=filters):
            all_records.extend(batch)
        
        return all_records
    
    def get_climate_data(self, states: List[str] = None, years: List[int] = None) -> List[Dict]:
        """Fetch climate data (temperature, humidity, etc.)"""
        resource_id = 'climate-data-district-wise'
        
        filters = {}
        if states:
            filters['state'] = '|'.join(states)
        if years:
            filters['year'] = '|'.join(map(str, years))
        
        all_records = []
        for batch in self.fetch_all_data(resource_id, filters=filters):
            all_records.extend(batch)
        
        return all_records
    
    def test_api_connection(self) -> Dict[str, Any]:
        """Test API connection with a simple request"""
        try:
            # Test with a simple request to get API info
            test_url = "https://api.data.gov.in/resource/test"
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'limit': 1
            }
            
            response = self.session.get(test_url, params=params, timeout=10)
            
            return {
                'success': True,
                'status_code': response.status_code,
                'message': 'API connection successful' if response.status_code == 200 else f'API returned status {response.status_code}',
                'api_key_configured': bool(self.api_key),
                'base_url': self.base_url
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'api_key_configured': bool(self.api_key),
                'base_url': self.base_url
            }
    
    def get_api_usage_info(self) -> str:
        """Get information about how to use the API"""
        return """
        How to use data.gov.in API:
        
        1. API Key Setup:
           - Register at https://data.gov.in/
           - Get your API key from the dashboard
           - Add it to your .env file as DATA_GOV_IN_API_KEY
        
        2. API Endpoint Format:
           https://api.data.gov.in/resource/{resource_id}?api-key={your_key}&format=json&limit=100
        
        3. Common Parameters:
           - api-key: Your API key (required)
           - format: json (default) or csv
           - limit: Number of records (max 10000)
           - offset: Starting record number
           - filters[field_name]: Filter by specific field values
        
        4. Example Usage:
           - Rainfall data: https://api.data.gov.in/resource/rainfall-district-wise-monthly
           - Crop data: https://api.data.gov.in/resource/crop-production-statistics
        
        5. Finding Resource IDs:
           - Visit https://data.gov.in/catalog
           - Search for datasets
           - Look for "API" or "Resource ID" in dataset details
        
        6. Rate Limits:
           - 1000 requests per day (free tier)
           - 100 requests per minute
        """
    
    def close(self):
        """Close the session"""
        self.session.close()

# Convenience functions for easy access
def get_fetcher() -> DataGovInFetcher:
    """Get a configured DataGovInFetcher instance"""
    return DataGovInFetcher()

def fetch_rainfall_data(states: List[str] = None, years: List[int] = None) -> List[Dict]:
    """Convenience function to fetch rainfall data"""
    fetcher = get_fetcher()
    try:
        return fetcher.get_rainfall_data(states, years)
    finally:
        fetcher.close()

def fetch_crop_data(states: List[str] = None, years: List[int] = None, 
                   crops: List[str] = None, seasons: List[str] = None) -> List[Dict]:
    """Convenience function to fetch crop production data"""
    fetcher = get_fetcher()
    try:
        return fetcher.get_crop_production_data(states, years, crops, seasons)
    finally:
        fetcher.close()
