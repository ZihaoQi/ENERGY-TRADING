import requests
import pandas as pd
from datetime import timedelta, datetime
from data.api.api_config import API_KEYS

class GieClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"x-key": self.api_key}

    def _get_data(self, url, params):
        """
        Helper to send GET request and return normalized DataFrame.
        """
        r = requests.get(url, params=params, headers=self.headers)
        r.raise_for_status()  # Raises HTTPError for bad responses
        data = r.json().get("data", [])
        return pd.json_normalize(data)

    def query_gas_eu(self, start_date: str, end_date: str):
        """
        Query overall EU gas data

        start date and end date format: YYYY-MM-DD, eg: 2025-07-23
        """
        url = "https://agsi.gie.eu/api"
        params = {"from": start_date, "to": end_date, "type": "EU", "size": 300}
        return self._get_data(url, params)

    def query_gas_country(self, country_code: str, start_date: str, end_date: str):
        """
        Query gas data for a specific country

        start date and end date format: YYYY-MM-DD, eg: 2025-07-23
        """
        url = "https://agsi.gie.eu/api"
        params = {"from": start_date, "to": end_date, "country": country_code, "size": 300}
        return self._get_data(url, params)

    def query_lng_eu(self, start_date: str, end_date: str):
        """
        Query overall EU LNG data

        start date and end date format: YYYY-MM-DD, eg: 2025-07-23
        """
        url = "https://alsi.gie.eu/api"
        params = {"from": start_date, "to": end_date, "type": "EU", "size": 300}
        return self._get_data(url, params)

    def query_lng_country(self, country_code: str, start_date: str, end_date: str):
        """
        Query LNG data for a specific country

        start date and end date format: YYYY-MM-DD, eg: 2025-07-23
        """
        url = "https://alsi.gie.eu/api"
        params = {"from": start_date, "to": end_date, "country": country_code, "size": 300}
        return self._get_data(url, params)
    
def fetch_gas_storage_data(start_date_post: str, end_date_post: str):
    """
    Query overall EU gas data

    start date and end date format: YYYY-MM-DD, eg: 2025-07-23
    """
    print('Fetching GIE gas storage data...\n')
    client = GieClient(API_KEYS['gie'])
    start_date = (datetime.strptime(start_date_post, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = (datetime.strptime(end_date_post, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    df = client.query_gas_eu(start_date, end_date)
    df['postDate'] = pd.to_datetime(df['GasDayEnd'])
    df_final = df[['postDate', 'full', 'netWithdrawal']]
    df_final = df_final.sort_values(by='postDate', ascending=True)
    df_final.set_index('postDate', inplace=True)
    print("Gas storage data fetched")
    return df_final

def fetch_lng_storage_data(start_date_post: str, end_date_post: str):
    """
    Query overall EU LNG data

    start date and end date format: YYYY-MM-DD, eg: 2025-07-23
    """
    print('Fetching GIE LNG storage data...\n')
    client = GieClient(API_KEYS['gie'])
    start_date = (datetime.strptime(start_date_post, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = (datetime.strptime(end_date_post, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    df = client.query_lng_eu(start_date, end_date)
    df['postDate'] = pd.to_datetime(df['GasDayEnd'])
    df_final = df[['postDate', 'sendOut']]
    df_final = df_final.sort_values(by='postDate', ascending=True)
    df_final.set_index('postDate', inplace=True)
    print("LNG storage data fetched")
    return df_final