"""
TransformerGuard Dashboard
Main Streamlit application for transformer fleet monitoring
"""

from pathlib import Path
from typing import Any, Dict, Optional

import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="TransformerGuard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Navigation pages (map display names to module names)
PAGES = {
    "Fleet Overview": "fleet_overview",
    "Transformer Detail": "transformer_detail",
    "DGA Analysis": "dga_analysis",
    "Trend Monitor": "trend_monitor",
    "Alert Center": "alert_center",
}


def get_api_client() -> requests.Session:
    """Create configured API client session."""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


@st.cache_data(ttl=60)
def fetch_from_api(endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Fetch data from API with caching.

    Args:
        endpoint: API endpoint path
        params: Optional query parameters

    Returns:
        JSON response data
    """
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to API at {API_BASE_URL}. Please ensure the API server is running."
        )
        return {"error": "connection_failed"}
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return {"error": "timeout"}
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
        return {"error": str(e)}
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return {"error": str(e)}


def post_to_api(endpoint: str, data: Dict) -> Dict[str, Any]:
    """
    Post data to API.

    Args:
        endpoint: API endpoint path
        data: Request payload

    Returns:
        JSON response data
    """
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to API at {API_BASE_URL}. Please ensure the API server is running."
        )
        return {"error": "connection_failed"}
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
        return {"error": str(e)}
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return {"error": str(e)}


def put_to_api(endpoint: str, data: Dict) -> Dict[str, Any]:
    """
    Put data to API.

    Args:
        endpoint: API endpoint path
        data: Request payload

    Returns:
        JSON response data
    """
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.put(url, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to API at {API_BASE_URL}. Please ensure the API server is running."
        )
        return {"error": "connection_failed"}
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
        return {"error": str(e)}
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return {"error": str(e)}


def check_api_health() -> bool:
    """Check if API is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.title("⚡ TransformerGuard")
    st.sidebar.info("Transformer Health Intelligence System")

    # API health check
    if not check_api_health():
        st.sidebar.error("⚠️ API Offline")
    else:
        st.sidebar.success("✅ API Connected")

    st.sidebar.markdown("---")

    # Navigation
    selection = st.sidebar.radio("Navigate", list(PAGES.keys()))

    # Import and run selected page
    page_module_name = PAGES.get(selection)
    try:
        page_module = __import__(f"pages.{page_module_name}", fromlist=["main"])
        page_module.main()
    except ImportError as e:
        st.error(f"Could not load page: {selection}")
        st.info("Please ensure all page files are properly configured.")
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")


if __name__ == "__main__":
    main()
