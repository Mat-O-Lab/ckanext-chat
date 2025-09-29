import ckan.plugins.toolkit as toolkit
import requests
import os


def service_available():
    url = toolkit.config.get("ckanext.chat.completion_url")
    if not url:
        return False
    try:
        # Perform a HEAD request (lightweight check) to see if the service responds
        response = requests.head(url, timeout=5, verify=False)
        if (200 <= response.status_code < 400) or response.status_code == 405:
            return True  # URL is reachable and returns a valid status code
        else:
            return False  # URL is reachable but response status is not valid
    except requests.RequestException as e:
        # If there's any issue (timeout, connection error, etc.)
        return False


def load_prompt(config_key: str, default_filename: str, extension_dir: str) -> str:
    """
    Load a prompt either from a URL (if set in config) or from a local file.
    """
    prompt_url = toolkit.config.get(config_key, None)
    if prompt_url:
        try:
            response = requests.get(prompt_url)
            if response.status_code == 200:
                return response.text
            else:
                print(f"Failed to retrieve {config_key} from URL. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error fetching {config_key} from URL: {e}")

    # fallback to local file
    prompt_path = os.path.join(extension_dir, default_filename)
    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


def get_helpers():
    return {
        "service_available": service_available,
    }

