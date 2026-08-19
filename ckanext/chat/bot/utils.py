import asyncio
import json
import re
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import ckan.plugins.toolkit as toolkit
import regex
import tiktoken
from ckan.lib.lazyjson import LazyJSONObject
from ckan.model.package import Package
from ckan.model.resource import Resource
from loguru import logger
from pydantic import BaseModel, ValidationError, computed_field, model_validator

log = logger.bind(module=__name__)

# --------------------- Dynamic Models Initialization ---------------------

dynamic_models_initialized = False


def init_dynamic_models():
    global dynamic_models_initialized
    if not dynamic_models_initialized:
        get_ckan_url_patterns()
        try:
            package_list = toolkit.get_action("package_list")({}, {})
            if package_list:
                sample_pkg = toolkit.get_action("package_show")(
                    {}, {"id": package_list[0]}
                )
                _ = DynamicDataset(**sample_pkg)
        except Exception as e:
            log.warning(f"Could not initialize sample dynamic models: {e}")
        dynamic_models_initialized = True


# --------------------- Dynamic Models ---------------------


class DynamicDataset(BaseModel):
    id: str  # CKAN dataset id
    view_url: Optional[str] = None

    class Config:
        extra = "allow"

    @model_validator(mode='before')
    @classmethod
    def calculate_computed_field(cls, data):
        route = find_route_by_endpoint("dataset.read")
        ckan_url = toolkit.config.get("ckan.site_url")
        if route and ckan_url:
            data["view_url"] = str(route.build_url(base_url=ckan_url,fill={"id": data.get("id")}))
        resources = data.get("resources")
        if not isinstance(resources, list):
            raise ValueError(
                'Input should have a "resources" key with a list of resources.'
            )
        validated_resources = [DynamicResource(**resource) for resource in resources]
        data["resources"] = validated_resources
        return data

    @classmethod
    def from_ckan(cls, package: Package) -> "DynamicDataset":
        data = package.as_dict() if hasattr(package, "as_dict") else package.__dict__
        return cls(**data)


class DynamicResource(BaseModel):
    id: str  # CKAN resource id
    view_url: Optional[str] = None

    class Config:
        extra = "allow"

    @model_validator(mode='before')
    @classmethod
    def calculate_computed_field(cls, data):
        route = find_route_by_endpoint("resource.read")
        ckan_url = toolkit.config.get("ckan.site_url")
        if route and ckan_url:
            data["view_url"] = str(route.build_url(fill={"id": data.get("id")}))
        return data

    @classmethod
    def from_ckan(cls, resource: Resource) -> "DynamicResource":
        data = resource.as_dict() if hasattr(resource, "as_dict") else resource.__dict__
        filtered_data = {
            k: v for k, v in data.items() if v not in ([], {}, "", "", "null")
        }
        return cls(**filtered_data)


# --------------------- CKAN Actions and URL Helpers ---------------------


class FuncSignature(BaseModel):
    doc: Any
    defaults: Optional[Dict[str, Any]] = {}


CKAN_ACTIONS: Dict[str, FuncSignature] = {}


@lru_cache(maxsize=1)
def get_ckan_actions() -> Dict[str,str]:
    """
    Get all CKAN actions with LRU caching.
    Cache is cleared when CKAN is restarted.
    """
    global CKAN_ACTIONS
    if not CKAN_ACTIONS:
        from ckan.logic import _actions
        from ckan.logic.action.get import help_show

        actions = [key for key in _actions.keys() if "_update" not in key]
        for item in actions:
            doc = help_show({}, {"name": item})
            CKAN_ACTIONS[item] = FuncSignature(doc=doc).model_dump()
    return {k: v["doc"].strip().splitlines()[0] if v.get("doc") else "" for k, v in CKAN_ACTIONS.items()}

def extract_defaults_from_signature(action_name: str) -> Dict[str, Any]:
    """
    Extract default parameter values directly from Python function signature.
    This is completely dynamic and adapts to API changes automatically.
    
    Args:
        action_name: The CKAN action name
        
    Returns:
        Dictionary mapping parameter names to their default values
    """
    try:
        import inspect
        from ckan.logic import get_action
        
        # Get the actual action function
        action_func = get_action(action_name)
        
        # Get its signature
        sig = inspect.signature(action_func)
        
        # Extract parameters with defaults
        defaults = {}
        for param_name, param in sig.parameters.items():
            # Skip framework parameters
            if param_name in ('context', 'data_dict'):
                continue
                
            # Check if parameter has a default value
            if param.default != inspect.Parameter.empty:
                defaults[param_name] = param.default
        
        return defaults
        
    except Exception as e:
        log.debug(f"Could not inspect signature for {action_name}: {e}")
        return {}


def get_ckan_action(action: str) -> FuncSignature:
    global CKAN_ACTIONS
    if not CKAN_ACTIONS:
        from ckan.logic import _actions
        from ckan.logic.action.get import help_show

        actions = [key for key in _actions.keys() if "_update" not in key]
        for item in actions:
            doc = help_show({}, {"name": item})
            # Extract defaults from signature
            defaults = extract_defaults_from_signature(item)
            CKAN_ACTIONS[item] = FuncSignature(doc=doc, defaults=defaults).model_dump()
    if action in CKAN_ACTIONS.keys():
        return CKAN_ACTIONS[action]
    else:
        return None


def parse_default_value(value_str: str) -> Any:
    """Convert string representation of default value to actual Python type"""
    value_str = value_str.strip()
    
    # Remove markdown/rst backticks that CKAN uses in docstrings
    value_str = value_str.replace('``', '').strip()
    
    # Remove trailing punctuation from docstring capture (., ,, ;)
    value_str = value_str.rstrip('.,;')
    
    # Remove surrounding quotes (matched pairs)
    if value_str.startswith(("'", '"')) and value_str.endswith(("'", '"')) and len(value_str) >= 2:
        value_str = value_str[1:-1].strip()
    
    # Remove unmatched quotes (parsing artifacts)
    if value_str.count("'") == 1:
        value_str = value_str.replace("'", "")
    if value_str.count('"') == 1:
        value_str = value_str.replace('"', "")
    
    # Clean again after quote removal
    value_str = value_str.strip().rstrip('.,;')
    
    # Boolean values
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'
    
    # None/null values
    if value_str.lower() in ('none', 'null'):
        return None
    
    # Integer values
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # Float values
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Return as-is if we can't parse it
    return value_str


def extract_param_defaults(action_doc: str) -> Dict[str, Any]:
    """
    Parse CKAN action docstring to extract parameter defaults dynamically.
    
    Looks for patterns like:
    - :param name: (optional, default: value)
    - :param name: description (default: value)
    - :param name: ... Default: value
    
    Args:
        action_doc: The docstring from help_show()
        
    Returns:
        Dictionary mapping parameter names to their default values
    """
    if not action_doc:
        return {}
    
    defaults = {}
    
    # Token-level capture: RST backtick-wrapped, quoted string, or bare word
    _val = r"""(``[^`]+``|"[^"]+"|'[^']+'|\S+)"""

    # Try multiple patterns to match different docstring formats
    patterns = [
        # Pattern 1: :param name: ... (default: value) — bounded by parens
        r':param\s+(\w+):\s*[^:]*?\(default:\s*([^)]+)\)',
        # Pattern 2: :param name: ... default: value — token-level capture
        r':param\s+(\w+):\s*[^:]*?default:\s*' + _val,
        # Pattern 3: :param name: ... Default: value — token-level capture
        r':param\s+(\w+):\s*[^:]*?Default:\s*' + _val,
        # Pattern 4: (optional, default: value) — bounded by parens
        r':param\s+(\w+):[^:]*?\(optional[^)]*default:\s*([^)]+)\)',
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, action_doc, re.MULTILINE | re.IGNORECASE):
            param_name = match.group(1)
            if param_name not in defaults:  # Don't override if already found
                default_value_str = match.group(2).strip().rstrip(')')
                defaults[param_name] = parse_default_value(default_value_str)
    
    return defaults


def merge_with_smart_defaults(action: str, provided_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge provided parameters with defaults extracted dynamically from action metadata.
    
    Pure dynamic approach using three tiers:
    1. Signature inspection (Python code defaults - most accurate)
    2. Docstring parsing (documented defaults from help_show)
    3. User parameters (always take precedence, except empty strings)
    
    Zero hardcoding - everything extracted dynamically from CKAN.
    
    Args:
        action: The CKAN action name
        provided_params: Parameters provided by the user/agent
        
    Returns:
        Merged dictionary with defaults filled in (provided params take precedence)
    """
    action_info = get_ckan_action(action)
    
    if not action_info:
        return provided_params
    
    # Filter out empty strings and None - treat as not provided
    # Empty string "" should use defaults, not override them
    filtered_params = {
        k: v for k, v in provided_params.items() 
        if v != "" and v is not None
    }
    
    # Tier 1: Signature defaults (from Python function signature)
    sig_defaults = action_info.get('defaults', {})
    
    # Tier 2: Docstring defaults (from help_show documentation)
    doc_defaults = extract_param_defaults(action_info.get('doc', ''))
    
    # Merge: signature defaults override docstring (more authoritative)
    # Filter None — sending None explicitly differs from omitting the param
    defaults = {k: v for k, v in {**doc_defaults, **sig_defaults}.items() if v is not None}

    # Tier 3: User params always override everything (but not empty strings)
    merged = {**defaults, **filtered_params}
    
    log.debug(f"merge_with_smart_defaults: action={action}, sig_defaults={sig_defaults}, doc_defaults={doc_defaults}, filtered_params={filtered_params}, merged={merged}")
    
    return merged


def detect_pagination_params(action_doc: str) -> Optional[Dict[str, str]]:
    """
    Detect pagination parameters from CKAN action documentation.
    
    Looks for common pagination patterns:
    - limit/offset
    - rows/start
    - per_page/page
    
    Args:
        action_doc: The docstring from help_show()
        
    Returns:
        Dict with 'limit' and 'offset' keys mapping to actual parameter names,
        or None if no pagination params found
    """
    if not action_doc:
        return None
    
    pagination_keywords = {
        'limit': ['limit', 'rows', 'per_page'],
        'offset': ['offset', 'start', 'page']
    }
    
    found_params = {}
    doc_lower = action_doc.lower()
    
    # Check for each pagination keyword
    for param_type, keywords in pagination_keywords.items():
        for keyword in keywords:
            # Look for :param keyword: in docstring
            if f':param {keyword}' in doc_lower:
                found_params[param_type] = keyword
                break
    
    # Only return if we found at least a limit parameter
    return found_params if 'limit' in found_params else None


def generate_pagination_hint(action_name: str, estimated_tokens: int, items_count: int, pagination_params: Optional[Dict[str, str]]) -> Optional[str]:
    """
    Generate pagination hint if response is large and action supports pagination.
    
    Args:
        action_name: The CKAN action name
        estimated_tokens: Estimated token count of response
        items_count: Number of items in response
        pagination_params: Dict with pagination parameter names (from detect_pagination_params)
        
    Returns:
        Pagination hint string, or None if not needed
    """
    # Only suggest pagination if response is large (>2000 tokens)
    if estimated_tokens < 2000:
        return None
    
    # Only suggest if action supports pagination
    if not pagination_params:
        return None
    
    limit_param = pagination_params.get('limit', 'limit')
    offset_param = pagination_params.get('offset', 'offset')
    
    # Calculate suggested page size and number of pages
    suggested_limit = min(50, max(10, items_count // 10))
    estimated_pages = (items_count + suggested_limit - 1) // suggested_limit if items_count > 0 else 1
    
    hint = (
        f"Response is large ({estimated_tokens} tokens, {items_count} items). "
        f"Consider pagination:\n"
        f"- Use '{limit_param}' parameter to set page size (suggested: {suggested_limit})\n"
        f"- Use '{offset_param}' parameter to iterate through pages\n"
        f"- Estimated pages needed: {estimated_pages}\n"
        f"Example: {action_name}({limit_param}={suggested_limit}, {offset_param}=0)"
    )
    
    return hint


# --------------------- CKAN Routing and URL Helpers ---------------------

VARIABLE_REGEX = re.compile(r"<(?:(?P<converter>[^:<>]+):)?(?P<variable>[^<>]+)>")


def extract_variables(rule: str) -> List[Dict[str, Optional[str]]]:
    return [match.groupdict() for match in VARIABLE_REGEX.finditer(rule)]


def repl(match):
    var = match.group("variable")
    return f"{{{var}}}"


class RouteModel(BaseModel):
    endpoint: str
    rule: str
    methods: Optional[list[str]] = []
    variables: Optional[list] = []
    full_url_pattern: Optional[str]

    @model_validator(mode='before')
    @classmethod
    def calculate_computed_field(cls, data):
        data["variables"] = extract_variables(data["rule"])
        data["full_url_pattern"] = VARIABLE_REGEX.sub(repl, data["rule"])
        return data

    def build_url(
        self,
        base_url: str = toolkit.config.get("ckan.site_url", ""),
        fill: Optional[Dict[str, Any]] = None,
    ) -> str:
        fill = fill or {}
        substitution = {
            var["variable"]: str(fill.get(var["variable"], f"<{var['variable']}>"))
            for var in self.variables
        }
        pattern = self.full_url_pattern
        if base_url.endswith("/") and pattern.startswith("/"):
            base_url = base_url[:-1]
        try:
            url_path = pattern.format(**substitution)
        except KeyError as e:
            raise ValueError(f"Missing substitution for variable: {e.args[0]}") from e
        return f"{base_url}{url_path}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "rule": self.rule,
            "methods": self.methods,
            "variables": self.variables,
            "full_url_pattern": self.full_url_pattern,
        }


CKAN_ROUTES: Dict[str, RouteModel] = {}


def find_route_by_endpoint(endpoint: str) -> Optional[RouteModel]:
    if endpoint in CKAN_ROUTES.keys():
        return CKAN_ROUTES[endpoint]
    return None


def truncate_output_by_token(
    output: str, token_limit: int, offset: int = 0, encoding_name="cl100k_base"
) -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(output)

    if len(tokens) > token_limit:
        # Skip the specified number of tokens
        truncated_tokens = tokens[offset : offset + token_limit]
        output = encoding.decode(truncated_tokens)
        # if last page of tokens, add a mark
        if len(truncated_tokens) < token_limit:
            output += "\n\n**End of Output**"

    return output


def truncate_value(value, max_length):
    if isinstance(value, str):
        return value[:max_length] + "..." if len(value) > max_length else value
    elif isinstance(value, list):
        truncated_list = [truncate_value(item, max_length) for item in value]
        return (
            truncated_list[:max_length] + ["..."]
            if len(truncated_list) > max_length
            else truncated_list
        )
    return value


def truncate_by_depth(data, max_depth, current_depth=0, placeholder="..."):
    """
    Truncate data by depth, removing empty/None values completely.
    This keeps the response minimal and meaningful.
    """
    if current_depth >= max_depth:
        return placeholder
    
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # Skip None, empty strings, empty lists, empty dicts
            if value in (None, "", [], {}):
                continue
            
            truncated = truncate_by_depth(
                truncate_value(value, max_length=200),
                max_depth,
                current_depth + 1,
                placeholder,
            )
            
            # Only include if the truncated value is meaningful
            if truncated not in (None, "", [], {}, placeholder):
                result[key] = truncated
        
        return result
    
    if isinstance(data, list):
        # Filter out None/empty items first
        filtered = [item for item in data if item not in (None, "", [], {})]
        data = truncate_value(filtered, max_length=200)
        
        result = []
        for item in data:
            truncated = truncate_by_depth(item, max_depth, current_depth + 1, placeholder)
            # Only include meaningful items
            if truncated not in (None, "", [], {}, placeholder):
                result.append(truncated)
        
        return result
    
    return data


def smart_truncate_response(response: Any, max_tokens: int = 1500) -> Dict[str, Any]:
    """
    Intelligently truncate CKAN response based on structure and size.
    
    Strategy:
    - Small responses (<500 tokens): No truncation
    - Medium responses (500-1.5K tokens): Depth truncation (keeps all items, limits detail)
    - Large responses (>1.5K tokens): Combined depth + item limit
    
    Args:
        response: Raw CKAN API response
        max_tokens: Maximum token budget (default: 8000)
        
    Returns:
        Dict with truncated data and metadata about truncation
    """
    # Estimate size
    json_str = json.dumps(response)
    estimated_tokens = len(json_str) // 4
    total_count = 1
    
    # Count items
    if isinstance(response, list):
        total_count = len(response)
        items = response
    elif isinstance(response, dict) and 'results' in response:
        total_count = response.get('count', len(response['results']))
        items = response['results']
    elif isinstance(response, dict):
        items = response
        total_count = 1
    else:
        items = response
        total_count = 1
    
    # Decision tree
    if estimated_tokens < 500:
        # Small response - no truncation needed
        return {
            'data': process_entity(response),
            'truncated': False,
            'truncation_method': 'none',
            'total_items': total_count,
            'showing_items': total_count if isinstance(items, list) else 1,
            'estimated_tokens': estimated_tokens
        }
    
    elif estimated_tokens < 1500:
        # Medium response - use depth truncation (keeps all items, less detail)
        # Process entities FIRST (before truncation to avoid DynamicResource errors)
        processed_response = process_entity(response)
        truncated_data = truncate_by_depth(processed_response, max_depth=3)
        
        # Re-estimate after truncation
        new_json_str = json.dumps(truncated_data)
        new_tokens = len(new_json_str) // 4
        
        return {
            'data': truncated_data,
            'truncated': True,
            'truncation_method': 'depth',
            'total_items': total_count,
            'showing_items': total_count if isinstance(items, list) else 1,
            'estimated_tokens': new_tokens,
            'original_tokens': estimated_tokens
        }
    
    else:
        # Large response - combine depth truncation + item limit
        # Process entities FIRST (before truncation to avoid DynamicResource errors)
        processed_response = process_entity(response)
        
        # Then limit items and truncate (reduced to 5 items for speed)
        if isinstance(processed_response, dict) and 'results' in processed_response:
            # Keep top-level structure, truncate individual results
            limited_response = {**processed_response}
            limited_response['results'] = [
                truncate_by_depth(item, max_depth=2) for item in processed_response['results'][:5]
            ]
        elif isinstance(processed_response, list):
            limited_response = [truncate_by_depth(item, max_depth=2) for item in processed_response[:5]]
        else:
            limited_response = truncate_by_depth(processed_response, max_depth=2)
        
        # Calculate actual showing count
        if isinstance(response, dict) and 'results' in response:
            showing_count = min(5, len(response['results']))
        elif isinstance(response, list):
            showing_count = min(5, len(response))
        else:
            showing_count = 1
        
        # Re-estimate after truncation
        new_json_str = json.dumps(limited_response)
        new_tokens = len(new_json_str) // 4
        
        return {
            'data': limited_response,
            'truncated': True,
            'truncation_method': 'depth+limit',
            'total_items': total_count,
            'showing_items': showing_count,
            'estimated_tokens': new_tokens,
            'original_tokens': estimated_tokens
        }


def unpack_lazy_json(obj):
    if isinstance(obj, dict):
        return {key: unpack_lazy_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [unpack_lazy_json(item) for item in obj]
    elif isinstance(obj, LazyJSONObject):
        return obj.encoded_json
    return obj


def process_entity(data: Any, depth: int = 0, max_depth: int = 4) -> Any:
    #log.debug(f"{type(data)},{depth},{max_depth}")
    if depth > max_depth:
        log.warning("Max recursion depth reached")
        return None
    data = unpack_lazy_json(data)
    if isinstance(data, dict):
        if "resources" in data:
            try:
                #log.debug("Dataset")
                dataset_dict = DynamicDataset(**data).model_dump(
                    exclude_unset=True, exclude_defaults=False, exclude_none=True
                )
                dataset_dict = {k: v for k, v in dataset_dict.items() if bool(v)}
                return truncate_by_depth(dataset_dict,max_depth-depth)
            except ValidationError as validation_error:
                log.warning(
                    f"Validation error converting to DynamicDataset: {validation_error.json()}"
                )
            except Exception as ex:
                log.warning(f"Conversion to DynamicDataset failed: {ex}")
        elif "package_id" in data or "url" in data:
            try:
                #log.debug("Resource")
                resource_dict = DynamicResource(**data).model_dump(
                    exclude_unset=True, exclude_defaults=False, exclude_none=True
                )
                resource_dict = {k: v for k, v in resource_dict.items() if bool(v)}
                return  truncate_by_depth(resource_dict,max_depth-depth)
            except ValidationError as validation_error:
                log.warning(
                    f"Validation error converting to DynamicResource: {validation_error.json()}"
                )
            except Exception as ex:
                log.warning(f"Conversion to DynamicResource failed: {ex}")
        else:
            #log.debug("Dictionary")
            new_dict = {}
            for key, value in data.items():
                processed_value = process_entity(value, depth + 1, max_depth)
                if processed_value not in ([], {}, "", None):
                    new_dict[key] = processed_value
            return new_dict

    elif isinstance(data, list):
        new_list = []
        for item in data:
            processed_item = process_entity(item, depth + 1, max_depth)
            if processed_item not in ([], {}, "", None):
                new_list.append(processed_item)
        return new_list
    else:
        return data

def get_ckan_url_patterns(endpoint: str = "") -> RouteModel:
    """Get URL Flask Blueprint routes to views in CKAN if the argument endpoint is None or empty it wil return a list of endpoints. If set to an endpoint it will return the RouteModel containing arguements and the pattern to create the url.

    Args:
        endpoint (str, optional): If empty returns a list of all possible endpoints. If set returns the details of the endpoint. Defaults to "".

    Returns:
        RouteModel: All details on the Route
    """
    global CKAN_ROUTES
    if not CKAN_ROUTES:
        from ckanext.chat.views import global_ckan_app

        for rule in global_ckan_app.url_map.iter_rules():
            if not rule.rule.startswith("/_debug_toolbar"):
                route = RouteModel(
                    endpoint=rule.endpoint,
                    rule=rule.rule,
                    methods=sorted(list(rule.methods)),
                )
                CKAN_ROUTES[rule.endpoint] = route
    if endpoint and endpoint in CKAN_ROUTES.keys():
        return CKAN_ROUTES[endpoint]
    else:
        endpoints = [str(key) for key in CKAN_ROUTES.keys()]
        return f"route endpoint not found. List of endpoints: {endpoints}"


# --- functions for pattern matching


def try_match(pat: str, text: str, max_err: int):
    if pat and text:
        fuzzy_pat = f"({pat})" + f"{{e<={max_err}}}"
        if len(text) <= len(pat):
            raise ValueError(
                f"length of 'text': {text} is smaller then pattern length: {pat}."
            )
        return (
            regex.search(
                fuzzy_pat, text, flags=regex.BESTMATCH | regex.IGNORECASE | regex.DOTALL
            ),
            fuzzy_pat,
        )
    else:
        raise ValueError("The 'pat' and 'text' parameters must be a non-empty strings.")


def _fuzzy_search_sync(
    pattern: str, text: str, threshold: float = 0.8
) -> Tuple[Optional[str], int, int]:
    max_err = max(1, int((1 - threshold) * len(pattern)))
    try:
        match, fuzzy_pat = try_match(pat=pattern, text=text, max_err=max_err)
    except regex.error as e:
        log.debug(f"Initial regex failed for pattern '{pattern[:80]}': {e}")
        match = None

    if not match:
        escaped = regex.escape(pattern)
        try:
            match, fuzzy_pat = try_match(pat=escaped, text=text, max_err=max_err)
        except regex.error as e:
            log.debug(f"Escaped regex also failed for pattern '{escaped[:80]}': {e}")
            return "", -1, -1

    if not match:
        return "", -1, -1

    return match.group(1), match.start(1), match.end(1)


def split_text_into_chunks(text, chunk_size, overlap):
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        if len(chunk) > 0:
            chunks.append((chunk, i))
    return chunks


async def fuzzy_search_early_cancel(
    pattern: str, text: str, threshold: float = 0.8
) -> Tuple[Optional[str], int, int]:
    # Überprüfe, ob der Pattern und der Text gültig sind
    if not pattern or not isinstance(pattern, str):
        raise ValueError("The 'pattern' parameter must be a non-empty string.")

    if not text or not isinstance(text, str):
        raise ValueError("The 'text' parameter must be a non-empty string.")

    start_time = time.perf_counter()
    chunk_size = 10000
    overlap = 1000

    if text and len(text) <= chunk_size:
        result = _fuzzy_search_sync(pattern, text, threshold)
        duration = time.perf_counter() - start_time
        # log.debug(
        #     f"Tried to match: '{pattern}' - found: {result[0] if result[1] >= 0 else 'no match'} - took {duration:.4f} seconds"
        # )
        return result

    tasks = []
    chunks = split_text_into_chunks(text, chunk_size, overlap)
    # Erstelle die Tasks direkt als awaitables
    tasks = [
        asyncio.to_thread(_fuzzy_search_sync, pattern, chunk[0], threshold)
        for chunk in chunks
    ]
    for completed_task in asyncio.as_completed(tasks):
        try:
            match, start, end = await completed_task
            if completed_task not in tasks:
                continue
            if start >= 0:
                #log.debug(f"Completed task: {completed_task}")
                # Finde den Index des abgeschlossenen Tasks in der ursprünglichen Zuordnung
                chunk_idx = tasks.index(completed_task)
                abs_start = chunks[chunk_idx][1] + start
                abs_end = chunks[chunk_idx][1] + end
                duration = time.perf_counter() - start_time
                # log.debug(
                #     f"Tried to match: '{pattern}' - found: {match} at {abs_start}-{abs_end} - took {duration:.4f} seconds"
                # )

                # Cancel all other tasks
                for t in tasks:
                    if not t.done():
                        t.cancel()
                return match, abs_start, abs_end

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(
                f"Error while processing fuzzy_search_early_cancel task:  {str(e)}"
            )

    duration = time.perf_counter() - start_time
    # log.debug(
    #     f"Tried to match: '{pattern}' - no match found - took {duration:.4f} seconds"
    # )
    return "", -1, -1
