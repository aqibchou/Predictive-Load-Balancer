import yaml
from typing import List, Dict
'''
Helper functions to load and validate backend servers (NOT USED YET)
'''
def load_servers_from_config(config_path: str) -> List[Dict[str, str]]:
    """
    Load backend server configuration from YAML file.
    
    Expected format:
    servers:
      - id: server_1
        url: http://backend_1:8000
      - id: server_2
        url: http://backend_2:8000
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            servers = config.get('servers', [])
            
            if not servers:
                raise ValueError("No servers found in configuration")
            
            for server in servers:
                if 'id' not in server or 'url' not in server:
                    raise ValueError(f"Invalid server config: {server}")
            
            return servers
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}")

def validate_server_health(server_url: str) -> bool:
    """
    Check if a backend server is healthy.
    Returns True if server responds to /health endpoint.
    """
    import httpx
    
    try:
        response = httpx.get(f"{server_url}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False

def discover_healthy_servers(config_path: str) -> List[Dict[str, str]]:
    """
    Load servers from config and filter to only healthy ones.
    """
    all_servers = load_servers_from_config(config_path)
    healthy_servers = []
    
    for server in all_servers:
        if validate_server_health(server['url']):
            healthy_servers.append(server)
            print(f"Server {server['id']} is healthy")
        else:
            print(f"Server {server['id']} is unhealthy (skipping)")
    
    if not healthy_servers:
        raise RuntimeError("No healthy servers available")
    
    return healthy_servers