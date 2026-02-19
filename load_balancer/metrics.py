from prometheus_client import Counter, Gauge, Histogram, generate_latest

# Request metrics
requests_total = Counter(
    'loadbalancer_requests_total',
    'Total requests processed',
    ['backend_server', 'status']
)

active_servers = Gauge(
    'loadbalancer_active_servers',
    'Current number of active backend servers'
)

# Prophet metrics
predicted_traffic = Gauge(
    'loadbalancer_predicted_traffic_5min',
    'Prophet 5-minute traffic prediction'
)

prediction_confidence_lower = Gauge(
    'loadbalancer_prediction_lower_bound',
    'Prophet prediction lower bound'
)

prediction_confidence_upper = Gauge(
    'loadbalancer_prediction_upper_bound',
    'Prophet prediction upper bound'
)

# Performance metrics
request_latency = Histogram(
    'loadbalancer_request_latency_seconds',
    'Request latency in seconds',
    ['route']
)

cache_hits = Counter(
    'loadbalancer_cache_hits_total',
    'Cache hit count',
    ['server']
)

cache_misses = Counter(
    'loadbalancer_cache_misses_total',
    'Cache miss count',
    ['server']
)

# Q-learning metrics
scaling_actions = Counter(
    'qlearning_actions_total',
    'Scaling actions taken',
    ['action']
)

qlearning_reward = Gauge(
    'qlearning_current_reward',
    'Most recent Q-learning reward'
)

qlearning_epsilon = Gauge(
    'qlearning_epsilon',
    'Current exploration rate'
)

# A* routing metrics
astar_server_scores = Gauge(
    'astar_server_scores',
    'A* heuristic scores per server',
    ['server']
)

def get_metrics():
    return generate_latest()