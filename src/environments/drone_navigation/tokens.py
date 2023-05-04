from collections import OrderedDict

# Dictionaries sorting Tokens by hierarchy. Hierarchies are based on game importance and similarity.
GROUND_TOKENS = OrderedDict(
    {"0": "Ground"}
)
OBSTACLE_TOKENS = OrderedDict(
    {"1": "Wall", "2": "LowObstacles"}
)

TOKEN_DOWNSAMPLING_HIERARCHY = [
    GROUND_TOKENS,
    OBSTACLE_TOKENS
]

TOKENS = OrderedDict(
    {**GROUND_TOKENS, **OBSTACLE_TOKENS}
)

TOKEN_GROUPS = [GROUND_TOKENS, OBSTACLE_TOKENS]

TOKEN_LIST = list(TOKENS.keys())
