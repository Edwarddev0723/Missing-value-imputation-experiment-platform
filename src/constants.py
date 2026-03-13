"""Platform-wide constants."""

# Default random seed
DEFAULT_SEED = 42

# Feature type identifiers
NUMERICAL = "numerical"
CATEGORICAL = "categorical"
BINARY = "binary"
ORDINAL = "ordinal"

# Task types
CLASSIFICATION = "classification"
REGRESSION = "regression"

# Missingness mechanisms
MCAR = "MCAR"
MAR = "MAR"
MNAR = "MNAR"
HYBRID = "hybrid"

# Method families
FAMILY_STATISTICAL = "statistical"
FAMILY_ML = "ml"
FAMILY_DL = "dl"

# Run status
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_RUNNING = "running"

# Default device
DEVICE_CPU = "cpu"
DEVICE_MPS = "mps"
DEVICE_CUDA = "cuda"
