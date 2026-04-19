# Example configuration for data augmentation
# Copy this to config.py and modify as needed

from data_augmentation import AugmentationConfig

# Default configuration for research/development
DEFAULT_CONFIG = AugmentationConfig(
    rotation_limit=45,          # ±45 degrees rotation
    crop_ratio_min=0.7,         # Minimum 70% of original size
    crop_ratio_max=1.0,         # Maximum 100% (no crop)
    zoom_min=0.8,              # 80% zoom out
    zoom_max=1.2,              # 120% zoom in
    num_augmentations=5,        # 5 augmentations per sample
    output_size=(640, 480),     # Standard VGA resolution
    brightness_limit=0.2,       # ±20% brightness variation
    contrast_limit=0.2          # ±20% contrast variation
)

# Conservative configuration (less aggressive augmentation)
CONSERVATIVE_CONFIG = AugmentationConfig(
    rotation_limit=30,
    crop_ratio_min=0.8,
    crop_ratio_max=1.0,
    zoom_min=0.9,
    zoom_max=1.1,
    num_augmentations=3,
    output_size=(640, 480),
    brightness_limit=0.1,
    contrast_limit=0.1
)

# Aggressive configuration (maximum augmentation)
AGGRESSIVE_CONFIG = AugmentationConfig(
    rotation_limit=60,
    crop_ratio_min=0.6,
    crop_ratio_max=1.0,
    zoom_min=0.7,
    zoom_max=1.3,
    num_augmentations=10,
    output_size=(640, 480),
    brightness_limit=0.3,
    contrast_limit=0.3
)

# High resolution configuration
HIGH_RES_CONFIG = AugmentationConfig(
    rotation_limit=45,
    crop_ratio_min=0.7,
    crop_ratio_max=1.0,
    zoom_min=0.8,
    zoom_max=1.2,
    num_augmentations=5,
    output_size=(1024, 768),    # Higher resolution output
    brightness_limit=0.2,
    contrast_limit=0.2
)