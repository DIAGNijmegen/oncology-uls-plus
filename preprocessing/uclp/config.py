from pathlib import Path
from typing import Any
import yaml


def load_config(yaml_path: str) -> dict[str, Any]:
    """Load and validate YAML configuration file"""
    config_path = Path(yaml_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    validate_config(config)
    apply_defaults(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required configuration fields"""
    required_fields = [
        'name',
        'id',
        'input_images',
        'crop_size',
        'num_augmentations'
    ]
    
    # input_labels is required unless using adapters that find labels in same location
    adapter = config.get('adapter', 'default')
    adapters_without_separate_labels = ['same_folder', 'worc', 'clm', 'longitudinal_ct', 'longitudinal_ct_test']
    if adapter not in adapters_without_separate_labels and 'input_labels' not in config:
        required_fields.append('input_labels')
    
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")
    
    if not isinstance(config['id'], int) or config['id'] < 0:
        raise ValueError("Dataset 'id' must be a non-negative integer")
    
    if not isinstance(config['num_augmentations'], int) or config['num_augmentations'] < 1:
        raise ValueError("'num_augmentations' must be a positive integer")
    
    if not isinstance(config['crop_size'], list) or len(config['crop_size']) != 3:
        raise ValueError("'crop_size' must be a list of 3 integers [x, y, z]")
    
    if not all(isinstance(s, int) and s > 0 for s in config['crop_size']):
        raise ValueError("All crop_size values must be positive integers")
    
    # Validate paths exist
    if not Path(config['input_images']).exists():
        raise ValueError(f"Path does not exist: {config['input_images']}")
    
    if 'input_labels' in config and not Path(config['input_labels']).exists():
        raise ValueError(f"Path does not exist: {config['input_labels']}")

    # Optional lesion type CSV directory
    if 'lesion_types_dir' in config:
        if not Path(config['lesion_types_dir']).exists():
            raise ValueError(f"Path does not exist: {config['lesion_types_dir']}")


def apply_defaults(config: dict[str, Any]) -> None:
    """Apply default values for optional parameters"""
    defaults = {
        'output_dir': './nnUNet_raw/',
        'random_seed': 42,
        'body_threshold_hu': -500,
        'max_sampling_attempts': 50,
        'min_lesion_size_voxels': 50,
        'adapter': 'default',
        'append_lesion_type': False,
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    # Validate adapter-specific requirements
    if config.get('adapter') == 'same_folder':
        if 'image_pattern' not in config or 'label_pattern' not in config:
            raise ValueError("same_folder adapter requires 'image_pattern' and 'label_pattern' in config")
