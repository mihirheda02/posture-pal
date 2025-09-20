import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import logging


@dataclass
class PostureThresholds:
    """Posture detection thresholds"""
    # MediaPipe confidence thresholds (standardized)
    pose_detection_confidence: float = 0.4      # Pose detection confidence
    pose_presence_confidence: float = 0.4       # Pose presence confidence  
    pose_tracking_confidence: float = 0.4       # Pose tracking confidence
    pose_confidence_threshold: float = 0.4      # Minimum confidence for scoring
    
    # Head tilt thresholds (degrees) - only major thresholds
    head_tilt_forward_threshold: float = 40.0   # Forward tilt threshold
    head_tilt_back_threshold: float = 5.0       # Backward tilt threshold
    
    # Back angle thresholds (degrees) - only major thresholds  
    back_angle_forward_threshold: float = 12.0  # Forward lean threshold
    back_angle_backward_threshold: float = -3.0 # Backward lean threshold
    
    # Head forward angle thresholds (degrees) - only major thresholds
    head_forward_threshold: float = 22.0        # Forward head threshold


@dataclass 
class ScoringSettings:
    """Posture scoring configuration"""
    # Base scoring system
    max_score: float = 10.0                    # Maximum possible score
    min_score: float = 0.0                     # Minimum possible score
    
    # Simplified penalties (only major issues penalized)
    head_forward_penalty: float = 3.0          # Penalty for head too far forward
    head_tilt_forward_penalty: float = 2.0     # Penalty for head tilted forward
    head_tilt_back_penalty: float = 2.0        # Penalty for head tilted backward
    back_forward_penalty: float = 2.5          # Penalty for leaning forward
    back_backward_penalty: float = 2.0         # Penalty for leaning backward
    low_confidence_penalty: float = 0.5        # Penalty for low pose confidence
    
    # Scoring behavior
    bad_posture_threshold: float = 7.0         # Score below this is "bad posture"
    history_frames: int = 10                   # Number of frames to average for score


@dataclass
class FeedbackSettings:
    """Feedback system settings"""
    feedback_interval_seconds: float = 180.0       # Changed to 3 minutes
    bad_posture_tolerance_seconds: float = 180.0   # Changed to 3 minutes 
    min_speech_interval: float = 60.0              # 1 minute minimum between speech
    max_feedback_messages: int = 2
    encouragement_probability: float = 0.3
    tip_probability: float = 0.1


@dataclass
class DisplaySettings:
    """Display and UI settings"""
    window_width: int = 640
    window_height: int = 480
    show_landmarks: bool = True
    show_pose_score: bool = True
    show_session_timer: bool = True
    fps_target: int = 30


@dataclass
class SpeechSettings:
    """Azure Speech service settings"""
    voice_name: str = "en-US-AriaNeural"
    speech_rate: float = 0.9
    speech_pitch: str = "medium"
    enable_ssml: bool = True
    volume: float = 1.0


@dataclass
class AppConfig:
    """Main application configuration"""
    posture_thresholds: PostureThresholds
    scoring_settings: ScoringSettings
    feedback_settings: FeedbackSettings
    display_settings: DisplaySettings
    speech_settings: SpeechSettings
    
    # Azure credentials (loaded from environment)
    azure_speech_key: Optional[str] = None
    azure_speech_endpoint: Optional[str] = None
    
    # Model path
    model_path: str = "models/pose_landmarker_heavy.task"
    
    # Debug settings
    debug_mode: bool = False
    log_level: str = "INFO"


class ConfigManager:
    """Manages application configuration"""
    
    DEFAULT_CONFIG_FILE = "posture_pal_config.json"
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.config = self._create_default_config()
        self.logger = logging.getLogger(__name__)
    
    def _create_default_config(self) -> AppConfig:
        """Create default configuration"""
        return AppConfig(
            posture_thresholds=PostureThresholds(),
            scoring_settings=ScoringSettings(),
            feedback_settings=FeedbackSettings(),
            display_settings=DisplaySettings(),
            speech_settings=SpeechSettings()
        )
    
    def load_config(self) -> AppConfig:
        """Load configuration from file and environment variables"""
        # Load from file if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self.config = self._dict_to_config(config_data)
                self.logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load config file: {e}. Using defaults.")
        
        # Override with environment variables
        self._load_from_environment()
        
        return self.config
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'AZURE_SPEECH_KEY': ('azure_speech_key', str),
            'AZURE_SPEECH_ENDPOINT': ('azure_speech_endpoint', str),
            'FEEDBACK_INTERVAL_SECONDS': ('feedback_settings.feedback_interval_seconds', float),
            'POSTURE_CONFIDENCE_THRESHOLD': ('posture_thresholds.pose_confidence_threshold', float),
            'BAD_POSTURE_TOLERANCE_SECONDS': ('feedback_settings.bad_posture_tolerance_seconds', float),
            'DEBUG_MODE': ('debug_mode', bool),
            'LOG_LEVEL': ('log_level', str),
            'MODEL_PATH': ('model_path', str),
            'VOICE_NAME': ('speech_settings.voice_name', str),
            'WINDOW_WIDTH': ('display_settings.window_width', int),
            'WINDOW_HEIGHT': ('display_settings.window_height', int),
        }
        
        for env_var, (config_path, var_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Convert environment variable to appropriate type
                    if var_type == bool:
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif var_type == float:
                        value = float(env_value)
                    elif var_type == int:
                        value = int(env_value)
                    else:
                        value = env_value
                    
                    # Set nested attribute
                    self._set_nested_attribute(self.config, config_path, value)
                    self.logger.debug(f"Set {config_path} = {value} from environment")
                    
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid environment variable {env_var}={env_value}: {e}")
    
    def _set_nested_attribute(self, obj, path: str, value):
        """Set nested attribute using dot notation"""
        parts = path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    def save_config(self, config: Optional[AppConfig] = None):
        """Save configuration to file"""
        if config:
            self.config = config
        
        try:
            config_dict = self._config_to_dict(self.config)
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            self.logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert config object to dictionary"""
        result = {}
        for key, value in asdict(config).items():
            if isinstance(value, dict):
                result[key] = value
            else:
                result[key] = value
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to config object"""
        # Create default config and update with loaded values
        config = self._create_default_config()
        
        # Update posture thresholds
        if 'posture_thresholds' in config_dict:
            for key, value in config_dict['posture_thresholds'].items():
                if hasattr(config.posture_thresholds, key):
                    setattr(config.posture_thresholds, key, value)
        
        # Update scoring settings
        if 'scoring_settings' in config_dict:
            for key, value in config_dict['scoring_settings'].items():
                if hasattr(config.scoring_settings, key):
                    setattr(config.scoring_settings, key, value)
        
        # Update feedback settings
        if 'feedback_settings' in config_dict:
            for key, value in config_dict['feedback_settings'].items():
                if hasattr(config.feedback_settings, key):
                    setattr(config.feedback_settings, key, value)
        
        # Update display settings
        if 'display_settings' in config_dict:
            for key, value in config_dict['display_settings'].items():
                if hasattr(config.display_settings, key):
                    setattr(config.display_settings, key, value)
        
        # Update speech settings
        if 'speech_settings' in config_dict:
            for key, value in config_dict['speech_settings'].items():
                if hasattr(config.speech_settings, key):
                    setattr(config.speech_settings, key, value)
        
        # Update main config
        for key in ['azure_speech_key', 'azure_speech_endpoint', 'model_path', 'debug_mode', 'log_level']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        return self.config
    
    def update_posture_thresholds(self, **kwargs):
        """Update posture detection thresholds"""
        for key, value in kwargs.items():
            if hasattr(self.config.posture_thresholds, key):
                setattr(self.config.posture_thresholds, key, value)
    
    def update_scoring_settings(self, **kwargs):
        """Update scoring settings"""
        for key, value in kwargs.items():
            if hasattr(self.config.scoring_settings, key):
                setattr(self.config.scoring_settings, key, value)
    
    def update_feedback_settings(self, **kwargs):
        """Update feedback settings"""
        for key, value in kwargs.items():
            if hasattr(self.config.feedback_settings, key):
                setattr(self.config.feedback_settings, key, value)
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self._create_default_config()
        self.logger.info("Configuration reset to defaults")
    
    def validate_config(self) -> bool:
        """Validate configuration values"""
        try:
            # Validate posture thresholds
            pt = self.config.posture_thresholds
            assert 0 < pt.pose_confidence_threshold <= 1, "pose_confidence_threshold must be between 0 and 1"
            
            # Validate scoring settings
            sc = self.config.scoring_settings
            assert sc.max_score > sc.min_score, "max_score must be greater than min_score"
            assert sc.max_score > 0, "max_score must be positive"
            assert sc.bad_posture_threshold <= sc.max_score, "bad_posture_threshold must be <= max_score"
            assert sc.history_frames > 0, "history_frames must be positive"
            assert all(penalty >= 0 for penalty in [
                sc.head_forward_penalty, sc.head_tilt_forward_penalty, 
                sc.head_tilt_back_penalty, sc.back_forward_penalty,
                sc.back_backward_penalty, sc.low_confidence_penalty
            ]), "All penalties must be non-negative"
            
            # Validate feedback settings
            fs = self.config.feedback_settings
            assert fs.feedback_interval_seconds > 0, "feedback_interval_seconds must be positive"
            assert fs.min_speech_interval > 0, "min_speech_interval must be positive"
            assert 0 <= fs.encouragement_probability <= 1, "encouragement_probability must be between 0 and 1"
            
            # Validate display settings
            ds = self.config.display_settings
            assert ds.window_width > 0 and ds.window_height > 0, "Window dimensions must be positive"
            assert ds.fps_target > 0, "fps_target must be positive"
            
            # Validate speech settings
            ss = self.config.speech_settings
            assert 0.5 <= ss.speech_rate <= 2.0, "speech_rate must be between 0.5 and 2.0"
            assert 0 <= ss.volume <= 1.0, "volume must be between 0 and 1"
            
            return True
            
        except AssertionError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False


# Global config instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def load_config() -> AppConfig:
    """Load and return application configuration"""
    return get_config_manager().load_config()

def save_config(config: AppConfig):
    """Save application configuration"""
    get_config_manager().save_config(config)