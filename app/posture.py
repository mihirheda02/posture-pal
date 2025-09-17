import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time
from config import AppConfig, load_config


@dataclass
class PostureMetrics:
    head_forward_angle: float
    head_tilt_angle: float
    back_straightness: float
    confidence: float
    timestamp: float
    issues: List[str]


class PostureAnalyzer:
    def __init__(self, model_path: str = "models/pose_landmarker_heavy.task", config: Optional[AppConfig] = None):
        self.model_path = model_path
        self.config = config or load_config()
        self._detector = None
        self._init_detector()
        
        # History for temporal analysis
        self.posture_history = []
        self.history_length = 30  # Keep last 30 frames
        
    def _init_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        
        # Use configurable confidence thresholds
        thresholds = self.config.posture_thresholds
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=thresholds.pose_detection_confidence,
            min_pose_presence_confidence=thresholds.pose_presence_confidence,
            min_tracking_confidence=thresholds.pose_tracking_confidence,
            output_segmentation_masks=False
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)
        self._latest_result = None
        
    def _calculate_angle(self, point1: Tuple[float, float], point2: Tuple[float, float], point3: Tuple[float, float]) -> float:
        """Calculate angle between three points"""
        v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms == 0:
            return 0
            
        cos_angle = dot_product / norms
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _analyze_head_position(self, landmarks) -> Tuple[float, float, List[str]]:
        """Analyze head posture"""
        issues = []
        
        left_ear = (landmarks[7].x, landmarks[7].y)
        left_shoulder = (landmarks[11].x, landmarks[11].y)
        nose = (landmarks[0].x, landmarks[0].y)
        
        # Calculate head forward/backward angle
        # Create a vertical line from shoulder upwards
        vertical_point = (left_shoulder[0], left_ear[1])
        head_forward_angle = self._calculate_angle(vertical_point, left_shoulder, left_ear)
        
        # Use simplified thresholds for head forward/backward position
        thresholds = self.config.posture_thresholds
        if head_forward_angle > thresholds.head_forward_threshold:
            issues.append("Head too far forward")

        # Calculate head tilt angle
        head_tilt_angle = self._calculate_angle(nose, left_ear, (nose[0], left_ear[1]))

        # Use simplified thresholds for head tilt
        if nose[1] > left_ear[1]:  # Head tilted forward
            if head_tilt_angle > thresholds.head_tilt_forward_threshold:
                issues.append("Head tilted forward")
        elif nose[1] < left_ear[1]:  # Head tilted back
            if head_tilt_angle > thresholds.head_tilt_back_threshold:
                issues.append("Head tilted back")
            
        return head_forward_angle, head_tilt_angle, issues
    
    def _analyze_back_straightness(self, landmarks) -> Tuple[float, List[str]]:
        """Analyze back straightness"""
        issues = []
        
        left_shoulder = (landmarks[11].x, landmarks[11].y)
        left_hip = (landmarks[23].x, landmarks[23].y)
        
        # Calculate spine angle from vertical
        spine_vector = (left_shoulder[0] - left_hip[0], left_shoulder[1] - left_hip[1])
        vertical_vector = (0, -1)
        
        dot_product = spine_vector[1] * vertical_vector[1]
        spine_magnitude = np.sqrt(spine_vector[0]**2 + spine_vector[1]**2)
        
        if spine_magnitude == 0:
            back_angle = 0
        else:
            cos_angle = dot_product / spine_magnitude
            cos_angle = np.clip(cos_angle, -1, 1)
            back_angle = np.degrees(np.arccos(cos_angle))
        
        # Use simplified thresholds for back position
        thresholds = self.config.posture_thresholds
        
        if left_shoulder[0] > left_hip[0]:  # Leaning forward
            if back_angle > thresholds.back_angle_forward_threshold:
                issues.append("Leaning forward")
        elif left_shoulder[0] < left_hip[0]:  # Leaning backward
            back_angle = -back_angle  # Negative angle for leaning backward
            if back_angle < thresholds.back_angle_backward_threshold:
                issues.append("Leaning backward")
                
        return back_angle, issues
    
    def analyze_frame(self, frame: np.ndarray) -> Optional[PostureMetrics]:
        """Analyze posture in a single frame"""
        if self._detector is None:
            return None
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process frame synchronously for real-time results
        self._latest_result = self._detector.detect(mp_image)
        
        if self._latest_result is None or not self._latest_result.pose_landmarks:
            return None
            
        landmarks = self._latest_result.pose_landmarks[0]
        
        # Check if we have enough visible landmarks
        visible_landmarks = sum(1 for lm in landmarks if lm.visibility > 0.4)
        if visible_landmarks < 8:  # Lowered for better detection
            return None
        
        # Calculate posture metrics
        head_forward_angle, head_tilt_angle, head_issues = self._analyze_head_position(landmarks)
        back_angle, back_issues = self._analyze_back_straightness(landmarks)
        
        # Combine all issues
        all_issues = head_issues + back_issues
        
        # Calculate overall confidence (based on landmark visibility)
        confidence = visible_landmarks / len(landmarks)  # Use consistent threshold
        
        metrics = PostureMetrics(
            head_forward_angle=head_forward_angle,
            head_tilt_angle=head_tilt_angle,
            back_straightness=back_angle,
            confidence=confidence,
            timestamp=time.time(),
            issues=all_issues
        )
        
        # Add to history
        self.posture_history.append(metrics)
        if len(self.posture_history) > self.history_length:
            self.posture_history.pop(0)
            
        return metrics
    
    def get_posture_score(self) -> float:
        """Calculate overall posture score using simplified scoring system"""
        if not self.posture_history:
            return self.config.scoring_settings.max_score / 2  # Default to middle score
            
        # Use configurable number of frames for averaging
        num_frames = min(len(self.posture_history), self.config.scoring_settings.history_frames)
        recent_metrics = self.posture_history[-num_frames:]
        
        total_score = 0
        scoring = self.config.scoring_settings
        thresholds = self.config.posture_thresholds
        
        for metrics in recent_metrics:
            score = scoring.max_score  # Start with maximum score
            
            # Deduct points for head position issues (simplified - only major penalties)
            issues = metrics.issues
            if "Head too far forward" in issues:
                score -= scoring.head_forward_penalty
                
            # Deduct points for head tilt issues
            if "Head tilted forward" in issues:
                score -= scoring.head_tilt_forward_penalty
            elif "Head tilted back" in issues:
                score -= scoring.head_tilt_back_penalty
                
            # Deduct points for back posture issues
            if "Leaning forward" in issues:
                score -= scoring.back_forward_penalty
            elif "Leaning backward" in issues:
                score -= scoring.back_backward_penalty
                
            # Deduct points for low confidence
            if metrics.confidence < thresholds.pose_confidence_threshold:
                score -= scoring.low_confidence_penalty
                
            # Ensure score doesn't go below minimum
            score = max(scoring.min_score, score)
            total_score += score
            
        return total_score / len(recent_metrics)
    
    def is_bad_posture(self, threshold: Optional[float] = None) -> bool:
        """Check if current posture is considered bad"""
        if threshold is None:
            threshold = self.config.scoring_settings.bad_posture_threshold
        return self.get_posture_score() < threshold
    
    def get_current_issues(self) -> List[str]:
        """Get current posture issues"""
        if not self.posture_history:
            return []
        return self.posture_history[-1].issues
    
    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw pose landmarks on frame"""
        if self._latest_result is None or not self._latest_result.pose_landmarks:
            return frame
            
        landmarks = self._latest_result.pose_landmarks[0]
        h, w = frame.shape[:2]
        
        # Define all pose connections (MediaPipe pose connections)
        pose_connections = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            # Torso
            (9, 10),
            (11, 12), (11, 13), (12, 14),
            (13, 15), (14, 16),
            (11, 23), (12, 24), (23, 24),
            # Left arm
            (15, 17), (15, 19), (15, 21),
            (17, 19),
            # Right arm  
            (16, 18), (16, 20), (16, 22),
            (18, 20),
            # Left leg
            (23, 25), (25, 27), (27, 29), (29, 31),
            (27, 31),
            # Right leg
            (24, 26), (26, 28), (28, 30), (30, 32),
            (28, 32)
        ]
        
        # Draw connections first (behind landmarks)
        for start_idx, end_idx in pose_connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]
                
                # Only draw if both landmarks are visible
                if start_lm.visibility > 0.3 and end_lm.visibility > 0.3:
                    start_x, start_y = int(start_lm.x * w), int(start_lm.y * h)
                    end_x, end_y = int(end_lm.x * w), int(end_lm.y * h)
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Draw landmarks on top
        for i, lm in enumerate(landmarks):
            if lm.visibility > 0.3:
                x, y = int(lm.x * w), int(lm.y * h)
                # Different colors for different body parts
                if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Face
                    color = (255, 255, 0)  # Yellow
                elif i in [11, 12, 13, 14, 15, 16]:  # Arms/shoulders
                    color = (0, 0, 255)  # Red
                elif i in [23, 24, 25, 26, 27, 28]:  # Legs/hips
                    color = (255, 0, 255)  # Magenta
                else:  # Hands and feet
                    color = (0, 255, 255)  # Cyan
                    
                cv2.circle(frame, (x, y), 4, color, -1)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)  # White border
                
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        if self._detector:
            self._detector.close()
            self._detector = None