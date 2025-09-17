import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time


@dataclass
class PostureMetrics:
    head_forward_angle: float
    head_tilt_angle: float
    back_straightness: float
    confidence: float
    timestamp: float
    issues: List[str]


class PostureAnalyzer:
    def __init__(self, model_path: str = "models/pose_landmarker_heavy.task"):
        self.model_path = model_path
        self._detector = None
        self._init_detector()
        
        # Reference angles for good posture (in degrees)
        self.good_head_angle_range = (0, 22)  # Slight forward is normal
        self.good_back_angle_threshold = (-4, 9)
        
        # History for temporal analysis
        self.posture_history = []
        self.history_length = 30  # Keep last 30 frames
        
    def _init_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,  # Changed from LIVE_STREAM for real-time
            num_poses=1,
            min_pose_detection_confidence=0.4,  # Lowered from 0.5
            min_pose_presence_confidence=0.4,   # Lowered from 0.5
            min_tracking_confidence=0.4,        # Lowered from 0.5
            output_segmentation_masks=False
            # Removed result_callback for synchronous processing
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
    
    def _analyze_head_position(self, landmarks) -> Tuple[float, List[str]]:
        """Analyze head posture"""
        issues = []
        
        left_ear = (landmarks[7].x, landmarks[7].y)
        left_shoulder = (landmarks[11].x, landmarks[11].y)
        left_eye = (landmarks[3].x, landmarks[3].y)
        
        # Calculate head forward/backward angle
        # Create a vertical line from shoulder upwards
        vertical_point = (left_shoulder[0], left_ear[1])
        head_forward_angle = self._calculate_angle(vertical_point, left_shoulder, left_ear)
        
        if head_forward_angle > self.good_head_angle_range[1]:
            issues.append("Head too far forward")
            print(f"Debug: head_forward_angle={head_forward_angle}")
        elif head_forward_angle < self.good_head_angle_range[0]:
            issues.append("Head too far backward")
            print(f"Debug: head_forward_angle={head_forward_angle}")

        head_tilt_angle = self.calculate_angle(left_eye, left_ear, (left_eye[0], left_ear[1]))

        if head_tilt_angle > 10:
            if left_eye[1] > left_ear[1]:
                issues.append("Head tilted back")
                print(f"Debug: head_tilt_angle={head_tilt_angle}")
            else:
                issues.append("Head tilted forward")
                print(f"Debug: head_tilt_angle={head_tilt_angle}")
            
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
        
        if left_shoulder[0] > left_hip[0] and back_angle > self.good_back_angle_threshold[1]:
            issues.append("Leaning forward")
        elif left_shoulder[0] < left_hip[0] and back_angle > self.good_back_angle_threshold[1]:
            back_angle = -back_angle  # Negative angle for leaning backward
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
        visible_landmarks = sum(1 for lm in landmarks if lm.visibility > 0.3)
        if visible_landmarks < 10:  # Need at least 10 visible landmarks
            return None
        
        # Calculate posture metrics
        head_forward_angle, head_tilt_angle, head_issues = self._analyze_head_position(landmarks)
        back_angle, back_issues = self._analyze_back_straightness(landmarks)
        
        # Combine all issues
        all_issues = head_issues + back_issues
        
        # Calculate overall confidence (based on landmark visibility)
        visible_landmarks = sum(1 for lm in landmarks if lm.visibility > 0.5)
        confidence = visible_landmarks / len(landmarks)
        
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
        """Calculate overall posture score (0-10)"""
        if not self.posture_history:
            return 5
            
        recent_metrics = self.posture_history[-10:]  # Last 10 frames
        
        total_score = 0
        for metrics in recent_metrics:
            score = 10
            
            # Deduct points for head position
            if metrics.head_forward_angle > 25:
                score -= 4
            elif metrics.head_forward_angle > 20:
                score -= 1
                
            # Deduct points for back straightness forward
            if metrics.back_straightness > 9:
                score -= 3
            elif metrics.back_straightness > 6:
                score -= 2

            # Deduct points for back straightness backward
            if metrics.back_straightness < -4:
                score -= 3
            elif metrics.back_straightness < -2:
                score -= 2
                
            # Deduct points for low confidence
            if metrics.confidence < 0.6:
                score -= 1
                
            total_score += max(0, score)
            
        return total_score / len(recent_metrics)
    
    def is_bad_posture(self, threshold: float = 7) -> bool:
        """Check if current posture is considered bad"""
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