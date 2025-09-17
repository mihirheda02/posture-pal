import cv2
import numpy as np
import time
import logging
import traceback

from posture import PostureAnalyzer
from feedback import PostureFeedbackSystem


class PosturePalTestApp:
    def __init__(self):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.posture_analyzer = None
        self.feedback_system = None
        self.camera = None
        self.running = False
        
        # UI settings
        self.window_name = "Posture Pal - TEST MODE (No Speech)"
        self.display_width = 640
        self.display_height = 480
        
        # Runtime variables
        self.session_start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Feedback timing
        self.last_feedback_update = 0
        self.feedback_interval = 5.0  # Update feedback every 5 seconds
        self.current_feedback_messages = []
    
    def initialize(self) -> bool:
        """Initialize all components"""
        self.logger.info("Initializing Posture Pal Test Mode...")
        
        try:
            # Initialize posture analyzer
            self.posture_analyzer = PostureAnalyzer()
            self.logger.info("Posture analyzer initialized")
            
            # Initialize feedback system
            self.feedback_system = PostureFeedbackSystem()
            self.logger.info("Feedback system initialized")
            
            # Initialize camera
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise RuntimeError("Could not open camera")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.logger.info("Camera initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return False
    
    def _draw_ui_overlay(self, frame: np.ndarray, posture_score: float, issues: list) -> np.ndarray:
        """Draw UI overlay with posture information"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Create semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (w-10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Posture score
        score_color = (0, 255, 0) if posture_score >= 8 else (0, 255, 255) if posture_score >= 6 else (0, 0, 255)
        cv2.putText(frame, f"Posture Score: {posture_score:.1f}/10", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        
        # Status
        status = self.feedback_system.get_posture_status_message(posture_score)
        status_color = (255, 255, 255)
            
        cv2.putText(frame, f"Status: {status}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Issues
        if issues:
            issues_text = f"Issues: {', '.join(issues[:2])}"  # Show max 2 issues
            cv2.putText(frame, issues_text, (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Feedback messages (text only) - Update every 5 seconds
        current_time = time.time()
        if current_time - self.last_feedback_update >= self.feedback_interval:
            self.current_feedback_messages = []
            if self.posture_analyzer and self.posture_analyzer.posture_history:
                self.current_feedback_messages = self.feedback_system.generate_feedback(
                    self.posture_analyzer.posture_history[-1], 
                    posture_score
                )
            self.last_feedback_update = current_time
        
        if self.current_feedback_messages:
            feedback_text = self.current_feedback_messages[0].message[:50] + "..." if len(self.current_feedback_messages[0].message) > 50 else self.current_feedback_messages[0].message
            cv2.putText(frame, f"Suggestion: {feedback_text}", (20, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Session info
        session_time = int(time.time() - self.session_start_time)
        
        # Add debug info
        debug_info = ""
        if hasattr(self.posture_analyzer, '_latest_result') and self.posture_analyzer._latest_result:
            if self.posture_analyzer._latest_result.pose_landmarks:
                landmarks = self.posture_analyzer._latest_result.pose_landmarks[0]
                visible_count = sum(1 for lm in landmarks if lm.visibility > 0.3)
                debug_info = f" | Landmarks: {visible_count}/33"
            else:
                debug_info = " | No landmarks"
        else:
            debug_info = " | No result"
        
        cv2.putText(frame, f"Session: {session_time//60:02d}:{session_time%60:02d} | FPS: {self.fps:.1f}{debug_info}", 
                   (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Test mode notice
        cv2.putText(frame, "TEST MODE - Visual feedback only (no speech)", 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Controls
        cv2.putText(frame, "Press 'q' to quit", 
                   (20, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def _process_frame(self, frame: np.ndarray) -> tuple:
        """Process a single frame and return posture metrics"""
        # Analyze posture
        metrics = self.posture_analyzer.analyze_frame(frame)

        current_time = time.time()
        # log metrics every 3 seconds
        if current_time % 3 < 0.1:
            self.logger.info(f"Metrics: {metrics}")
        
        if metrics is None or metrics.confidence < 0.5:  # Lowered threshold
            return None, 50, []
        
        # Get posture score and issues
        posture_score = self.posture_analyzer.get_posture_score()
        issues = self.posture_analyzer.get_current_issues()
        
        return metrics, posture_score, issues
    
    def run(self):
        """Main application loop"""
        if not self.initialize():
            self.logger.error("Failed to initialize. Exiting.")
            return
        
        self.running = True
        self.logger.info("Starting posture monitoring in test mode...")
        
        frame_time_start = time.time()
        frame_count = 0
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.error("Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                metrics, posture_score, issues = self._process_frame(frame)
                
                # Draw pose landmarks
                if metrics:
                    frame = self.posture_analyzer.draw_landmarks(frame)
                else:
                    # No pose detected
                    posture_score = 50
                    issues = ["No pose detected"]
                
                # Draw UI overlay
                frame = self._draw_ui_overlay(frame, posture_score, issues)
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    self.fps = 30 / (current_time - frame_time_start)
                    frame_time_start = current_time
                
                # Display frame
                cv2.imshow(self.window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the application and clean up resources"""
        self.running = False
        self.logger.info("Stopping Posture Pal Test...")
        
        if self.posture_analyzer:
            self.posture_analyzer.cleanup()
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        self.logger.info("Cleanup complete")


def main():
    """Main entry point"""
    print("=" * 60)
    print("🏃‍♂️ POSTURE PAL - TEST MODE (Visual Only)")
    print("=" * 60)
    print("Features:")
    print("• Real-time posture analysis using MediaPipe")
    print("• Camera-adaptive shoulder detection")
    print("• Visual posture score and issue detection")
    print("• Text-based feedback suggestions")
    print("• No speech (test mode)")
    print("\nControls:")
    print("• 'q' - Quit application")
    print("=" * 60)
    
    app = PosturePalTestApp()
    app.run()


if __name__ == "__main__":
    main()