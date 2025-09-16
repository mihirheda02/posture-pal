import cv2
import numpy as np
import time
import os
import logging
from dotenv import load_dotenv
from typing import Optional
import signal
import sys

from posture import PostureAnalyzer
from feedback import PostureFeedbackSystem
from speech import create_speech_service, PostureSpeechManager


class PosturePalApp:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.posture_analyzer = None
        self.feedback_system = None
        self.speech_service = None
        self.speech_manager = None
        self.camera = None
        self.running = False
        
        # Configuration
        self.feedback_interval = float(os.getenv('FEEDBACK_INTERVAL_SECONDS', '30'))
        self.confidence_threshold = float(os.getenv('POSTURE_CONFIDENCE_THRESHOLD', '0.7'))
        self.bad_posture_tolerance = float(os.getenv('BAD_POSTURE_TOLERANCE_SECONDS', '10'))
        
        # UI settings
        self.window_name = "Posture Pal - Real-time Monitoring"
        self.display_width = 640
        self.display_height = 480
        
        # Runtime variables
        self.last_feedback_time = 0
        self.session_start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info("Shutdown signal received. Cleaning up...")
        self.stop()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """Initialize all components"""
        self.logger.info("Initializing Posture Pal...")
        
        try:
            # Initialize posture analyzer
            self.posture_analyzer = PostureAnalyzer()
            self.logger.info("Posture analyzer initialized")
            
            # Initialize feedback system
            self.feedback_system = PostureFeedbackSystem()
            self.logger.info("Feedback system initialized")
            
            # Initialize speech service
            self.speech_service = create_speech_service()
            if self.speech_service:
                self.speech_manager = PostureSpeechManager(self.speech_service)
                self.logger.info("Speech service initialized")
            else:
                self.logger.warning("Speech service not available - continuing without voice feedback")
            
            # Initialize camera
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise RuntimeError("Could not open camera")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.logger.info("Camera initialized")
            
            # Welcome message
            if self.speech_manager:
                self.speech_manager.speak_posture_feedback(
                    "Welcome to Posture Pal! I'll help you maintain good posture throughout your day.", 
                    priority=1, 
                    force=True
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def _draw_ui_overlay(self, frame: np.ndarray, posture_score: float, issues: list) -> np.ndarray:
        """Draw UI overlay with posture information"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Create semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (w-10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Posture score
        score_color = (0, 255, 0) if posture_score >= 80 else (0, 255, 255) if posture_score >= 60 else (0, 0, 255)
        cv2.putText(frame, f"Posture Score: {posture_score:.1f}/100", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        
        # Status
        status = self.feedback_system.get_posture_status_message(posture_score)
        cv2.putText(frame, f"Status: {status}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Issues
        if issues:
            issues_text = f"Issues: {', '.join(issues[:2])}"  # Show max 2 issues
            cv2.putText(frame, issues_text, (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Session info
        session_time = int(time.time() - self.session_start_time)
        cv2.putText(frame, f"Session: {session_time//60:02d}:{session_time%60:02d} | FPS: {self.fps:.1f}", 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(frame, "Press 'q' to quit, 's' to speak status", 
                   (20, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def _process_frame(self, frame: np.ndarray) -> tuple:
        """Process a single frame and return posture metrics"""
        # Analyze posture
        metrics = self.posture_analyzer.analyze_frame(frame)
        
        if metrics is None or metrics.confidence < self.confidence_threshold:
            return None, 50, []
        
        # Get posture score and issues
        posture_score = self.posture_analyzer.get_posture_score()
        issues = self.posture_analyzer.get_current_issues()
        
        return metrics, posture_score, issues
    
    def _handle_feedback(self, metrics, posture_score: float, issues: list):
        """Handle feedback generation and speech"""
        current_time = time.time()
        
        # Generate feedback messages
        feedback_messages = self.feedback_system.generate_feedback(metrics, posture_score)
        
        # Check if we should give feedback
        if self.feedback_system.should_give_feedback(self.feedback_interval):
            if feedback_messages and self.speech_manager:
                # Speak the highest priority message
                message = feedback_messages[0]
                self.speech_manager.speak_posture_feedback(
                    message.message, 
                    priority=message.priority
                )
                self.feedback_system.mark_feedback_given()
        
        # Handle urgent feedback (high priority, immediate)
        urgent_messages = [msg for msg in feedback_messages if msg.priority >= 4]
        if urgent_messages and self.speech_manager:
            for msg in urgent_messages[:1]:  # Only one urgent message at a time
                self.speech_manager.speak_posture_feedback(
                    msg.message, 
                    priority=msg.priority, 
                    force=True
                )
    
    def run(self):
        """Main application loop"""
        if not self.initialize():
            self.logger.error("Failed to initialize. Exiting.")
            return
        
        self.running = True
        self.logger.info("Starting posture monitoring...")
        
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
                    
                    # Handle feedback
                    self._handle_feedback(metrics, posture_score, issues)
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
                elif key == ord('s') and self.speech_manager:
                    # Speak current status
                    status = self.feedback_system.get_posture_status_message(posture_score)
                    self.speech_manager.speak_status(f"Current status: {status}")
                elif key == ord(' '):
                    # Space bar to pause/resume speech
                    if self.speech_manager:
                        if self.speech_manager.is_speaking():
                            self.speech_manager.emergency_stop()
                        else:
                            self.speech_manager.speak_status("Speech resumed")
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the application and clean up resources"""
        self.running = False
        self.logger.info("Stopping Posture Pal...")
        
        # Cleanup components
        if self.speech_service:
            self.speech_service.cleanup()
        
        if self.posture_analyzer:
            self.posture_analyzer.cleanup()
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        self.logger.info("Cleanup complete")


def main():
    """Main entry point"""
    print("=" * 50)
    print("🏃‍♂️ POSTURE PAL - Real-time Posture Monitoring")
    print("=" * 50)
    print("Features:")
    print("• Real-time posture analysis using MediaPipe")
    print("• Intelligent feedback with Azure AI Speech")
    print("• Personalized posture improvement suggestions")
    print("• Visual posture score and issue detection")
    print("\nControls:")
    print("• 'q' - Quit application")
    print("• 's' - Speak current status")
    print("• 'space' - Pause/resume speech")
    print("=" * 50)
    
    app = PosturePalApp()
    app.run()


if __name__ == "__main__":
    main()