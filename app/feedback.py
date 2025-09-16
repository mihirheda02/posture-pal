import random
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from posture import PostureMetrics


@dataclass
class FeedbackMessage:
    message: str
    priority: int  # 1-5, higher is more urgent
    category: str  # "correction", "encouragement", "tip", "warning"


class PostureFeedbackSystem:
    def __init__(self):
        self.last_feedback_time = 0
        self.feedback_history = []
        self.consecutive_bad_posture_count = 0
        self.last_posture_good = True
        
        # Feedback messages organized by issue type
        self.feedback_messages = {
            "Head too far forward": {
                "corrections": [
                    "Pull your chin back and imagine a string pulling the top of your head up.",
                    "Tuck your chin slightly and align your ears over your shoulders.",
                    "Think about making a double chin - this helps retract your head position.",
                    "Imagine you're holding a tennis ball between your chin and chest."
                ],
                "tips": [
                    "Place a sticky note on your monitor as a reminder to check your head position.",
                    "Set your screen at eye level to avoid craning your neck forward.",
                    "Take regular breaks to do neck stretches and resets."
                ]
            },
            "Head tilted back": {
                "corrections": [
                    "Gently lower your chin and bring your head forward slightly.",
                    "Relax your neck and let your head find a neutral position.",
                    "Check if your screen is too low - this might be causing you to tilt back."
                ]
            },
            "Right shoulder elevated": {
                "corrections": [
                    "Lower your right shoulder and take a deep breath to relax.",
                    "Roll your right shoulder backward and down several times.",
                    "Check if your right armrest is too high or if you're tensing up."
                ],
                "tips": [
                    "Make sure both armrests are at the same height.",
                    "Consider if you're favoring one side due to mouse usage."
                ]
            },
            "Left shoulder elevated": {
                "corrections": [
                    "Lower your left shoulder and take a deep breath to relax.",
                    "Roll your left shoulder backward and down several times.",
                    "Check if your left armrest is too high or if you're tensing up."
                ],
                "tips": [
                    "Make sure both armrests are at the same height.",
                    "Pay attention to whether you're leaning on one elbow more than the other."
                ]
            },
            "Shoulders rounded forward": {
                "corrections": [
                    "Pull your shoulder blades back and down, opening up your chest.",
                    "Imagine squeezing a pencil between your shoulder blades.",
                    "Roll your shoulders up, back, and down to reset their position.",
                    "Take a deep breath and let your chest expand naturally."
                ],
                "tips": [
                    "Strengthen your upper back muscles with simple exercises.",
                    "Be mindful of how you position your arms while typing.",
                    "Consider a ergonomic chair that supports proper shoulder alignment."
                ]
            },
            "Leaning forward": {
                "corrections": [
                    "Sit back in your chair and use the backrest for support.",
                    "Engage your core muscles to support your spine.",
                    "Move your chair closer to your desk instead of leaning forward.",
                    "Check that your feet are flat on the floor or footrest."
                ],
                "tips": [
                    "Adjust your chair so you can sit back while still reaching your keyboard.",
                    "Use a lumbar support pillow if your chair doesn't provide enough support.",
                    "Keep frequently used items within easy reach to avoid leaning."
                ]
            },
            "Leaning backward": {
                "corrections": [
                    "Sit up straighter and engage your core muscles.",
                    "Move slightly forward in your chair to find a more upright position.",
                    "Make sure your feet are properly supported on the floor."
                ],
                "tips": [
                    "Check if your chair recline is too far back for work tasks.",
                    "Ensure your monitor is at the right distance and height."
                ]
            }
        }
        
        self.encouragement_messages = [
            "Great job! Your posture is looking much better.",
            "Excellent! Keep up the good posture awareness.",
            "Nice correction! Your body will thank you for this.",
            "Perfect! You're developing great posture habits.",
            "Well done! You caught that posture issue quickly.",
            "Fantastic! Your spine alignment is much improved.",
            "Keep it up! Good posture is becoming your natural state.",
            "Excellent self-awareness! You're mastering posture control."
        ]
        
        self.general_tips = [
            "Remember to take movement breaks every 30 minutes.",
            "Stay hydrated to keep your muscles and joints healthy.",
            "Consider doing some gentle stretches at your desk.",
            "Good posture is a journey, not a destination. Keep practicing!",
            "Your future self will thank you for maintaining good posture today.",
            "Small adjustments throughout the day make a big difference.",
            "Breathing deeply can help reset your posture naturally.",
            "Think of your spine as a stack of building blocks - keep them aligned!"
        ]
        
        self.warning_messages = [
            "You've been in poor posture for a while. Time for a posture check!",
            "Your posture needs attention. Let's make some adjustments.",
            "Extended poor posture detected. Please take a moment to reset.",
            "Time for a posture reset! Your body will feel much better.",
            "Poor posture alert! A quick adjustment can prevent discomfort later."
        ]
    
    def generate_feedback(self, posture_metrics: PostureMetrics, posture_score: float) -> List[FeedbackMessage]:
        """Generate appropriate feedback based on current posture"""
        current_time = time.time()
        feedback_messages = []
        
        # Track consecutive bad posture
        if posture_score < 70:
            if self.last_posture_good:
                self.consecutive_bad_posture_count = 1
            else:
                self.consecutive_bad_posture_count += 1
            self.last_posture_good = False
        else:
            if not self.last_posture_good and posture_score > 80:
                # Improved posture - give encouragement
                feedback_messages.append(FeedbackMessage(
                    message=random.choice(self.encouragement_messages),
                    priority=2,
                    category="encouragement"
                ))
            self.consecutive_bad_posture_count = 0
            self.last_posture_good = True
        
        # Generate specific corrections for detected issues
        if posture_metrics.issues:
            for issue in posture_metrics.issues:
                if issue in self.feedback_messages:
                    issue_feedback = self.feedback_messages[issue]
                    
                    # Choose correction message
                    if "corrections" in issue_feedback:
                        correction_msg = random.choice(issue_feedback["corrections"])
                        priority = 4 if self.consecutive_bad_posture_count > 5 else 3
                        
                        feedback_messages.append(FeedbackMessage(
                            message=correction_msg,
                            priority=priority,
                            category="correction"
                        ))
                    
                    # Add tips occasionally
                    if "tips" in issue_feedback and random.random() < 0.3:
                        tip_msg = random.choice(issue_feedback["tips"])
                        feedback_messages.append(FeedbackMessage(
                            message=tip_msg,
                            priority=1,
                            category="tip"
                        ))
        
        # Warning for extended poor posture
        if self.consecutive_bad_posture_count > 10:
            warning_msg = random.choice(self.warning_messages)
            feedback_messages.append(FeedbackMessage(
                message=warning_msg,
                priority=5,
                category="warning"
            ))
        
        # General tips occasionally when posture is good
        elif posture_score > 80 and random.random() < 0.1:
            tip_msg = random.choice(self.general_tips)
            feedback_messages.append(FeedbackMessage(
                message=tip_msg,
                priority=1,
                category="tip"
            ))
        
        # Sort by priority (highest first)
        feedback_messages.sort(key=lambda x: x.priority, reverse=True)
        
        return feedback_messages[:2]  # Return max 2 messages to avoid overwhelming
    
    def should_give_feedback(self, min_interval: float = 15.0) -> bool:
        """Check if enough time has passed to give new feedback"""
        current_time = time.time()
        return (current_time - self.last_feedback_time) >= min_interval
    
    def mark_feedback_given(self):
        """Mark that feedback was just given"""
        self.last_feedback_time = time.time()
    
    def get_posture_status_message(self, posture_score: float) -> str:
        """Get a general status message based on posture score"""
        if posture_score >= 90:
            return "Excellent posture!"
        elif posture_score >= 80:
            return "Good posture!"
        elif posture_score >= 70:
            return "Fair posture - minor adjustments needed."
        elif posture_score >= 60:
            return "Poor posture - needs attention."
        else:
            return "Very poor posture - immediate correction recommended."
    
    def get_smart_suggestion(self, issues: List[str], posture_score: float) -> str:
        """Generate a smart, contextual suggestion"""
        if not issues:
            if posture_score > 85:
                return random.choice([
                    "You're doing great! Keep maintaining this excellent posture.",
                    "Perfect posture! Remember to take a movement break soon.",
                    "Excellent alignment! Your spine is properly supported."
                ])
            else:
                return "Overall posture is good, but stay mindful of minor adjustments."
        
        # Prioritize the most critical issues
        critical_issues = [issue for issue in issues if "forward" in issue.lower() or "rounded" in issue.lower()]
        
        if critical_issues:
            issue = critical_issues[0]
        else:
            issue = issues[0]
        
        if issue in self.feedback_messages and "corrections" in self.feedback_messages[issue]:
            return random.choice(self.feedback_messages[issue]["corrections"])
        
        return "Please adjust your posture and sit up straight."