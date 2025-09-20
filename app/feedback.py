import random
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from posture import PostureMetrics


@dataclass
class FeedbackMessage:
    message: str
    priority: int
    category: str = "correction"  # "correction", "tip", "warning", "llm_generated"


@dataclass
class IssueMetrics:
    """Track metrics for a specific posture issue"""
    issue_name: str
    head_angle: float
    spine_angle: float
    confidence: float
    timestamp: float
    
    
class PostureFeedback:
    def __init__(self):
        self.last_feedback_time = 0
        self.feedback_history = []
        self.consecutive_bad_posture_count = 0
        self.last_posture_good = True
        
        # Enhanced tracking for LLM feedback
        self.bad_posture_start_time = None
        self.issue_history = defaultdict(deque)  # Track recent instances of each issue
        self.prevalent_issues = []  # Track most common issues during bad posture periods
        self.current_metrics_buffer = deque(maxlen=60)  # Last 60 measurements for context
        
        # Feedback messages organized by issue type (fallback for when LLM is unavailable)
        self.fallback_feedback_messages = {
            "Head too far forward": {
                "corrections": [
                    "Pull your chin back and imagine a string pulling the top of your head up.",
                    "Tuck your chin slightly and align your ears over your shoulders."
                ],
                "tips": [
                    "Place a sticky note on your monitor as a reminder to check your head position.",
                    "Set your screen at eye level to avoid craning your neck forward.",
                    "Take regular breaks to do neck stretches and resets."
                ]
            },
            "Head too far backward": {
                "corrections": [
                    "Bring your head slightly forward to align ears over shoulders.",
                    "Relax your neck and find a neutral head position."
                ]
            },
            "Head tilted forward": {
                "corrections": [
                    "Lift your chin up and look straight ahead at your monitor.",
                    "Raise your monitor height so you don't need to look down.",
                    "Sit up tall and elongate your neck.",
                    "Adjust your chair height to bring your eye level up."
                ],
                "tips": [
                    "Place books or a monitor stand under your screen to raise it.",
                    "Check that your screen top is at or slightly below eye level."
                ]
            },
            "Head tilted back": {
                "corrections": [
                    "Gently lower your chin and bring your head forward slightly.",
                    "Relax your neck and let your head find a neutral position.",
                    "Check if your screen is too high - this might be causing you to tilt back."
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
                    "Use a lumbar support pillow if your chair doesn't provide enough support."
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
    
    def update_metrics_buffer(self, posture_metrics: PostureMetrics):
        """Update the metrics buffer for LLM context"""
        current_time = time.time()
        
        # Add current metrics to buffer
        self.current_metrics_buffer.append({
            'timestamp': current_time,
            'head_angle': getattr(posture_metrics, 'head_tilt_angle', 0),
            'spine_angle': getattr(posture_metrics, 'back_angle', 0),
            'confidence': posture_metrics.confidence,
            'issues': posture_metrics.issues.copy() if posture_metrics.issues else []
        })
        
        # Track issue occurrences
        if posture_metrics.issues:
            for issue in posture_metrics.issues:
                issue_metric = IssueMetrics(
                    issue_name=issue,
                    head_angle=getattr(posture_metrics, 'head_tilt_angle', 0),
                    spine_angle=getattr(posture_metrics, 'back_angle', 0),
                    confidence=posture_metrics.confidence,
                    timestamp=current_time
                )
                self.issue_history[issue].append(issue_metric)
                
                # Keep only recent history (last 10 minutes)
                while (self.issue_history[issue] and 
                       current_time - self.issue_history[issue][0].timestamp > 600):
                    self.issue_history[issue].popleft()
    
    def get_prevalent_issues(self, duration_minutes: float = 0.5) -> List[Tuple[str, List[IssueMetrics]]]:
        """Get the most prevalent issues over the specified duration"""
        current_time = time.time()
        cutoff_time = current_time - (duration_minutes * 60)
        
        logging.debug(f"Looking for issues in last {duration_minutes} minutes (cutoff: {cutoff_time})")
        
        # Count recent occurrences of each issue
        issue_counts = {}
        issue_metrics = {}
        
        for issue, history in self.issue_history.items():
            recent_metrics = [m for m in history if m.timestamp >= cutoff_time]
            if recent_metrics:
                issue_counts[issue] = len(recent_metrics)
                issue_metrics[issue] = recent_metrics
                logging.debug(f"Issue '{issue}': {len(recent_metrics)} recent occurrences")
        
        # Sort by frequency and return top 2
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        logging.debug(f"Sorted issues by frequency: {sorted_issues}")
        
        result = []
        for issue, count in sorted_issues[:2]:
            result.append((issue, issue_metrics[issue]))
        
        return result
    
    def generate_llm_context(self, prevalent_issues: List[Tuple[str, List[IssueMetrics]]]) -> str:
        """Generate context string for LLM"""
        if not prevalent_issues:
            logging.warning("No prevalent issues to generate LLM context")
            return "No specific posture issues detected in recent history."
        
        context_parts = []
        context_parts.append("The user has been experiencing the following posture issues:")
        
        for issue_name, metrics_list in prevalent_issues:
            if not metrics_list:
                continue
                
            # Calculate averages
            avg_head_angle = sum(m.head_angle for m in metrics_list) / len(metrics_list)
            avg_spine_angle = sum(m.spine_angle for m in metrics_list) / len(metrics_list)
            avg_confidence = sum(m.confidence for m in metrics_list) / len(metrics_list)
            
            context_parts.append(
                f"- {issue_name}: detected {len(metrics_list)} times in the last 10 seconds. "
                f"Average head tilt: {avg_head_angle:.1f}°, "
                f"Average spine angle: {avg_spine_angle:.1f}°, "
                f"Detection confidence: {avg_confidence:.2f}"
            )
        
        context = "\n".join(context_parts)
        logging.info(f"Generated LLM context with {len(prevalent_issues)} issues: {context}")
        return context
    
    def generate_feedback(self, posture_metrics: PostureMetrics, posture_score: float) -> List[FeedbackMessage]:
        """Generate appropriate feedback based on current posture"""
        current_time = time.time()
        feedback_messages = []
        
        # Update metrics tracking
        self.update_metrics_buffer(posture_metrics)
        
        # Track consecutive bad posture
        if posture_score < 7:
            if self.last_posture_good:
                self.consecutive_bad_posture_count = 1
                self.bad_posture_start_time = current_time
            else:
                self.consecutive_bad_posture_count += 1
            self.last_posture_good = False
        else:
            if not self.last_posture_good and posture_score > 8:
                # Improved posture - give encouragement
                feedback_messages.append(FeedbackMessage(
                    message=random.choice(self.encouragement_messages),
                    priority=2,
                    category="encouragement"
                ))
            self.consecutive_bad_posture_count = 0
            self.last_posture_good = True
            self.bad_posture_start_time = None
        
        # Check if we've had bad posture for 10 seconds - this triggers LLM feedback
        bad_posture_duration = 0
        if self.bad_posture_start_time:
            bad_posture_duration = current_time - self.bad_posture_start_time
        
        logging.info(f"Bad posture duration: {bad_posture_duration:.1f}s, Score: {posture_score:.1f}, Issues: {posture_metrics.issues}")

        if bad_posture_duration >= 10:  # 10 seconds for testing
            # Generate LLM-based feedback for sustained poor posture
            prevalent_issues = self.get_prevalent_issues(0.2)  # Last 12 seconds
            
            logging.info(f"Sustained bad posture detected! Duration: {bad_posture_duration:.1f}s")
            logging.info(f"Prevalent issues found: {len(prevalent_issues)}")
            for issue_name, metrics_list in prevalent_issues:
                logging.info(f"  - {issue_name}: {len(metrics_list)} occurrences")
            
            if prevalent_issues:
                # This will be handled by the speech manager with LLM integration
                context = self.generate_llm_context(prevalent_issues)
                logging.info(f"Generated LLM context: {context}")
                feedback_messages.append(FeedbackMessage(
                    message=f"LLM_CONTEXT:{context}",
                    priority=5,
                    category="llm_generated"
                ))
                
                # Reset the timer to prevent continuous LLM calls
                self.bad_posture_start_time = current_time
                logging.info("LLM feedback message created and timer reset")
            else:
                logging.warning("No prevalent issues found despite sustained bad posture")        # Fallback to traditional feedback for immediate issues
        elif posture_metrics.issues:
            for issue in posture_metrics.issues[:1]:  # Only one immediate issue
                if issue in self.fallback_feedback_messages:
                    issue_feedback = self.fallback_feedback_messages[issue]
                    
                    # Choose correction message
                    if "corrections" in issue_feedback:
                        correction_msg = random.choice(issue_feedback["corrections"])
                        priority = 3 if bad_posture_duration > 60 else 2
                        
                        feedback_messages.append(FeedbackMessage(
                            message=correction_msg,
                            priority=priority,
                            category="correction"
                        ))
        
        # General tips occasionally when posture is good
        elif posture_score > 8 and random.random() < 0.1:
            tip_msg = random.choice(self.general_tips)
            feedback_messages.append(FeedbackMessage(
                message=tip_msg,
                priority=1,
                category="tip"
            ))
        
        # Sort by priority (highest first)
        feedback_messages.sort(key=lambda x: x.priority, reverse=True)
        
        return feedback_messages[:2]  # Return max 2 messages to avoid overwhelming
    
    def should_give_feedback(self, min_interval: float = 180.0) -> bool:
        """Check if enough time has passed to give new feedback (now 3 minutes)"""
        current_time = time.time()
        return (current_time - self.last_feedback_time) >= min_interval
    
    def mark_feedback_given(self):
        """Mark that feedback was just given"""
        self.last_feedback_time = time.time()
    
    def get_posture_status_message(self, posture_score: float) -> str:
        """Get a general status message based on posture score"""
        if posture_score >= 9:
            return "Excellent posture!"
        elif posture_score >= 8:
            return "Good posture!"
        elif posture_score >= 7:
            return "Fair posture - minor adjustments needed."
        elif posture_score >= 6:
            return "Poor posture - needs attention."
        else:
            return "Very poor posture - immediate correction recommended."
    
    def get_smart_suggestion(self, issues: List[str], posture_score: float) -> str:
        """Generate a smart, contextual suggestion"""
        if not issues:
            if posture_score > 8.5:
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
        
        if issue in self.fallback_feedback_messages and "corrections" in self.fallback_feedback_messages[issue]:
            return random.choice(self.fallback_feedback_messages[issue]["corrections"])
        
        return "Please adjust your posture and sit up straight."