import azure.cognitiveservices.speech as speechsdk
import threading
import queue
import time
import os
from typing import Optional, List
import logging


class AzureSpeechService:
    def __init__(self, speech_key: str, speech_endpoint: str):
        self.speech_key = speech_key
        self.speech_endpoint = speech_endpoint
        self.speech_config = None
        self.audio_config = None
        self.speech_synthesizer = None
        self.is_speaking = False
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.stop_worker = False
        
        self._initialize_speech_service()
        self._start_worker_thread()
    
    def _initialize_speech_service(self):
        """Initialize Azure Speech service"""
        if not self.speech_key or not self.speech_endpoint:
            raise ValueError("Azure Speech key and endpoint must be provided")

        try:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                endpoint=self.speech_endpoint
            )
            self.audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
            
            # Configure speech synthesis
            self.speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
            
            # Create speech synthesizer
            self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=self.audio_config)
            
            logging.info("Azure Speech service initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Azure Speech service: {e}")
            raise
    
    def _start_worker_thread(self):
        """Start background thread for speech synthesis"""
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()
    
    def _speech_worker(self):
        """Background worker thread for processing speech requests"""
        while not self.stop_worker:
            try:
                # Get speech request from queue (timeout after 1 second)
                speech_text = self.speech_queue.get(timeout=1)
                
                if speech_text is None:  # Shutdown signal
                    break
                
                self._synthesize_speech_sync(speech_text)
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in speech worker: {e}")
    
    def _synthesize_speech_sync(self, text: str):
        """Synchronously synthesize speech"""
        if not self.speech_synthesizer:
            logging.warning("Speech synthesizer not initialized")
            return
        
        try:
            self.is_speaking = True
            
            # Create SSML for better speech control
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="en-US-AriaNeural">
                    <prosody rate="0.9" pitch="medium">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
            result = self.speech_synthesizer.speak_ssml(ssml)
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logging.debug(f"Speech synthesis completed: {text}")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logging.error(f"Speech synthesis canceled: {cancellation_details.reason}")
                if cancellation_details.error_details:
                    logging.error(f"Error details: {cancellation_details.error_details}")
            
        except Exception as e:
            logging.error(f"Error during speech synthesis: {e}")
        finally:
            self.is_speaking = False
    
    def speak_async(self, text: str, priority: bool = False):
        """Add text to speech queue for asynchronous synthesis"""
        if not text.strip():
            return
        
        # Prevent queue buildup - limit queue size to prevent spam
        current_queue_size = self.speech_queue.qsize()
        max_queue_size = 3 if priority else 1
        
        if current_queue_size >= max_queue_size:
            if priority:
                # For priority messages, clear some old messages
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    pass
            else:
                # For normal messages, skip if queue is full
                logging.debug(f"Speech queue full, skipping: {text}")
                return
        
        # Add to queue
        self.speech_queue.put(text)
        logging.debug(f"Added to speech queue: {text} (queue size: {current_queue_size + 1})")
    
    def speak_immediate(self, text: str):
        """Speak text immediately, interrupting current speech if necessary"""
        if not text.strip():
            return
        
        # Stop current speech
        self.stop_current_speech()
        
        # Speak immediately
        threading.Thread(target=self._synthesize_speech_sync, args=(text,), daemon=True).start()
    
    def stop_current_speech(self):
        """Stop current speech synthesis"""
        if self.speech_synthesizer:
            # Clear the queue
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break
    
    def is_currently_speaking(self) -> bool:
        """Check if currently speaking"""
        return self.is_speaking
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_worker = True
        
        # Send shutdown signal
        self.speech_queue.put(None)
        
        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)
        
        if self.speech_synthesizer:
            # Note: Azure Speech SDK doesn't have explicit cleanup method
            self.speech_synthesizer = None
        
        logging.info("Azure Speech service cleaned up")


class PostureSpeechManager:
    def __init__(self, speech_service: AzureSpeechService, llm_feedback_generator=None):
        self.speech_service = speech_service
        self.llm_feedback_generator = llm_feedback_generator
        self.last_spoken_time = 0
        self.min_speech_interval = 60  # Increased to 1 minute minimum between speech
        self.last_spoken_message = ""
        self.message_cooldowns = {}  # Track cooldowns for specific message types
        
    def speak_posture_feedback(self, message: str, priority: int = 1, force: bool = False):
        """Speak posture feedback with intelligent timing and LLM integration"""
        current_time = time.time()
        
        # Check if this is an LLM context request
        if message.startswith("LLM_CONTEXT:"):
            context = message[12:]  # Remove "LLM_CONTEXT:" prefix
            self._handle_llm_feedback(context, priority, force)
            return
        
        # Avoid repeating the same message too frequently (5 minute cooldown)
        if message == self.last_spoken_message and (current_time - self.last_spoken_time) < 300:
            return
        
        # Check message type cooldown
        message_key = message[:20]  # Use first 20 chars as key
        if message_key in self.message_cooldowns:
            if (current_time - self.message_cooldowns[message_key]) < 120:  # 2 minute cooldown per message type
                return
        
        # Check if enough time has passed or if it's high priority
        should_speak = (
            force or 
            priority >= 5 or  # Only very high priority messages bypass timing
            (current_time - self.last_spoken_time) >= self.min_speech_interval
        )
        
        if should_speak:
            if priority >= 5:
                self.speech_service.speak_immediate(message)
            else:
                self.speech_service.speak_async(message, priority=priority >= 4)
            
            self.last_spoken_time = current_time
            self.last_spoken_message = message
            self.message_cooldowns[message_key] = current_time
            logging.info(f"Speaking posture feedback (priority {priority}): {message}")
    
    def _handle_llm_feedback(self, context: str, priority: int, force: bool):
        """Handle LLM-generated feedback"""
        if not self.llm_feedback_generator:
            # Fallback to generic message if LLM is not available
            fallback_message = "You've been in poor posture for a few minutes. Let's take a moment to reset - sit back in your chair, roll your shoulders back, and align your head over your shoulders."
            self.speak_posture_feedback(fallback_message, priority=5, force=True)
            return
        
        try:
            # Generate LLM feedback
            llm_response = self.llm_feedback_generator.generate_posture_feedback(context)
            
            if llm_response:
                # Speak the LLM-generated feedback with high priority
                self.speak_posture_feedback(llm_response, priority=5, force=True)
                logging.info(f"LLM feedback delivered: {llm_response}")
            else:
                # Fallback if LLM fails
                fallback_message = "Time for a posture check! Let's reset your position and get back to good alignment."
                self.speak_posture_feedback(fallback_message, priority=5, force=True)
                
        except Exception as e:
            logging.error(f"Error generating LLM feedback: {e}")
            # Fallback message
            fallback_message = "Your posture needs attention. Please take a moment to adjust your position."
            self.speak_posture_feedback(fallback_message, priority=5, force=True)
        
    def speak_encouragement(self, message: str):
        """Speak encouragement message"""
        self.speech_service.speak_async(message, priority=False)
        logging.info(f"Speaking encouragement: {message}")
    
    def speak_status(self, message: str):
        """Speak status update"""
        current_time = time.time()
        if (current_time - self.last_spoken_time) >= 5:  # Shorter interval for status
            self.speech_service.speak_async(message, priority=False)
            self.last_spoken_time = current_time
    
    def emergency_stop(self):
        """Stop all speech immediately"""
        self.speech_service.stop_current_speech()
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self.speech_service.is_currently_speaking()


def create_speech_service(speech_key: str = None, speech_endpoint: str = None) -> Optional[AzureSpeechService]:
    """Factory function to create speech service with environment variables fallback"""
    
    # Use provided parameters or fall back to environment variables
    if not speech_key:
        speech_key = os.getenv('AZURE_SPEECH_KEY')
    if not speech_endpoint:
        speech_endpoint = os.getenv('AZURE_SPEECH_ENDPOINT')
    
    if not speech_key or not speech_endpoint:
        logging.warning("Azure Speech credentials not provided. Speech functionality will be disabled.")
        return None
    
    try:
        return AzureSpeechService(speech_key, speech_endpoint)
    except Exception as e:
        logging.error(f"Failed to create Azure Speech service: {e}")
        return None