import azure.cognitiveservices.speech as speechsdk
import threading
import queue
import time
import os
from typing import Optional, List
import logging


class AzureSpeechService:
    def __init__(self, speech_key: str, speech_region: str):
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.speech_config = None
        self.synthesizer = None
        self.is_speaking = False
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.stop_worker = False
        
        self._initialize_speech_service()
        self._start_worker_thread()
    
    def _initialize_speech_service(self):
        """Initialize Azure Speech service"""
        if not self.speech_key or not self.speech_region:
            raise ValueError("Azure Speech key and region must be provided")
        
        try:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key, 
                region=self.speech_region
            )
            
            # Configure speech synthesis
            self.speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
            self.speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
            )
            
            # Create synthesizer
            self.synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
            
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
        if not self.synthesizer:
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
            
            result = self.synthesizer.speak_ssml(ssml)
            
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
        
        if priority:
            # Clear queue and add high priority message
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break
        
        # Add to queue
        self.speech_queue.put(text)
        logging.debug(f"Added to speech queue: {text}")
    
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
        if self.synthesizer:
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
        
        if self.synthesizer:
            # Note: Azure Speech SDK doesn't have explicit cleanup method
            self.synthesizer = None
        
        logging.info("Azure Speech service cleaned up")


class PostureSpeechManager:
    def __init__(self, speech_service: AzureSpeechService):
        self.speech_service = speech_service
        self.last_spoken_time = 0
        self.min_speech_interval = 10  # Minimum seconds between speech
        self.last_spoken_message = ""
        
    def speak_posture_feedback(self, message: str, priority: int = 1, force: bool = False):
        """Speak posture feedback with intelligent timing"""
        current_time = time.time()
        
        # Avoid repeating the same message too frequently
        if message == self.last_spoken_message and (current_time - self.last_spoken_time) < 30:
            return
        
        # Check if enough time has passed or if it's high priority
        should_speak = (
            force or 
            priority >= 4 or 
            (current_time - self.last_spoken_time) >= self.min_speech_interval
        )
        
        if should_speak:
            if priority >= 4:
                self.speech_service.speak_immediate(message)
            else:
                self.speech_service.speak_async(message, priority=priority >= 3)
            
            self.last_spoken_time = current_time
            self.last_spoken_message = message
            logging.info(f"Speaking posture feedback: {message}")
    
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


def create_speech_service(speech_key: str = None, speech_region: str = None) -> Optional[AzureSpeechService]:
    """Factory function to create speech service with environment variables fallback"""
    
    # Use provided parameters or fall back to environment variables
    if not speech_key:
        speech_key = os.getenv('AZURE_SPEECH_KEY')
    if not speech_region:
        speech_region = os.getenv('AZURE_SPEECH_REGION')
    
    if not speech_key or not speech_region:
        logging.warning("Azure Speech credentials not provided. Speech functionality will be disabled.")
        return None
    
    try:
        return AzureSpeechService(speech_key, speech_region)
    except Exception as e:
        logging.error(f"Failed to create Azure Speech service: {e}")
        return None