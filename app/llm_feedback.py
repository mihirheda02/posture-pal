import os
import logging
from typing import Optional

# Try to import openai, but make it optional
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not installed. LLM feedback will be disabled.")

import time

class LLMFeedbackGenerator:
    """Generate intelligent posture feedback using Azure OpenAI"""
    
    def __init__(self):
        self.client = None
        self.model_name = "gpt-4"  # Default model name
        self.last_request_time = 0
        self.rate_limit_interval = 30  # Minimum 30 seconds between requests
        self.logger = logging.getLogger(__name__)
        
        # Initialize Azure OpenAI client
        self._initialize_client()
        
    def _initialize_client(self) -> bool:
        """Initialize Azure OpenAI client"""
        if not OPENAI_AVAILABLE:
            self.logger.warning("OpenAI package not available. Install with: pip install openai")
            return False
            
        try:
            # Get credentials from environment
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            azure_key = os.getenv('AZURE_OPENAI_KEY')
            model_name = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4')
            
            self.logger.info(f"Checking Azure OpenAI credentials...")
            self.logger.info(f"Endpoint: {'✅ Set' if azure_endpoint else '❌ Missing'}")
            self.logger.info(f"API Key: {'✅ Set' if azure_key else '❌ Missing'}")
            self.logger.info(f"Model: {model_name}")
            
            if not azure_endpoint or not azure_key:
                self.logger.warning("Azure OpenAI credentials not found. LLM feedback will be disabled.")
                return False
            
            # For Azure OpenAI, use the correct base URL format
            if 'openai.azure.com' in azure_endpoint:
                # Extract resource name from the endpoint
                # Format: https://posture-pal-azure-openai.openai.azure.com/
                resource_name = azure_endpoint.split('.')[0].replace('https://', '')
                base_url = f"https://{resource_name}.openai.azure.com/openai/v1/"
                
                self.logger.info(f"Using Azure OpenAI base URL: {base_url}")
                
                self.client = OpenAI(
                    api_key=azure_key,
                    base_url=base_url
                )
            else:
                # Regular OpenAI format (fallback)
                self.client = OpenAI(
                    api_key=azure_key,
                    base_url=azure_endpoint
                )
            
            self.model_name = model_name
            self.logger.info("✅ Azure OpenAI client initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            return False
    
    def _is_rate_limited(self) -> bool:
        """Check if we're still within rate limit period"""
        current_time = time.time()
        return (current_time - self.last_request_time) < self.rate_limit_interval
    
    def generate_posture_feedback(self, context: str) -> Optional[str]:
        """Generate intelligent posture feedback based on context"""
        self.logger.info(f"🤖 LLM feedback request received. Context: {context[:100]}...")
        
        if not self.client:
            self.logger.warning("❌ Azure OpenAI client not available")
            return None
            
        if self._is_rate_limited():
            self.logger.debug("⏱️ Rate limited - skipping LLM request")
            return None
        
        try:
            prompt = self._create_posture_prompt(context)
            self.logger.info(f"📝 Sending prompt to Azure OpenAI: {prompt[:150]}...")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a friendly, knowledgeable posture coach assistant. Your job is to help users improve their posture by providing concise, actionable, and encouraging feedback. 

                                    Guidelines:
                                    - Keep responses to 1-2 sentences maximum
                                    - Be warm and encouraging, not clinical or robotic
                                    - Focus on immediate, actionable steps
                                    - Use conversational, friendly language
                                    - Avoid technical jargon
                                    - Start with acknowledgment, then provide specific guidance
                                    - End with motivation when appropriate

                                    Example good responses:
                                    "I notice you've been leaning forward for a while - let's reset by sitting back in your chair and rolling your shoulders back."
                                    "Your head has been tilted down quite a bit - try raising your monitor or adjusting your chair height so you can look straight ahead."
                                    "Time for a posture check! Pull your chin back slightly and imagine a string pulling the top of your head toward the ceiling."
                                    """
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            self.last_request_time = time.time()
            
            if response.choices and response.choices[0].message:
                feedback = response.choices[0].message.content.strip()
                self.logger.info(f"✅ LLM feedback generated successfully: {feedback}")
                return feedback
            else:
                self.logger.warning("❌ No response content from Azure OpenAI")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error generating LLM feedback: {e}")
            return None
    
    def _create_posture_prompt(self, context: str) -> str:
        """Create a prompt for the LLM based on posture context"""
        base_prompt = f"""The user has been maintaining poor posture for about 10 seconds. Here's what I detected:

{context}

Please provide a brief, friendly, and actionable suggestion to help them improve their posture. Focus on the most significant issues mentioned and give specific steps they can take right now. Keep it conversational and encouraging - imagine you're a helpful colleague gently reminding them to check their posture."""
        
        return base_prompt
    
    def test_connection(self) -> bool:
        """Test the Azure OpenAI connection"""
        if not self.client:
            return False
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return bool(response.choices and response.choices[0].message)
        except Exception as e:
            self.logger.error(f"Azure OpenAI connection test failed: {e}")
            return False


def create_llm_feedback_generator() -> Optional[LLMFeedbackGenerator]:
    """Factory function to create LLM feedback generator"""
    try:
        generator = LLMFeedbackGenerator()
        if generator.client:
            return generator
        else:
            logging.warning("LLM feedback generator not available - missing credentials")
            return None
    except Exception as e:
        logging.error(f"Failed to create LLM feedback generator: {e}")
        return None