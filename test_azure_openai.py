#!/usr/bin/env python3
"""
Test script for Azure OpenAI connection
This script helps diagnose Azure OpenAI connection issues and test different configurations
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_imports():
    """Test if required packages are available"""
    print("🔍 Testing imports...")
    try:
        from openai import OpenAI
        print("✅ OpenAI package imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import OpenAI package: {e}")
        print("💡 Install with: pip install openai")
        return False

def check_environment_variables():
    """Check if all required environment variables are set"""
    print("\n🔍 Checking environment variables...")
    
    required_vars = {
        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'AZURE_OPENAI_KEY': os.getenv('AZURE_OPENAI_KEY'),
        'AZURE_OPENAI_MODEL': os.getenv('AZURE_OPENAI_MODEL')
    }
    
    all_set = True
    for var_name, var_value in required_vars.items():
        if var_value:
            print(f"✅ {var_name}: {'*' * 20}...{var_value[-4:] if len(var_value) > 4 else '****'}")
        else:
            print(f"❌ {var_name}: Not set")
            all_set = False
    
    return all_set, required_vars

def test_basic_connection():
    """Test basic Azure OpenAI connection"""
    print("\n🔍 Testing basic Azure OpenAI connection...")
    
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    key = os.getenv('AZURE_OPENAI_KEY')
    model = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4')
    
    if not endpoint or not key:
        print("❌ Missing required environment variables")
        return False, None
    
    from openai import OpenAI
    
    try:
        print(f"\n📝 Testing Configuration: Standard Azure OpenAI format")
        
        # Extract resource name from endpoint
        if 'openai.azure.com' in endpoint:
            resource_name = endpoint.split('.')[0].replace('https://', '')
            base_url = f"https://{resource_name}.openai.azure.com/openai/v1/"
        else:
            base_url = endpoint
        
        print(f"   Base URL: {base_url}")
        print(f"   Model/Deployment: {model}")
        
        client = OpenAI(
            api_key=key,
            base_url=base_url
        )
        
        print("   Testing simple connection...")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        if response.choices and response.choices[0].message:
            print(f"✅ Basic connection SUCCESS: {response.choices[0].message.content}")
            return True, client
        else:
            print("❌ Basic connection FAILED: No response content")
            return False, None
            
    except Exception as e:
        print(f"❌ Basic connection FAILED: {e}")
        
        # Try alternative deployment names if main one fails
        common_deployment_names = ['gpt-4', 'gpt-4o', 'gpt-35-turbo', 'gpt-4-turbo']
        
        for test_model in common_deployment_names:
            if test_model == model:
                continue  # Already tested above
                
            try:
                print(f"\n📝 Trying alternative deployment name: '{test_model}'")
                
                response = client.chat.completions.create(
                    model=test_model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                
                if response.choices and response.choices[0].message:
                    print(f"✅ SUCCESS with '{test_model}': {response.choices[0].message.content}")
                    print(f"💡 Update your .env file: AZURE_OPENAI_MODEL={test_model}")
                    return True, client
                else:
                    print(f"❌ '{test_model}' FAILED: No response content")
                    
            except Exception as e2:
                print(f"❌ '{test_model}' FAILED: {e2}")
        
        return False, None

def test_posture_feedback_generation(client):
    """Test the actual posture feedback generation like the main app"""
    print("\n🔍 Testing posture feedback message generation...")
    
    if not client:
        print("❌ No client available for testing")
        return False
    
    model = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4')
    
    # Test with a realistic posture context like the main app uses
    posture_context = """The user has been experiencing the following posture issues:
- Leaning forward: detected 154 times in the last 10 seconds. Average head tilt: 38.0°, Average spine angle: 0.0°, Detection confidence: 1.00
- Head too far forward: detected 154 times in the last 10 seconds. Average head tilt: 38.0°, Average spine angle: 0.0°, Detection confidence: 1.00"""
    
    posture_prompt = f"""The user has been maintaining poor posture for about 10 seconds. Here's what I detected:

{posture_context}

Please provide a brief, friendly, and actionable suggestion to help them improve their posture. Focus on the most significant issues mentioned and give specific steps they can take right now. Keep it conversational and encouraging - imagine you're a helpful colleague gently reminding them to check their posture."""
    
    try:
        print("📝 Testing posture feedback generation...")
        print(f"   Model: {model}")
        print(f"   Context length: {len(posture_context)} characters")
        
        response = client.chat.completions.create(
            model=model,
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
                    "content": posture_prompt
                }
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        if response.choices and response.choices[0].message:
            feedback = response.choices[0].message.content.strip()
            print(f"✅ Posture feedback SUCCESS:")
            print(f"   Generated feedback: '{feedback}'")
            print(f"   Response length: {len(feedback)} characters")
            print(f"   Token usage: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
            return True
        else:
            print("❌ Posture feedback FAILED: No response content")
            return False
            
    except Exception as e:
        print(f"❌ Posture feedback FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")
        if hasattr(e, 'response'):
            print(f"   HTTP Status: {e.response.status_code if hasattr(e.response, 'status_code') else 'N/A'}")
        return False

def test_various_message_types(client):
    """Test different types of messages to isolate the issue"""
    print("\n🔍 Testing various message types...")
    
    if not client:
        print("❌ No client available for testing")
        return False
    
    model = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4')
    
    test_cases = [
        {
            "name": "Simple message",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10
        },
        {
            "name": "System + User message",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"}
            ],
            "max_tokens": 10
        },
        {
            "name": "Longer message",
            "messages": [{"role": "user", "content": "Please provide a brief suggestion for improving posture."}],
            "max_tokens": 50
        },
        {
            "name": "Complex posture prompt",
            "messages": [
                {"role": "system", "content": "You are a posture coach."},
                {"role": "user", "content": "The user has been leaning forward. Give a brief suggestion."}
            ],
            "max_tokens": 100
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"\n📝 Test {i}: {test_case['name']}")
            
            response = client.chat.completions.create(
                model=model,
                messages=test_case['messages'],
                max_tokens=test_case['max_tokens'],
                temperature=0.7
            )
            
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content.strip()
                print(f"   ✅ SUCCESS: '{content[:50]}{'...' if len(content) > 50 else ''}'")
                success_count += 1
            else:
                print(f"   ❌ FAILED: No response content")
                
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
    
    print(f"\n📊 Test Results: {success_count}/{len(test_cases)} passed")
    return success_count == len(test_cases)

def get_deployment_suggestions():
    """Provide suggestions for finding the correct deployment name"""
    print("\n💡 How to find your correct deployment name:")
    print("1. Go to https://oai.azure.com/")
    print("2. Navigate to 'Deployments' in the left sidebar")
    print("3. Look at the 'Deployment name' column (NOT 'Model name')")
    print("4. Copy the exact deployment name and update your .env file")
    print("\nCommon deployment names to try:")
    print("- gpt-4")
    print("- gpt-4o") 
    print("- gpt-35-turbo")
    print("- gpt-4-turbo")
    print("- my-gpt-4 (or whatever custom name you used)")

def main():
    """Main test function"""
    print("🚀 Azure OpenAI Connection Test")
    print("=" * 50)
    
    # Test 1: Check imports
    if not test_imports():
        sys.exit(1)
    
    # Test 2: Check environment variables
    env_ok, env_vars = check_environment_variables()
    if not env_ok:
        print("\n❌ Environment variables missing. Please check your .env file.")
        sys.exit(1)
    
    # Test 3: Test basic connection
    connection_success, client = test_basic_connection()
    
    if not connection_success:
        print("\n❌ Basic connection failed!")
        get_deployment_suggestions()
        
        print("\n🔧 Quick fixes to try:")
        print("1. Check your deployment name in Azure OpenAI Studio")
        print("2. Verify your API key is correct and not expired")
        print("3. Ensure your Azure OpenAI resource is in a supported region")
        print("4. Try different deployment names listed above")
        
        return False
    
    print("\n✅ Basic Azure OpenAI connection successful!")
    
    # Test 4: Test posture feedback generation (the actual use case)
    feedback_success = test_posture_feedback_generation(client)
    
    if not feedback_success:
        print("\n❌ Posture feedback generation failed!")
        print("🔧 This means the connection works but message generation has issues.")
        
        # Test 5: Test various message types to isolate the issue
        print("\n🔍 Running detailed message testing to isolate the issue...")
        test_various_message_types(client)
        
        return False
    else:
        print("\n✅ Posture feedback generation successful!")
        
        # Test 6: Run additional message tests for completeness
        test_various_message_types(client)
        
        print("\n🎉 All tests passed! Your Azure OpenAI setup is working correctly.")
        return True

if __name__ == "__main__":
    main()