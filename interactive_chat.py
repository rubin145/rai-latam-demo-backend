#!/usr/bin/env python3
"""
Interactive chat script to test the API endpoint.
"""

import requests
import json

# API configuration
BASE_URL = "http://localhost:8000"
CHATBOT_ID = "banking"  # Change if needed
USE_GUARDRAILS = True   # Change to False to test without guardrails

def chat_interactive():
    """Interactive chat loop with the API."""
    
    print(f"🚀 Interactive Chat with {CHATBOT_ID} chatbot")
    print(f"🛡️  Guardrails: {'ON' if USE_GUARDRAILS else 'OFF'}")
    print("💡 Type 'quit' or 'exit' to stop\n")
    
    session_id = None
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
            
            # Prepare request
            url = f"{BASE_URL}/api/chatbots/{CHATBOT_ID}/chat"
            params = {"use_guardrails": USE_GUARDRAILS}
            
            payload = {
                "query": user_input,
                "session_id": session_id  # Will be None for first request
            }
            
            # Make API call
            print("🤖 Thinking...")
            response = requests.post(url, json=payload, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract response
                bot_response = data.get("response", "No response")
                session_id = data.get("session_id")  # Keep session for next requests
                filter_decision = data.get("filter_decision")
                
                print(f"Bot: {bot_response}")
                
                # Show filter info if available
                if filter_decision:
                    status = "🟢 SAFE" if filter_decision == "safe" else "🔴 BLOCKED"
                    print(f"     {status}")
                
                print()  # Empty line for readability
                
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                print()
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except requests.exceptions.ConnectionError:
            print("❌ Connection error. Is the server running on http://localhost:8000?")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    chat_interactive()