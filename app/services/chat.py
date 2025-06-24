import os
import uuid
from typing import Tuple

try:
    from air import DistillerClient
    AI_REFINERY_AVAILABLE = True
except ImportError:
    AI_REFINERY_AVAILABLE = False
    print("⚠️ AI Refinery SDK not available. ChatService will run in mock mode.")

class ChatService:
    def __init__(self, distiller_client: any, project_name: str):
        self.distiller_client = distiller_client
        self.project_name = project_name

    async def handle_chat(self, query: str, session_id: str = None) -> Tuple[str, str]:
        if not session_id:
            session_id = str(uuid.uuid4())

        if not self.distiller_client or not AI_REFINERY_AVAILABLE:
            print(f"ChatService for project '{self.project_name}' is in mock mode.")
            return f"Mock response for {self.project_name}: {query}", session_id

        try:
            async with self.distiller_client(project=self.project_name, uuid=session_id) as dc:
                responses = await dc.query(query=query.strip())
                ai_response = ""
                async for response in responses:
                    if 'content' in response:
                        ai_response = response['content']
                        break
                
                if not ai_response:
                    return "Sorry, I couldn't get a response from the chat service.", session_id

                return ai_response, session_id
        except Exception as e:
            print(f"Error during AI Refinery chat query for project '{self.project_name}': {e}")
            return "An error occurred while processing your request.", session_id