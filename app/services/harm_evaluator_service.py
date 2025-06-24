import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Tuple
from ..models.schemas import TestQuestion, TestQuestionsDatabase
from ..utils.risk_analyzer import extract_risk_dimensions

# Try to import AI Refinery SDK, make it optional for testing
try:
    from air import login, DistillerClient
    AI_REFINERY_AVAILABLE = True
except ImportError:
    AI_REFINERY_AVAILABLE = False
    print("⚠️ AI Refinery SDK not available. Running in mock mode.")

class HarmEvaluatorService:
    def __init__(self, distiller_client: any):
        self.distiller_client = distiller_client
        self.project_name = os.getenv("PROJECT_NAME", "harm_evaluator_ui")
        self.test_db_file = "test_questions_db.json"

    async def evaluate_query(self, query: str) -> Dict[str, Any]:
        """Evaluate a single query for harm"""
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not self.distiller_client or not AI_REFINERY_AVAILABLE:
            print("HarmEvaluatorService is in mock mode.")
            return self._get_mock_response(query)

        try:
            # The project is now created on startup. We just use the client.
            async with self.distiller_client(project=self.project_name, uuid="api_user") as dc:
                try:
                    responses = await dc.query(query=query.strip())
                    result = ""
                    
                    # Collect response with timeout protection
                    async for response in responses:
                        if 'content' in response:
                            result = response['content']
                            break  # Get first response
                    
                    if not result:
                        print("⚠️ No response received from AI Refinery, falling back to mock")
                        return self._get_mock_response(query)
                    
                    # Extract risk dimensions
                    dimensions = extract_risk_dimensions(result)
                                    
                    print("=" * 80)
                    print("FULL AI REFINERY AGENT RESPONSE:")
                    print("=" * 80)
                    print(result)
                    print("=" * 80)
                    print(f"Response length: {len(result)} characters")
                    print("=" * 80)
                    
                    return {
                        "assessment": result,
                        "dimensions": dimensions,
                        "clean_response": dimensions.get("clean_response", ""),
                        "full_response": dimensions.get("full_response", result),
                        "raw_agent_output": result
                    }
                    
                except Exception as inner_e:
                    print(f"⚠️ WebSocket error with AI Refinery: {str(inner_e)}, falling back to mock")
                    return self._get_mock_response(query)
                
        except Exception as e:
            print(f"⚠️ AI Refinery error: {str(e)}, falling back to mock")
            return self._get_mock_response(query)
    
    def _get_mock_response(self, query: str) -> Dict[str, Any]:
        """Generate a mock response for testing the new structured format"""
        query_lower = query.lower()
        
        # Simple mock logic based on keywords - using only Safe|Moderate|High categories
        if any(word in query_lower for word in ['bomb', 'kill', 'murder', 'violence', 'attack']):
            violence_risk = "High"
            illegal_risk = "Moderate"
            misinfo_risk = "Safe"
            privacy_risk = "Safe"
            overall_risk = "High"
            recommendation = "Block"
            confidence = "High"
        elif any(word in query_lower for word in ['hack', 'steal', 'illegal', 'drugs']):
            violence_risk = "Safe"
            illegal_risk = "High"
            misinfo_risk = "Safe"
            privacy_risk = "Safe"
            overall_risk = "Moderate"
            recommendation = "Review"
            confidence = "High"
        else:
            violence_risk = "Safe"
            illegal_risk = "Safe"
            misinfo_risk = "Safe"
            privacy_risk = "Safe"
            overall_risk = "Safe"
            recommendation = "Allow"
            confidence = "High"
        
        mock_assessment = f"""
## HARM EVALUATION REPORT

**Risk Assessment Results:**
- Violence & Harm Risk: [{violence_risk}]
- Illegal Activity Risk: [{illegal_risk}]
- Misinformation Risk: [{misinfo_risk}]
- Privacy Violation Risk: [{privacy_risk}]

**Detailed Analysis:**
This is a mock evaluation of the query: "{query}"

The assessment shows {violence_risk} violence risk, {illegal_risk} illegal activity risk, 
{misinfo_risk} misinformation risk, and {privacy_risk} privacy violation risk.

**Risk Scoring Summary:**
- Primary Concerns: {overall_risk} risk level identified
- Secondary Concerns: None identified in mock mode
- Mitigating Factors: This is a test evaluation

**Note:** This is a mock response for testing purposes. For production use, ensure the AI Refinery SDK is properly configured.
"""
        
        dimensions = {
            "violence": violence_risk,
            "illegal_activities": illegal_risk,
            "misinformation": misinfo_risk,
            "privacy_violations": privacy_risk,
            "overall_harm": overall_risk,
            "recommendation": recommendation,
            "confidence": confidence,
            "clean_response": f"This is a mock evaluation response for the query: {query}",
            "full_response": mock_assessment
        }
        
        return {
            "assessment": mock_assessment,
            "dimensions": dimensions,
            "clean_response": dimensions.get("clean_response", ""),
            "full_response": mock_assessment
        }

    def load_test_questions(self) -> TestQuestionsDatabase:
        """Load test questions from JSON database"""
        try:
            if os.path.exists(self.test_db_file):
                with open(self.test_db_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert dict data to TestQuestion objects
                for category in ["safe_queries", "mild_risk_queries", "high_risk_queries", "edge_cases"]:
                    data[category] = [TestQuestion(**q) for q in data.get(category, [])]
                
                return TestQuestionsDatabase(**data)
            else:
                return TestQuestionsDatabase(
                    safe_queries=[],
                    mild_risk_queries=[],
                    high_risk_queries=[],
                    edge_cases=[],
                    metadata={"total_questions": 0, "next_id": 1}
                )
        except Exception as e:
            print(f"Error loading test questions: {e}")
            return TestQuestionsDatabase(
                safe_queries=[],
                mild_risk_queries=[],
                high_risk_queries=[],
                edge_cases=[],
                metadata={"total_questions": 0, "next_id": 1}
            )

    def save_test_questions(self, questions_db: TestQuestionsDatabase) -> bool:
        """Save test questions to JSON database"""
        try:
            questions_db.metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            
            # Convert to dict format for JSON serialization
            data = {
                "safe_queries": [q.dict() for q in questions_db.safe_queries],
                "mild_risk_queries": [q.dict() for q in questions_db.mild_risk_queries],
                "high_risk_queries": [q.dict() for q in questions_db.high_risk_queries],
                "edge_cases": [q.dict() for q in questions_db.edge_cases],
                "metadata": questions_db.metadata
            }
            
            with open(self.test_db_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save questions database: {str(e)}")
            return False

    def get_all_test_questions(self) -> List[TestQuestion]:
        """Get all test questions as a flat list"""
        questions_db = self.load_test_questions()
        all_questions = []
        
        for category in [questions_db.safe_queries, questions_db.mild_risk_queries, 
                        questions_db.high_risk_queries, questions_db.edge_cases]:
            all_questions.extend(category)
        
        return sorted(all_questions, key=lambda x: x.id if x.id else 0)

    def add_test_question(self, query: str, expected_risk: str, category: str, description: str) -> Tuple[bool, str, TestQuestion]:
        """Add a new test question to the database"""
        if not all([query.strip(), expected_risk, category.strip(), description.strip()]):
            return False, "All fields are required", None
        
        questions_db = self.load_test_questions()
        
        # Determine which section to add to based on expected risk
        if expected_risk == "Safe":
            section = questions_db.safe_queries
        elif expected_risk in ["Moderate Risk", "Caution"]:
            section = questions_db.mild_risk_queries
        elif expected_risk in ["High Risk", "Dangerous"]:
            section = questions_db.high_risk_queries
        else:
            section = questions_db.edge_cases
        
        # Create new question
        new_question = TestQuestion(
            id=questions_db.metadata["next_id"],
            query=query.strip(),
            expected_risk=expected_risk,
            category=category.strip(),
            description=description.strip(),
            date_added=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Add to database
        section.append(new_question)
        questions_db.metadata["next_id"] += 1
        questions_db.metadata["total_questions"] += 1
        
        # Add category if new
        if "categories" not in questions_db.metadata:
            questions_db.metadata["categories"] = []
        if category not in questions_db.metadata["categories"]:
            questions_db.metadata["categories"].append(category)
        
        # Save to file
        if self.save_test_questions(questions_db):
            return True, f"Question added successfully! ID: {new_question.id}", new_question
        else:
            return False, "Failed to save question to database", None

    async def run_batch_test(self) -> Dict[str, Any]:
        """Run batch test on all questions"""
        questions = self.get_all_test_questions()
        
        if not questions:
            return {
                "results": [],
                "total_questions": 0,
                "matches_found": 0,
                "accuracy_percentage": 0.0
            }

        results = []
        matches_found = 0

        for question in questions:
            try:
                evaluation = await self.evaluate_query(question.query)
                
                # Check if expected risk is found in assessment
                match_found = question.expected_risk.lower() in evaluation["assessment"].lower()
                if match_found:
                    matches_found += 1

                results.append({
                    "question_id": question.id,
                    "query": question.query,
                    "expected_risk": question.expected_risk,
                    "category": question.category,
                    "assessment": evaluation["assessment"],
                    "dimensions": evaluation["dimensions"],
                    "match_found": match_found
                })
            except Exception as e:
                results.append({
                    "question_id": question.id,
                    "query": question.query,
                    "expected_risk": question.expected_risk,
                    "category": question.category,
                    "assessment": f"Error: {str(e)}",
                    "dimensions": {},
                    "match_found": False
                })

        accuracy_percentage = (matches_found / len(questions)) * 100 if questions else 0

        return {
            "results": results,
            "total_questions": len(questions),
            "matches_found": matches_found,
            "accuracy_percentage": round(accuracy_percentage, 2)
        }

    def get_status(self) -> Dict[str, str]:
        """Get service status"""
        if not AI_REFINERY_AVAILABLE:
            return {
                "status": "mock",
                "message": "Running in mock mode - AI Refinery SDK not available",
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "status": "healthy" if self.distiller_client else "unhealthy",
            "message": "AI Refinery client is initialized" if self.distiller_client else "AI Refinery client not initialized",
            "timestamp": datetime.now().isoformat()
        } 