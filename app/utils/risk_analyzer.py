import re
from typing import Dict

def extract_risk_dimensions(assessment: str) -> Dict[str, str]:
    """Extract specific risk dimensions from structured harm evaluation response."""
    
    dimensions = {
        "violence": "Unknown", 
        "illegal_activities": "Unknown",
        "misinformation": "Unknown", 
        "privacy_violations": "Unknown", 
        "overall_harm": "Unknown",
        "recommendation": "Unknown", 
        "confidence": "Unknown",
        "clean_response": assessment, 
        "full_response": assessment
    }
    
    try:
        # Enhanced patterns for the new structured format
        risk_patterns = {
            "violence": [
                r'Violence & Harm Risk:\s*\[([^\]]+)\]',
                r'Violence & Harm Risk:\s*([A-Za-z]+)',
                r'Violence[^:]*:\s*([A-Za-z]+)'
            ],
            "illegal_activities": [
                r'Illegal Activity Risk:\s*\[([^\]]+)\]',
                r'Illegal Activity Risk:\s*([A-Za-z]+)',
                r'Illegal[^:]*:\s*([A-Za-z]+)'
            ],
            "misinformation": [
                r'Misinformation Risk:\s*\[([^\]]+)\]',
                r'Misinformation Risk:\s*([A-Za-z]+)',
                r'Misinformation[^:]*:\s*([A-Za-z]+)'
            ],
            "privacy_violations": [
                r'Privacy Violation Risk:\s*\[([^\]]+)\]',
                r'Privacy Violation Risk:\s*([A-Za-z]+)',
                r'Privacy[^:]*:\s*([A-Za-z]+)'
            ]
        }
        
        # Extract each risk dimension using multiple patterns
        for dim, pattern_list in risk_patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, assessment, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    # Clean up the value
                    value = re.sub(r'[\*\[\]]+', '', value).strip()
                    if value and value.lower() not in ['unknown', '']:
                        dimensions[dim] = value
                        break

        # Enhanced overall harm extraction
        harm_patterns = [
            r'Overall Risk Classification:\s*\[([^\]]+)\]',
            r'Overall Risk Classification:\s*([A-Za-z\s]+?)(?:\n|$|\*)',
            r'Overall[^:]*:\s*([A-Za-z\s]+?)(?:\n|$|\*)',
        ]
        
        for pattern in harm_patterns:
            match = re.search(pattern, assessment, re.IGNORECASE | re.MULTILINE)
            if match:
                value = re.sub(r'[\*\[\]]+', '', match.group(1)).strip()
                if value and value.lower() not in ['unknown', '']:
                    dimensions["overall_harm"] = value
                    break

        # Enhanced recommendation extraction
        rec_patterns = [
            r'Recommendation:\s*\[([^\]]+)\]',
            r'Recommendation:\s*([A-Za-z\s]+?)(?:\n|$|\*)',
            r'Action[^:]*:\s*([A-Za-z\s]+?)(?:\n|$|\*)',
        ]
        
        for pattern in rec_patterns:
            match = re.search(pattern, assessment, re.IGNORECASE | re.MULTILINE)
            if match:
                value = re.sub(r'[\*\[\]]+', '', match.group(1)).strip()
                if value and value.lower() not in ['unknown', '']:
                    dimensions["recommendation"] = value
                    break

        # Extract confidence level
        conf_patterns = [
            r'Confidence Level:\s*\[([^\]]+)\]',
            r'Confidence Level:\s*([A-Za-z\s]+?)(?:\n|$|\*)',
            r'Confidence[^:]*:\s*([A-Za-z\s]+?)(?:\n|$|\*)',
        ]
        
        for pattern in conf_patterns:
            match = re.search(pattern, assessment, re.IGNORECASE | re.MULTILINE)
            if match:
                value = re.sub(r'[\*\[\]]+', '', match.group(1)).strip()
                if value and value.lower() not in ['unknown', '']:
                    dimensions["confidence"] = value
                    break

        # Inference logic for missing values - using only Safe|Moderate|High categories
        risk_levels = [dimensions[d] for d in ["violence", "illegal_activities", "misinformation", "privacy_violations"] 
                      if dimensions[d] != "Unknown"]
        
        if risk_levels and dimensions["overall_harm"] == "Unknown":
            if all(level.lower() == 'safe' for level in risk_levels):
                dimensions["overall_harm"] = "Safe"
            elif any(level.lower() == 'high' for level in risk_levels):
                dimensions["overall_harm"] = "High"
            elif any(level.lower() == 'moderate' for level in risk_levels):
                dimensions["overall_harm"] = "Moderate"
            else:
                dimensions["overall_harm"] = "Safe"  # Default to Safe if unclear
        
        if dimensions["recommendation"] == "Unknown" and dimensions["overall_harm"] != "Unknown":
            if dimensions["overall_harm"].lower() == "safe":
                dimensions["recommendation"] = "Allow"
            elif dimensions["overall_harm"].lower() == "moderate":
                dimensions["recommendation"] = "Review"
            elif dimensions["overall_harm"].lower() == "high":
                dimensions["recommendation"] = "Block"

        # Extract clean response (remove the structured assessment parts)
        clean_response = assessment
        
        # Remove the structured assessment sections
        clean_patterns = [
            r'## HARM EVALUATION REPORT.*?(?=\n\n|\Z)',
            r'\*\*Risk Assessment Results:\*\*.*?(?=\n\n|\Z)',
            r'\*\*Overall Risk Classification:\*\*.*?(?=\n\n|\Z)',
            r'\*\*Recommendation:\*\*.*?(?=\n\n|\Z)',
            r'\*\*Confidence Level:\*\*.*?(?=\n\n|\Z)',
            r'\*\*Risk Scoring Summary:\*\*.*?(?=\n\n|\Z)',
            r'- Violence & Harm Risk:.*?(?=\n|\Z)',
            r'- Illegal Activity Risk:.*?(?=\n|\Z)',
            r'- Misinformation Risk:.*?(?=\n|\Z)',
            r'- Privacy Violation Risk:.*?(?=\n|\Z)',
        ]
        
        for pattern in clean_patterns:
            clean_response = re.sub(pattern, '', clean_response, flags=re.DOTALL | re.IGNORECASE)
        
        # Look for the detailed analysis section
        analysis_match = re.search(r'\*\*Detailed Analysis:\*\*\s*(.+?)(?=\*\*|$)', assessment, re.DOTALL | re.IGNORECASE)
        if analysis_match:
            dimensions["clean_response"] = analysis_match.group(1).strip()
        else:
            dimensions["clean_response"] = clean_response.strip() or "Assessment completed - see full response for details."

        return dimensions
        
    except Exception as e:
        print(f"Error extracting risk dimensions: {e}")
        print(f"Assessment preview: {assessment[:300]}...")
        return dimensions

def format_risk_analysis(dimensions: Dict[str, str]) -> str:
    """Format risk analysis in a structured format"""
    analysis = "**Harm Evaluation Results:**\n\n"
    analysis += f"* Violence & Harm: {dimensions.get('violence', 'Unknown')}\n"
    analysis += f"* Illegal Activities: {dimensions.get('illegal_activities', 'Unknown')}\n"
    analysis += f"* Misinformation: {dimensions.get('misinformation', 'Unknown')}\n"
    analysis += f"* Privacy Violations: {dimensions.get('privacy_violations', 'Unknown')}\n\n"
    analysis += f"**Overall Risk Classification:** {dimensions.get('overall_harm', 'Unknown')}\n\n"
    analysis += f"**Recommendation:** {dimensions.get('recommendation', 'Unknown')}\n\n"
    analysis += f"**Confidence Level:** {dimensions.get('confidence', 'Unknown')}\n"
    
    return analysis 