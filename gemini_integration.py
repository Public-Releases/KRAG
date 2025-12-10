import os
import json
import time
from typing import Optional, Dict, Any

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")

def generate_with_gemini(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model_name: str = "gemini-1.5-pro-latest",
    temperature: float = 0.3,
    max_tokens: Optional[int] = None
) -> str:
    """
    Generate text using Google Gemini API
    
    Args:
        system_prompt: System instructions for the model
        user_prompt: User's prompt/question
        api_key: Google AI Studio API key
        model_name: Gemini model to use (gemini-1.5-pro-latest, gemini-1.5-flash, etc.)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text response
    """
    
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai package not installed")
    
    if not api_key:
        raise ValueError("Google API key not configured in authentication.py")
    
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
    }
    
    if max_tokens:
        generation_config["max_output_tokens"] = max_tokens
    
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        system_instruction=system_prompt
    )
    
    try:
        response = model.generate_content(user_prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        raise

def generate_json_with_gemini(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model_name: str = "gemini-1.5-pro-latest",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate JSON using Gemini with automatic parsing
    """
    
    # Enhance prompts for JSON output
    enhanced_system = f"""{system_prompt}

CRITICAL: You MUST respond with valid JSON only. No explanations, no markdown, no text before or after the JSON object."""
    
    enhanced_user = f"""{user_prompt}

Response format: Return ONLY the JSON object. Start with {{ and end with }}. No markdown formatting, no explanations."""
    
    # Generate response
    response_text = generate_with_gemini(
        system_prompt=enhanced_system,
        user_prompt=enhanced_user,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Parse JSON
    try:
        # Try direct parsing
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        # Try extracting from markdown code blocks
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
            return json.loads(json_str)
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
            return json.loads(json_str)
        else:
            # Try to extract JSON object
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
    
    return None