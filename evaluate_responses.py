import pandas as pd
import json
import time
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse
from gemini_integration import generate_with_gemini
from authentication import google_api_key

def create_rag_evaluation_prompt(patient_query: str, clean_symptoms: str, kg_context: str, true_diagnosis: str) -> str:
    prompt = f"""
    You are an expert medical evaluator. You evaluate the RAG retrieval results for coherence, medical accuracy, quality, and relevance to patient symptoms. Always respond with valid JSON only.
    
    **Patient Information:**
    Patient Query: {patient_query}
    Symptoms: {clean_symptoms}
    
    **Knowledge Graph Context:**
    {kg_context}
    
    **Doctor's Diagnosis:**
    {true_diagnosis}
    
    **Evaluation Criteria:**
    - Coherence: Does the RAG retrieval results make sense given the patient symptoms?
    - Medical Accuracy: Are the RAG retrieval results medically accurate?
    - Quality: Are the RAG retrieval results of good quality?
    - Relevance: Are the RAG retrieval results relevant to the patient symptoms?
    - Accuracy: Are the RAG retrieval results accurate compared to the doctor's diagnosis?
    
    **Scoring Guidelines:**
    - 9-10: Excellent - Highly coherent, medically sound, well-structured
    - 7-8: Good - Mostly coherent with minor issues
    - 5-6: Fair - Some coherence but notable gaps or inconsistencies
    - 3-4: Poor - Significant coherence issues or medical inconsistencies
    - 1-2: Very Poor - Largely incoherent or medically unsound
    
    **Output Format:**
    Provide ONLY a JSON object with the following structure:
    {{
        "score": <integer from 1-10>,
        "reasoning": "<brief explanation of the score, 2-3 sentences>"
    }}
    
    Do not include any text before or after the JSON object.
    """
    return prompt

def create_evaluation_prompt(patient_query: str, clean_symptoms: str, generated_report: str, true_diagnosis: str) -> str:
    """
    Create a prompt for Gemini to evaluate the generated report.
    
    Args:
        patient_query: The original patient query/symptoms
        clean_symptoms: Cleaned symptoms list
        generated_report: The generated diagnosis report to evaluate
        true_diagnosis: True diagnosis from the ground truth
    Returns:
        Formatted evaluation prompt
    """
    prompt = f"""You are an expert medical evaluator. Your task is to evaluate the quality and coherence of a generated medical diagnosis report.

**Patient Information:**
Patient Query: {patient_query}
Symptoms: {clean_symptoms}

**Generated Report:**
{generated_report}

**Doctor's Diagnosis:**
{true_diagnosis}

**Evaluation Criteria:**
Rate the generated report on a scale of 1-10 based on:
1. **Medical Coherence**: Does the diagnosis make sense given the symptoms?
2. **Logical Consistency**: Are the explanations, treatments, and recommendations consistent with the diagnosis?
3. **Completeness**: Does the report address the key aspects (diagnosis, explanations, treatments)?
4. **Relevance**: Are the suggested treatments and evaluations appropriate for the condition?
5. **Overall Quality**: Does the report read as a coherent, professional medical assessment?
6. **Accuracy**: How accurate is the diagnosis compared to the doctor's diagnosis?

**Scoring Guidelines:**
- 9-10: Excellent - Highly coherent, medically sound, well-structured
- 7-8: Good - Mostly coherent with minor issues
- 5-6: Fair - Some coherence but notable gaps or inconsistencies
- 3-4: Poor - Significant coherence issues or medical inconsistencies
- 1-2: Very Poor - Largely incoherent or medically unsound

**Output Format:**
Provide ONLY a JSON object with the following structure:
{{
    "score": <integer from 1-10>,
    "reasoning": "<brief explanation of the score, 2-3 sentences>"
}}

Do not include any text before or after the JSON object."""
    return prompt


def evaluate_single_response(
    patient_query: str,
    clean_symptoms: str,
    generated_report: str,
    kg_context: str,
    true_diagnosis: str,
    api_key: str,
    max_retries: int = 3
) -> Tuple[int, str]:
    """
    Evaluate a single generated response using Gemini 2.0 Flash.
    """
    system_prompt = """You are an expert medical evaluator. You evaluate both the RAG retrieval results and AI generated diagnosis reportfor coherence, medical accuracy, quality, and relevance to patient symptoms. Always respond with valid JSON only."""
    
    user_prompt = create_evaluation_prompt(patient_query, clean_symptoms, generated_report, true_diagnosis)
    user_prompt_rag = create_rag_evaluation_prompt(patient_query, clean_symptoms, kg_context, true_diagnosis)
    for attempt in range(max_retries):
        try:
            
            response = generate_with_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_key=api_key,
                model_name="models/gemini-2.0-flash",
                temperature=0.3,
                max_tokens=500
            )
            response_rag = generate_with_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt_rag,
                api_key=api_key,
                model_name="models/gemini-2.0-flash",
                temperature=0.3,
                max_tokens=500
            )
            # Try to parse JSON from response
            response = response.strip()
            response_rag = response_rag.strip()
            # Remove markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            if "```json" in response_rag:
                response_rag = response_rag.split("```json")[1].split("```")[0].strip()
            elif "```" in response_rag:
                response_rag = response_rag.split("```")[1].split("```")[0].strip()
            # Extract JSON object
            if response.startswith("{"):
                json_str = response
            else:
                # Find JSON object boundaries
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                else:
                    raise ValueError("No JSON object found in response")
            if response_rag.startswith("{"):
                json_str_rag = response_rag
            else:
                # Find JSON object boundaries
                start_idx = response_rag.find("{")
                end_idx = response_rag.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str_rag = response_rag[start_idx:end_idx]
                else:
                    raise ValueError("No JSON object found in response")
            result = json.loads(json_str)
            result_rag = json.loads(json_str_rag)
            score = int(result.get("score", 5))
            reasoning = result.get("reasoning", "No reasoning provided")
            score_rag = int(result_rag.get("score", 5))
            reasoning_rag = result_rag.get("reasoning", "No reasoning provided")
            
            # Validate score is in range
            if score < 1 or score > 10:
                score = 5  # Default to middle score if invalid
                reasoning = f"Invalid score provided, defaulted to 5. Original: {reasoning}"
            
            return score, reasoning, score_rag, reasoning_rag
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue
            else:
                print(f"Warning: Failed to parse JSON after {max_retries} attempts. Response: {response[:200]}")
                return 5, f"Failed to parse evaluation response: {str(e)}"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                print(f"Warning: Error evaluating response: {str(e)}")
                return 5, f"Error during evaluation: {str(e)}"
    
    return 5, "Failed to evaluate after multiple attempts"

def evaluate_results_csv(
    csv_path: str,
    output_path: str = None,
    max_samples: int = None,
    api_key: str = None
) -> Dict:
    """
    Evaluate all responses in a results CSV file.
    
    Args:
        csv_path: Path to the results CSV file
        output_path: Optional path to save detailed results
        max_samples: Optional limit on number of samples to evaluate
        api_key: Google API key (uses authentication.py if not provided)
        
    Returns:
        Dictionary with evaluation statistics
    """
    if api_key is None:
        api_key = google_api_key
    
    if not api_key or api_key == "your_google_api_key_here":
        raise ValueError("Google API key not configured. Please set it in authentication.py")
    
    print(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path, quoting=1)
    
    if max_samples:
        df = df.head(max_samples)
        print(f"Evaluating first {max_samples} samples...")
    else:
        print(f"Evaluating all {len(df)} samples...")
    
    scores = []
    evaluations = []
    scores_rag = []
    
    print("\nEvaluating responses with Gemini 2.0 Flash...")
    print("=" * 80)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        participant_no = row.get("Participant No.", idx + 1)
        patient_query = str(row.get("Patient Query", ""))
        clean_symptoms = str(row.get("Clean Symptoms", ""))
        generated_report = str(row.get("Generated Report", ""))
        kg_context = str(row.get("KG Context", ""))
        true_diagnosis = str(row.get("True Diagnosis", ""))
        if not generated_report or generated_report == "nan":
            print(f"\nWarning: Skipping participant {participant_no} - missing generated report")
            continue
        
        score, reasoning, score_rag, reasoning_rag = evaluate_single_response(
            patient_query=patient_query,
            clean_symptoms=clean_symptoms,
            generated_report=generated_report,
            kg_context=kg_context,
            true_diagnosis=true_diagnosis,
            api_key=api_key
        )
        
        scores.append(score)
        scores_rag.append(score_rag)
        evaluations.append({
            "Participant No.": participant_no,
            "Score": score,
            "Reasoning": reasoning,
            "Score RAG": score_rag,
            "Reasoning RAG": reasoning_rag,
            "Generated Diagnosis": row.get("Generated Diagnosis", ""),
            "True Diagnosis": row.get("True Diagnosis", ""),
            "KG Context": kg_context,
            "KG Context RAG": kg_context,
        })
        
        time.sleep(0.5)
    
    if not scores:
        print("No valid evaluations completed!")
        return {}
    
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    avg_score_rag = sum(scores_rag) / len(scores_rag)
    min_score_rag = min(scores_rag)
    max_score_rag = max(scores_rag)
    score_distribution = {}
    for score in range(1, 11):
        score_distribution[score] = scores.count(score)
    score_distribution_rag = {}
    for score in range(1, 11):
        score_distribution_rag[score] = scores_rag.count(score)
    results = {
        "total_evaluated": len(scores),
        "average_score": round(avg_score, 2),
        "min_score": min_score,
        "max_score": max_score,
        "score_distribution": score_distribution,
        "evaluations": evaluations,
        "average_score_rag": round(avg_score_rag, 2),
        "min_score_rag": min_score_rag,
        "max_score_rag": max_score_rag,
        "score_distribution_rag": score_distribution_rag,
    }
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Responses Evaluated: {results['total_evaluated']}")
    print(f"Average Score: {results['average_score']:.2f}/10")
    print(f"Minimum Score: {results['min_score']}/10")
    print(f"Maximum Score: {results['max_score']}/10")
    print("\nScore Distribution:")
    for score in range(1, 11):
        count = score_distribution[score]
        percentage = (count / len(scores)) * 100 if scores else 0
        bar = "█" * int(percentage / 2)
        print(f"  {score:2d}: {count:4d} ({percentage:5.1f}%) {bar}")
    print(f"Average Score RAG: {results['average_score_rag']:.2f}/10")
    print(f"Minimum Score RAG: {results['min_score_rag']}/10")
    print(f"Maximum Score RAG: {results['max_score_rag']}/10")
    print("\nScore Distribution RAG:")
    for score in range(1, 11):
        count = score_distribution_rag[score]
        percentage = (count / len(scores_rag)) * 100 if scores_rag else 0
        bar = "█" * int(percentage / 2)
        print(f"  {score:2d}: {count:4d} ({percentage:5.1f}%) {bar}")
    # Save detailed results if output path provided
    if output_path:
        # Save as CSV
        eval_df = pd.DataFrame(evaluations)
        eval_df.to_csv(output_path, index=False)
        print(f"\nDetailed results saved to: {output_path}")
        
        # Save summary as JSON
        summary_path = output_path.replace(".csv", "_summary.json")
        summary = {
            "total_evaluated": results["total_evaluated"],
            "average_score": results["average_score"],
            "min_score": results["min_score"],
            "max_score": results["max_score"],
            "score_distribution": results["score_distribution"],
            "average_score_rag": results["average_score_rag"],
            "min_score_rag": results["min_score_rag"],
            "max_score_rag": results["max_score_rag"],
            "score_distribution_rag": results["score_distribution_rag"],
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated medical diagnosis reports using Gemini 1.5 Pro"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the results CSV file to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed evaluation results (default: <input_name>_evaluations.csv)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        import os
        base_name = os.path.splitext(args.csv_path)[0]
        args.output = f"{base_name}_evaluations.csv"
    
    # Run evaluation
    results = evaluate_results_csv(
        csv_path=args.csv_path,
        output_path=args.output,
        max_samples=args.max_samples
    )
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
