#!/usr/bin/env python3
"""
Test OpenEvidence-level extraction with a sample medical document.
"""

import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def test_openevidence_extraction():
    """Test extraction with LIBERATE trial-like content."""
    
    # Sample text mimicking LIBERATE trial
    sample_text = """
    Title: A Multicenter RCT of Zephyr Endobronchial Valve Treatment in Heterogeneous Emphysema (LIBERATE)
    
    Authors: Criner GJ, Sue R, Wright S, et al.
    Journal: Am J Respir Crit Care Med 2018;198(9):1151-1164
    Trial Registration: NCT01796392
    
    ABSTRACT
    Background: The Zephyr Endobronchial Valve (EBV) redirects airflow through healthier lung tissue.
    
    Methods: We conducted a multicenter randomized controlled trial at 24 sites. Patients with heterogeneous 
    emphysema and little to no collateral ventilation were randomized 2:1 to EBV treatment (n=128) or 
    standard of care (n=62). Primary outcome was the percentage of subjects with ≥15% improvement in FEV1 
    at 12 months post-procedure.
    
    Results: At 12 months, 47.7% of EBV subjects vs 16.8% of controls achieved ≥15% FEV1 improvement 
    (difference 29.2%, 95% CI 16.0-42.3, p<0.001). Mean FEV1 change was +106 mL for EBV vs +3 mL for 
    controls (p<0.001). Six-minute walk distance improved by 39.3m in EBV group vs -26.3m in controls 
    (p<0.001). SGRQ improved by -7.05 points for EBV vs -0.5 for controls (p=0.004).
    
    Pneumothorax occurred in 34/128 (26.6%) EBV subjects, most within 45 days. Four deaths occurred 
    in EBV group (3.1%) vs 0 in controls during 12-month follow-up.
    
    Conclusions: Zephyr EBV treatment in heterogeneous emphysema with intact fissures significantly 
    improves lung function, exercise capacity, and quality of life, with an acceptable safety profile.
    
    DETAILED RESULTS (Table 2, Page 1156):
    Primary Endpoint - FEV1 Responders ≥15% at 12 months:
    - EBV Group: 61/128 (47.7%)
    - Control Group: 10/62 (16.1%)
    - Difference: 31.6% (95% CI: 18.6% to 44.6%), p<0.001
    
    Secondary Endpoints at 12 months:
    - Absolute FEV1 Change (L):
      EBV: +0.106 ± 0.23, Control: -0.003 ± 0.19, Difference: 0.109 L, p<0.001
    
    - 6MWD Change (meters):
      EBV: +39.3 ± 85.2, Control: -26.3 ± 73.8, Difference: 65.6 m, p<0.001
      
    - SGRQ Total Score Change:
      EBV: -7.05 ± 13.2, Control: -0.5 ± 11.3, Difference: -6.55 points, p=0.004
    
    SAFETY (Table 3, Page 1158):
    Serious Adverse Events within 45 days:
    - Pneumothorax: EBV 34/128 (26.6%), Control 0/62 (0%)
    - COPD Exacerbation: EBV 10/128 (7.8%), Control 3/62 (4.8%)
    - Pneumonia: EBV 4/128 (3.1%), Control 2/62 (3.2%)
    - Death: EBV 1/128 (0.8%), Control 0/62 (0%)
    
    Management of Pneumothorax:
    - Chest tube required: 30/34 cases
    - Valve removal: 4/34 cases
    - Median time to resolution: 5 days
    """
    
    prompt = f"""Extract OpenEvidence-level structured data from this medical paper.

{sample_text}

Return a complete JSON object with:
- source (document_id, trial_registration_id, etc)
- document (metadata with integer year, sections)
- pico (population, intervention, comparison, outcomes)
- design (study details)
- arms (with exact n_randomized)
- outcomes_normalized (with raw data, statistics, and provenance)
- safety_normalized (adverse events with rates and provenance)
- risk_of_bias assessment
- retrieval (keywords, summary)

Ensure:
- Year is INTEGER not string
- Include page numbers in provenance
- Extract exact numbers for events/totals
- Calculate proper statistics
- Use ISO 8601 for timepoints (P12M = 12 months)

Return ONLY the JSON:"""

    print("Testing OpenEvidence Extraction")
    print("="*60)
    
    try:
        client = OpenAI(timeout=60)
        
        print("Calling GPT-5...")
        response = client.responses.create(
            model="gpt-5",
            input=prompt
        )
        
        if hasattr(response, 'output_text'):
            result_text = response.output_text
        else:
            print("ERROR: Unexpected response format")
            return False
        
        # Clean and parse JSON
        if "```" in result_text:
            result_text = result_text.replace("```json", "").replace("```", "")
        
        # Fix common issues
        result_text = result_text.replace("−", "-").replace("–", "-")
        
        # Find JSON
        start = result_text.find("{")
        end = result_text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = result_text[start:end]
        else:
            json_str = result_text
        
        result = json.loads(json_str)
        
        print("\n✓ Extraction successful!")
        print("\nKey extracted data:")
        print("-"*40)
        
        # Validate key fields
        if "document" in result and "metadata" in result["document"]:
            meta = result["document"]["metadata"]
            print(f"Title: {meta.get('title', 'N/A')}")
            print(f"Year: {meta.get('year')} (type: {type(meta.get('year')).__name__})")
            
        if "arms" in result:
            print(f"\nArms ({len(result['arms'])}):")
            for arm in result["arms"]:
                print(f"  - {arm.get('name')}: N={arm.get('n_randomized')}")
        
        if "outcomes_normalized" in result:
            print(f"\nOutcomes ({len(result['outcomes_normalized'])}):")
            for outcome in result["outcomes_normalized"][:3]:
                print(f"  - {outcome.get('name')}")
                if outcome.get('comparison'):
                    comp = outcome['comparison']
                    print(f"    Effect: {comp.get('est')} (p={comp.get('p_value')})")
                if outcome.get('provenance'):
                    prov = outcome['provenance']
                    print(f"    Source: {prov.get('tables', [])} page {prov.get('pages', [])}")
        
        if "safety_normalized" in result:
            print(f"\nSafety Events ({len(result['safety_normalized'])}):")
            for event in result["safety_normalized"][:3]:
                print(f"  - {event.get('event_name')}")
                for group in event.get('groups', []):
                    print(f"    {group.get('arm_id')}: {group.get('patients')}/{group.get('events')} events")
        
        # Save output
        output_file = "test_openevidence_output.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\nFull output saved to: {output_file}")
        print("\n✓ All tests passed!")
        return True
        
    except json.JSONDecodeError as e:
        print(f"\n✗ JSON parsing failed: {e}")
        print(f"Raw output: {result_text[:500]}")
        return False
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False


if __name__ == "__main__":
    success = test_openevidence_extraction()
    if not success:
        print("\nTroubleshooting:")
        print("1. Check OPENAI_API_KEY is set")
        print("2. Ensure GPT-5 access is enabled")
        print("3. Review raw output for formatting issues")