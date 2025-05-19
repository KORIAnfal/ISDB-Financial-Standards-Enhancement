import re
from collections import defaultdict
# Removed statistics import as it's only needed in aggregation.py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from utils import invoke_agent_with_retry

# Validator Agent Personas (kept here as central config for validators)
VALIDATOR_PERSONAS = {
    "Technical_Validator": {
        "name": "Technical Feasibility Validator",
        "description": "Validates proposals based on their technical feasibility, implementability using modern technology (including AI), and operational challenges.",
    },
    "Shariah_Validator": {
        "name": "Shari’ah Compliance Validator",
        "description": "Validates proposals based on alignment with Shari’ah principles, avoidance of prohibited elements (Riba, Gharar, Maysir), risk-sharing, and Shari’ah justification.",
    },
    "Global_Consistency_Validator": {
        "name": "Global Consistency Validator",
        "description": "Validates proposals based on their potential impact on global consistency in Islamic finance practices and alignment with broader accounting or regulatory trends.",
    },
     "Clarity_Auditability_Validator": {
        "name": "Clarity & Auditability Validator",
        "description": "Validates proposals based on whether they enhance the clarity of the standard, ease of understanding, and potential for independent audit and verification.",
    }
}


# Validator Agent (Specific - Reviews ALL proposals and gives a SCORE out of 100)
# CORRECTED: Ensure {finance_news} placeholder is NOT present
validator_template_str = """
**You are an Esteemed AAOIFI Expert Validator, acting as the {persona_name} ({persona_description}).** Your specific task is to rigorously validate *each* proposed enhancement presented in the provided SUMMARY below for the AAOIFI Standard: '{standard_name}' ('{standard_short_name}'). Assess each proposal based on its compliance, practicality, and clarity *from your specific {persona_name} perspective*, and provide a numerical score between 0.0 and 100.0. Consider the provided analysis output for context, as it incorporates relevant news-driven gaps.

**Standard Being Assessed:** '{standard_name}' ('{standard_short_name}')

**Input: Summarized Proposed Enhancements (from various perspectives):**
---
{all_proposals_summarized}
---

**Input: Original Standard Analysis (for context regarding '{standard_short_name}', including news-driven gaps):**
---
{analysis_output}
---


**Instructions - For *each* proposal found in the 'Summarized Proposed Enhancements' input, provide an assessment and score (0.0-100.0) from your {persona_name} perspective. Be precise and direct:**
Review each proposal summary. Evaluate it based on your {persona_name} expertise and the provided context (especially the Analysis).
1.  **Proposal Title:** Repeat the exact title of the proposal being assessed from the input summary.
2.  **Assessment ({persona_name} Perspective):** Evaluate the proposal summary based on your specific criteria (defined by your persona). Reference relevant details from the summary or analysis. Be concise.
3.  **Score:** Provide a numerical score between 0.0 and 100.0. Format: **Score:** [0.0-100.0]
4.  **Justification ({persona_name} Perspective):** Brief explanation for your score and assessment. **If your score is <= 90.0, explicitly state the main fault or contradiction ({persona_name} issue, e.g., Technical issue, Shari’ah contradiction, Global consistency issue, Clarity/Auditability issue, etc.) from your perspective, referencing the Analysis or the proposal summary if applicable.**

**Output Format:** Provide a detailed validation report structured as a Markdown list, with assessment details, score, and justification for *each* individual proposal listed in the input summary. Use the exact proposal title from the input summary.

---
## Validation by {persona_name} for: [Exact Proposal Title from Input Summary]
**Assessment ({persona_name} Perspective):** ...
**Score:** [0.0-100.0]
**Justification ({persona_name} Perspective):** ... **(If score <= 90.0, explain fault/contradiction clearly)**
---
... (Continue for all proposals in the summary) ...
"""
validator_prompt = ChatPromptTemplate.from_template(validator_template_str)


def parse_validator_scores_and_justifications(markdown_text, validator_key):
    """
    Parses a single specific validator's markdown output, extracting proposal titles, scores, and justifications.
    Scores are expected 0-100 and are normalized to 0.0-1.0.
    Uses re.finditer for more robust extraction of validation blocks.
    """
    validations = []
    validator_name = VALIDATOR_PERSONAS.get(validator_key, {}).get("name", validator_key) # Get name from key

    # --- DEBUG PRINT ---
    print(f"\n--- Debug: Parsing Validator Output for '{validator_name}' ---")
    print(f"Raw Markdown snippet (first 500 chars):\n{markdown_text[:500]}...")
    # --- END DEBUG PRINT ---

    # Use finditer to find all occurrences of the validation header
    # Pattern looks for ## Validation by [anything] for: [title]
    validation_matches = list(re.finditer(r'##\s*Validation by .+? for: (.*)', markdown_text, re.DOTALL))

    # --- DEBUG PRINT ---
    print(f"--- Debug: Found {len(validation_matches)} validation block starts based on pattern. ---")
    for i, match in enumerate(validation_matches):
         print(f"  - Match {i+1} found at index {match.start()}, Title candidate: '{match.group(1).strip()[:100]}...'")
    print("------------------------------------------------------------")
    # --- END DEBUG PRINT ---


    for i, match in enumerate(validation_matches):
        title = match.group(1).strip() # This is the Proposal Title being validated
        start_index = match.end() # Start parsing content after the title captured by the header regex

        # Determine the end index for the current validation's content
        end_index = len(markdown_text)
        if i + 1 < len(validation_matches):
            # If there's a next validation block, the current one ends before it
            end_index = validation_matches[i+1].start()

        content = markdown_text[start_index:end_index].strip()

        # --- DEBUG PRINT ---
        print(f"\n--- Debug: Parsing Content for Title '{title}' (Validator '{validator_name}') ---")
        print(f"Content snippet (first 300 chars):\n{content[:300]}...")
        # --- END DEBUG PRINT ---


        validation_data = {
            "proposalTitle": title, # Use the exact title captured from the header
            "validatingValidator": validator_name,
            "score": None,
            "rawScore": None,
            "assessment": None,
            "justification": None,
            "individualVerdict": None
        }

        # Extract the Assessment (comes after the title, before Score or Justification)
        # Use re.search within the *current content block*
        assessment_match = re.search(r'\*\*Assessment \(.+?\s*Perspective\):\*\*\s*(.*?)(?=\n\*\*Score:|\n\*\*Justification:|$)', content, re.DOTALL)
        if not assessment_match: # Fallback
             assessment_match = re.search(r'\*\*Assessment:\*\*\s*(.*?)(?=\n\*\*Score:|\n\*\*Justification:|$)', content, re.DOTALL)
        if assessment_match:
             validation_data["assessment"] = assessment_match.group(1).strip()
        else:
            # --- DEBUG PRINT ---
            print(f"  - Assessment pattern not found for '{title}'.")
            # --- END DEBUG PRINT ---


        # Extract the Score (looking for **Score:** followed by a number)
        # Use re.search within the *current content block*
        score_match = re.search(r'\*\*Score:\*\*\s*([\d\.]+)', content) # Match digits and dot one or more times
        if score_match:
            score_str = score_match.group(1).strip()
            try:
                raw_score = float(score_str)
                raw_score = max(0.0, min(100.0, raw_score)) # Clamp
                validation_data["rawScore"] = raw_score
                validation_data["score"] = raw_score / 100.0 # Normalize
                 # --- DEBUG PRINT ---
                print(f"  - Successfully parsed score for '{title}': Raw={raw_score}, Normalized={validation_data['score']}")
                # --- END DEBUG PRINT ---
            except ValueError:
                # --- DEBUG PRINT ---
                print(f"  - Failed to parse score '{score_str}' as float for '{title}'.")
                # --- END DEBUG PRINT ---
                validation_data["score"] = None
                validation_data["rawScore"] = None
        else:
            # --- DEBUG PRINT ---
            print(f"  - Score pattern not found for '{title}'.")
            # --- END DEBUG PRINT ---


        # Extract Justification (comes after Score if present, or after Assessment, until next section or end)
        # Need to be careful about the order of fields in the content
        # Let's look for justification after either Assessment or Score
        justification_match = re.search(r'\*\*Justification \(.+?\s*Perspective\):\*\*\s*(.*)', content, re.DOTALL) # Look for specific header until end of content
        if not justification_match: # Fallback
             justification_match = re.search(r'\*\*Justification:\*\*\s*(.*)', content, re.DOTALL) # Look for simpler header until end of content

        if justification_match:
            validation_data["justification"] = justification_match.group(1).strip()
        else:
            # --- DEBUG PRINT ---
            print(f"  - Justification pattern not found for '{title}'.")
            # --- END DEBUG PRINT ---


        # Extract individual verdict if present (Optional, usually at the end)
        individual_verdict_match = re.search(r'\*\*Overall Verdict:\*\*\s*(.*?)(?:\s*\([^)]*\))?\n', content, re.DOTALL) # Use DOTALL for multi-line
        if individual_verdict_match:
             validation_data["individualVerdict"] = individual_verdict_match.group(1).strip()
             # --- DEBUG PRINT ---
             print(f"  - Parsed individual verdict for '{title}': '{validation_data['individualVerdict']}'")
             # --- END DEBUG PRINT ---
        else:
            # --- DEBUG PRINT ---
            print(f"  - Individual Verdict pattern not found for '{title}'.")
            # --- END DEBUG PRINT ---


        # Only add validation if a title was successfully extracted from the header
        if title:
             validations.append(validation_data)
        else:
             # --- DEBUG PRINT ---
             # This print is already in the finditer loop
             pass # print(f"  - Skipping validation block found at index {match.start()} because title was empty after parsing header.")
             # --- END DEBUG PRINT ---


    # --- DEBUG PRINT ---
    print(f"--- Debug: Finished Parsing for '{validator_name}'. Found {len(validations)} validations with titles. ---")
    for val in validations:
        print(f"  - Final Parsed Validated Title: '{val.get('proposalTitle')}', Score: {val.get('rawScore')}")
    print("------------------------------------------------------------")
    # --- END DEBUG PRINT ---


    return validations


def run_validator_agents(llm_structured: ChatOpenAI, standard_name: str, standard_short_name: str, analysis_output: str, all_proposals_summarized_markdown: str, max_retries: int, initial_retry_delay: float, RateLimitError, validator_personas):
    """Initializes and runs all Validator agents."""
    validator_outputs_by_name = {}
    print(f"\n--- Step 4: Running {len(validator_personas)} Validator Agents (Specific) ---")

    # Only run validators if there was a summary generated that contains actual items
    # Check if the summarized markdown looks like a summary list (starts with header and has list items)
    if all_proposals_summarized_markdown.startswith("## Summarized Proposals for") and "*" in all_proposals_summarized_markdown:
         for validator_key, validator_config in validator_personas.items():
            print(f"\nRunning Validator Agent: {validator_config['name']} ({validator_key})...")

            validator_chain = LLMChain(llm=llm_structured, prompt=validator_prompt)

            inputs = {
                "standard_name": standard_name,
                "standard_short_name": standard_short_name,
                "analysis_output": analysis_output,
                "all_proposals_summarized": all_proposals_summarized_markdown,
                "persona_name": validator_config["name"],
                "persona_description": validator_config["description"],
                # Finance news intentionally removed from validator prompt/inputs for brevity
                # **VERIFIED:** The placeholder {finance_news} is now also removed from validator_template_str above.
            }

            validation_result = invoke_agent_with_retry(validator_chain, inputs, max_retries, initial_retry_delay, RateLimitError)
            validator_output = validation_result['text'].strip()
            validator_outputs_by_name[validator_key] = validator_output

            print(f"\nRaw Validator Output ({validator_config['name']}):\n{validator_output}") # Keep raw output print

    else:
         print("Skipping validators as no significant proposals were generated or summarized.")
         # Populate outputs with a message indicating they were skipped
         validator_outputs_by_name = {vk: f"No proposals were generated or summarized, so validation by {vc['name']} was not performed." for vk, vc in validator_personas.items()}

    # Pass validator_personas back as well, needed for aggregation
    return validator_outputs_by_name, validator_personas