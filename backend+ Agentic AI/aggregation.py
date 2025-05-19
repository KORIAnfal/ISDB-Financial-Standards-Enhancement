from collections import defaultdict
import statistics
import re

# Import the parsing functions and personas from their modules
from proposers import parse_proposals_markdown # Only need the parser here now
from validators import parse_validator_scores_and_justifications # Only need the parser here now

# Personas are now passed from app.py during the call to aggregate_scores_and_verdicts

def aggregate_scores_and_verdicts(proposer_outputs_by_persona, validator_outputs_by_name, proposer_personas, validator_personas):
    """
    Aggregates proposals from all proposers (using raw outputs to get full details),
    collects scores for each proposal from specific validators,
    calculates mean scores, and determines final verdicts.
    Accepts proposer_personas and validator_personas dicts as arguments.
    """
    # Dictionary to store unique proposals {cleaned_lower_title: {original_details}}
    all_proposals_dict = {}
    # Dictionary to map cleaned_lower_title back to original case titles for final output
    original_titles_map = {}


    # 1. Collect all proposed enhancements and identify unique ones (using cleaned, lowercased title as key)
    for persona_key, proposals_markdown in proposer_outputs_by_persona.items():
        parsed_proposals = parse_proposals_markdown(proposals_markdown) # Use imported parser
        persona_name = proposer_personas.get(persona_key, {}).get("name", persona_key) # Use passed personas
        for proposal in parsed_proposals:
            original_title = proposal.get("title", f"Untitled Proposal from {persona_name}").strip()
            # Clean and lowercase title for robust matching
            cleaned_lower_title = re.sub(r'^\s*Proposed Enhancement:\s*', '', original_title, flags=re.IGNORECASE).strip().lower()
            cleaned_lower_title = re.sub(r'\*\*:?$', '', cleaned_lower_title).strip()

            if cleaned_lower_title and cleaned_lower_title not in all_proposals_dict: # Ensure title is not empty
                 # Store using the cleaned, lowercased title as key
                 all_proposals_dict[cleaned_lower_title] = proposal
                 all_proposals_dict[cleaned_lower_title]["sourcePersonas"] = [persona_name]
                 # Also store the original title mapping
                 original_titles_map[cleaned_lower_title] = original_title
            elif cleaned_lower_title: # If exists and is not new, add source persona if not already present
                 source_name = proposer_personas.get(persona_key, {}).get("name", persona_key)
                 if source_name not in all_proposals_dict[cleaned_lower_title].get("sourcePersonas", []):
                      all_proposals_dict[cleaned_lower_title].setdefault("sourcePersonas", []).append(source_name)
            # else: Skip proposals with empty titles after cleaning

    # --- DEBUG PRINT ---
    print("\n--- Debug: Aggregation - Collected Unique Proposal Titles (Cleaned, Lowercased) ---")
    if all_proposals_dict:
        for title in all_proposals_dict.keys():
            print(f"  - Unique Proposal Key: '{title}' (Original: '{original_titles_map.get(title, 'N/A')}')")
    else:
        print("  - No unique proposals collected.")
    print("--------------------------------------------------------------------")
    # --- END DEBUG PRINT ---


    # 2. Collect all validation results (including scores) for each proposal, grouped by cleaned, lowercased proposal title
    validations_by_proposal_title = defaultdict(list) # {cleaned_lower_title: [list of validation_data]}

    for validator_key, validations_markdown in validator_outputs_by_name.items():
        validator_name = validator_personas.get(validator_key, {}).get("name", validator_key) # Use passed personas
        if validations_markdown and not validations_markdown.startswith("No proposals were generated") and not validations_markdown.startswith("No proposals were generated or summarized"):
            parsed_validations = parse_validator_scores_and_justifications(validations_markdown, validator_key) # Use imported parser, pass key
            for validation in parsed_validations:
                validation_proposal_title = validation.get("proposalTitle", "").strip()
                 # Clean and lowercase validation title the same way as proposer titles for matching
                cleaned_lower_validation_title = re.sub(r'^\s*Proposed Enhancement:\s*', '', validation_proposal_title, flags=re.IGNORECASE).strip().lower()
                cleaned_lower_validation_title = re.sub(r'\*\*:?$', '', cleaned_lower_validation_title).strip()


                # Find the matching title in our collected proposals dictionary keys (cleaned, lowercased)
                if cleaned_lower_validation_title in all_proposals_dict:
                    # Store the validation data using the cleaned, lowercased title as the key
                    # Keep the original title parsed by the validator in the validation data itself
                    validations_by_proposal_title[cleaned_lower_validation_title].append(validation)
                    # --- DEBUG PRINT ---
                    print(f"  - Matched validation for cleaned key '{cleaned_lower_validation_title}' (Original Validator Title: '{validation_proposal_title}') from '{validator_name}'. Score: {validation.get('rawScore')}")
                    # --- END DEBUG PRINT ---
                elif cleaned_lower_validation_title: # Only warn if the title wasn't empty after cleaning
                    # --- DEBUG PRINT ---
                    print(f"--- Warning: Validation from '{validator_name}' found for unknown proposal title (cleaned): '{cleaned_lower_validation_title}' (Original: '{validation_proposal_title}'). Skipping. ---")
                    # --- END DEBUG PRINT ---


    # 3. Calculate mean scores and determine final verdict for each unique proposal (using cleaned, lowercased keys)
    final_proposals_list = []
    num_validators_configured = len(validator_personas) # Use passed personas

    for cleaned_lower_title, proposal_details in all_proposals_dict.items():
        # Get validations using the cleaned, lowercased key
        validations_for_this_proposal = validations_by_proposal_title.get(cleaned_lower_title, [])

        # Filter for valid numerical scores (normalized 0-1)
        scores_normalized = [v['score'] for v in validations_for_this_proposal if v.get('score') is not None]

        # --- DEBUG PRINT ---
        print(f"\n--- Debug: Calculating Mean for Proposal (Cleaned Key) '{cleaned_lower_title}' ---")
        print(f"  - Collected Normalized Scores ({len(scores_normalized)}): {scores_normalized}")
        # --- END DEBUG PRINT ---

        mean_score_normalized = statistics.mean(scores_normalized) if scores_normalized else 0.0
        mean_score_hundred = round(mean_score_normalized * 100.0, 2)

        num_scores_received = len(scores_normalized)

        # Determine final verdict based on mean score > 90
        final_verdict = "Rejected" # Default
        final_justification = f"Mean score ({mean_score_hundred:.2f}/100) is below the threshold (90/100)."

        if mean_score_hundred > 90.0 and num_scores_received > 0:
            final_verdict = "Approved"
            final_justification = f"Mean score ({mean_score_hundred:.2f}/100) meets or exceeds the threshold (90/100)."
        elif num_scores_received == 0 and num_validators_configured > 0:
            final_verdict = "Undetermined"
            final_justification = f"No valid scores received from any validator out of {num_validators_configured} configured validators."
        elif mean_score_hundred <= 90.0 and num_scores_received > 0:
             # Prioritize rejection justification based on validator type if available and score is low (<= 90)
             low_scoring_validations = sorted([v for v in validations_for_this_proposal if v.get('rawScore', 100.0) <= 90.0 and v.get('justification')],
                                               key=lambda x: x.get('rawScore', 100.0))

             if low_scoring_validations:
                 main_concern_val = low_scoring_validations[0]
                 # Ensure the validator name is correctly pulled from the validation data itself
                 final_justification = f"Rejected due to {main_concern_val.get('validatingValidator', 'Validator')} concern: {main_concern_val['justification']}"
             else:
                  general_reasons = [v.get("justification", "No justification provided") for v in validations_for_this_proposal if v.get("justification")]
                  final_justification = f"Rejected due to low mean score ({mean_score_hundred:.2f}/100). " + (f"Reasons from validators: {'; '.join(general_reasons)}" if general_reasons else "No specific reasons provided by validators.")

        # Use the original title from the map for the final output structure
        original_title = original_titles_map.get(cleaned_lower_title, cleaned_lower_title) # Fallback to cleaned if original not found (shouldn't happen)

        final_proposal_data = {
            **proposal_details, # Include original description, rationale, etc. based on the first time it was parsed
            "title": original_title, # Use the original casing title
            "sourcePersonas": proposal_details.get("sourcePersonas", []),
            "individualValidations": validations_for_this_proposal,
            "meanScore": mean_score_hundred,
            "numScoresReceived": num_scores_received,
            "totalValidatorsConfigured": num_validators_configured,
            "finalVerdict": final_verdict,
            "finalVerdictJustification": final_justification
        }
        final_proposals_list.append(final_proposal_data)

    # Sort proposals (optional, e.g., by mean score descending)
    final_proposals_list.sort(key=lambda x: (x.get('meanScore') is not None, x.get('meanScore', -1)), reverse=True)

    return final_proposals_list