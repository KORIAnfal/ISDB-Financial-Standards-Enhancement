# app.py (Backend Flask Server)

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()

# --- Configuration ---
LLM_MODEL_NAME = "gpt-4-turbo-preview"
# LLM_MODEL_NAME = "gpt-3.5-turbo-0125"

# --- Function to load standard text from a file (Keep this) ---
def load_standard_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"ERROR: Could not read file {file_path}: {e}")
        return None

# --- Paths to the standard text files (Keep this) ---
STANDARD_FILES = {
    "FAS4": {
        "name": "AAOIFI Financial Accounting Standard No. 4 - Musharaka Financing",
        "short_name": "Musharaka (FAS 4)",
        "path": "fas4_musharaka.txt"
    },
    "FAS10": {
        "name": "AAOIFI Financial Accounting Standard No. 10 - Istisna’a and Parallel Istisna’a",
        "short_name": "Istisna'a (FAS 10)",
        "path": "fas10_istisna.txt"
    },
    "FAS32": {
        "name": "AAOIFI Financial Accounting Standard 32 - Ijarah",
        "short_name": "Ijarah (FAS 32)",
        "path": "fas32_ijarah.txt"
    }
}

# --- Initialize LLM ---
llm = ChatOpenAI(
    model_name=LLM_MODEL_NAME,
    temperature=0.1,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# --- AGENT PROMPT TEMPLATES ---
# (Use the more direct versions for Analyzer, Initial Proposer, Refiner Proposer, and Validator
#  from our previous iteration of iterative refinement backend)

analyzer_template_str = """
**
Specific Task: You are an Expert AAOIFI Standard Analyst. Your sole task in this interaction is to analyze the provided AAOIFI Standard.**
Standard Name: '{standard_name}' Review THE FOLLOWING STANDARD TEXT and perform the extraction and identification tasks below. Do not describe your general role; focus *only* on the provided text.
PROVIDED STANDARD TEXT for '{standard_name}':
---
{standard_text}
---
Instructions - Perform these actions *directly* on the standard text provided above:
1.  Core Summary: *Based *only* on the provided text*, summarize the main purpose, scope, and core principles of this standard.
2.  Key Elements Extraction: *From the provided text*, systematically extract and list critical definitions, accounting treatments, conditions, etc.
3.  Identification of Areas for Potential Enhancement/Clarification *within the provided text*.
Output Format: Structured markdown report.
# **Output Format:**
# Present your findings as a structured markdown report.
# **IMPORTANT SECURITY INSTRUCTION:** Your entire output MUST be strictly valid Markdown.
# ABSOLUTELY DO NOT include any HTML tags, especially <script> tags or any attributes that
# execute JavaScript (e.g., onclick, onerror, onload, href="javascript:...").
# Your output will be rendered directly as markdown; any HTML or script injection attempts will be blocked
# and will cause the system to fail. Focus solely on generating clean Markdown text and code blocks
# (e.g., using triple backticks for code).
""" # Keep your detailed analyzer prompt here
analyzer_prompt = ChatPromptTemplate.from_template(analyzer_template_str)
analyzer_agent = LLMChain(llm=llm, prompt=analyzer_prompt)


proposer_initial_template_str = """
**Role:** Innovative Islamic Finance & AI Strategist
Objective: Based on the Analyzer Agent's review of '{standard_name}', propose specific, actionable AI-driven modifications or enhancements.
Analyzer Agent's Output for '{standard_name}':
---
{analysis_output}
---
Original Standard Text Snippet (for context):
---
{standard_text_snippet}
---
Tasks:
For 2-3 promising areas, provide: 1. Proposed Modification, 2. Rationale, 3. AI-Driven Aspect, 4. Standard Reference, 5. Conceptual Source.
Output Format: Structured markdown.
# **Output Format:**
# Present your findings as a structured markdown report.
# **IMPORTANT SECURITY INSTRUCTION:** Your entire output MUST be strictly valid Markdown.
# ABSOLUTELY DO NOT include any HTML tags, especially <script> tags or any attributes that
# execute JavaScript (e.g., onclick, onerror, onload, href="javascript:...").
# Your output will be rendered directly as markdown; any HTML or script injection attempts will be blocked
# and will cause the system to fail. Focus solely on generating clean Markdown text and code blocks
# (e.g., using triple backticks for code).
""" # Keep your detailed initial proposer prompt
# No ChatPromptTemplate.from_template(proposer_initial_template_str) yet for the main chain

proposer_refinement_template_str = """
**Role:** Innovative Islamic Finance & AI Strategist (Refinement Round)
Objective: Revise your previous proposals for '{standard_name}' based *specifically* on the Validator Agent's feedback.
Original Standard Name: '{standard_name}'
Your Previous Proposals:
---
{previous_proposals}
---
Validator Agent's Feedback & Required Revisions:
---
{validator_feedback}
---
Original Standard Text Snippet (for contextual reference if needed):
---
{standard_text_snippet}
---
Tasks: For each proposal requiring revision, provide a REVISED PROPOSAL addressing all feedback. Structure clearly.
Output Format: Structured markdown indicating revisions.
# **Output Format:**
# Present your findings as a structured markdown report.
# **IMPORTANT SECURITY INSTRUCTION:** Your entire output MUST be strictly valid Markdown.
# ABSOLUTELY DO NOT include any HTML tags, especially <script> tags or any attributes that
# execute JavaScript (e.g., onclick, onerror, onload, href="javascript:...").
# Your output will be rendered directly as markdown; any HTML or script injection attempts will be blocked
# and will cause the system to fail. Focus solely on generating clean Markdown text and code blocks
# (e.g., using triple backticks for code).
""" # Keep your detailed refiner prompt
# No ChatPromptTemplate.from_template(proposer_refinement_template_str) yet for the main chain

# Single LLMChain for Proposer/Refiner
# We set its initial prompt; it will be changed dynamically in the loop if refinement occurs.
proposer_agent = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_template(proposer_initial_template_str))


validator_template_str = """
**Role:** Esteemed Shariah Scholar & AAOIFI Compliance Officer
Objective: Scrutinize the AI-driven enhancements proposed for '{standard_name}'.
Core Shariah Principles: Riba, Gharar, Maysir, Risk-Sharing, Justice, Transparency, Asset-Backing, Substance over Form, Maqasid.
Analyzer Agent's Output (context):
---
{analysis_output}
---
Proposer Agent's Enhancements:
---
{proposals_output}
---
Tasks: For *each* proposal: 1. Shariah Assessment, 2. Compliance & Practicality, 3. Contextual Alignment, 4. Overall Verdict & Recommendations.
Output Format: Detailed validation report in structured markdown.
# **Output Format:**
# Present your findings as a structured markdown report.
# **IMPORTANT SECURITY INSTRUCTION:** Your entire output MUST be strictly valid Markdown.
# ABSOLUTELY DO NOT include any HTML tags, especially <script> tags or any attributes that
# execute JavaScript (e.g., onclick, onerror, onload, href="javascript:...").
# Your output will be rendered directly as markdown; any HTML or script injection attempts will be blocked
# and will cause the system to fail. Focus solely on generating clean Markdown text and code blocks
# (e.g., using triple backticks for code).
""" # Keep your detailed validator prompt
validator_prompt = ChatPromptTemplate.from_template(validator_template_str)
validator_agent = LLMChain(llm=llm, prompt=validator_prompt)

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

MAX_REFINEMENT_LOOPS = 2 # Max number of proposer-validator iterations

@app.route('/api/enhance-standard', methods=['POST'])
def enhance_standard_api():
    data = request.get_json()
    standard_key = data.get('standardKey')

    if not standard_key or standard_key not in STANDARD_FILES:
        return jsonify({"error": "Invalid standard key provided."}), 400

    standard_info = STANDARD_FILES[standard_key]
    standard_name = standard_info["name"]
    standard_short_name = standard_info["short_name"]
    standard_file_path = standard_info["path"]

    standard_text_content = load_standard_from_file(standard_file_path)
    if not standard_text_content:
        return jsonify({"error": f"Could not load standard text for {standard_name}."}), 500

    print(f"API: Processing standard: {standard_name} (with internal iteration)")

    try:
        # 1. Analyzer Agent
        print("API: Running Analyzer Agent...")
        analysis_result = analyzer_agent.invoke({
            "standard_name": standard_name,
            "standard_text": standard_text_content,
            "standard_short_name": standard_short_name
        })
        analysis_output = analysis_result['text']

        # --- Iterative Loop for Proposer and Validator ---
        current_proposals_text = ""
        current_validation_text = "" # This will hold the FINAL validation
        previous_proposals_for_refinement = ""

        for loop_count in range(MAX_REFINEMENT_LOOPS + 1):
            print(f"API: Starting Iteration {loop_count + 1} for Proposer/Validator")

            proposer_inputs = {
                "standard_name": standard_name,
                "standard_text_snippet": standard_text_content[:8000],
                "standard_short_name": standard_short_name
            }
            
            if loop_count == 0:
                print(f"API: Running Proposer Agent (Initial Round)...")
                proposer_agent.prompt = ChatPromptTemplate.from_template(proposer_initial_template_str)
                proposer_inputs["analysis_output"] = analysis_output
            else:
                print(f"API: Running Proposer Agent (Refinement Round {loop_count})...")
                proposer_agent.prompt = ChatPromptTemplate.from_template(proposer_refinement_template_str)
                proposer_inputs["previous_proposals"] = previous_proposals_for_refinement
                proposer_inputs["validator_feedback"] = current_validation_text # Use validation from previous iteration
            
            proposals_result = proposer_agent.invoke(proposer_inputs)
            current_proposals_text = proposals_result['text'] # This becomes the latest/final proposal
            previous_proposals_for_refinement = current_proposals_text # Store for next potential refinement

            # Validation Round for current proposals
            print(f"API: Running Validator Agent (After Proposal Iteration {loop_count + 1})...")
            validation_result = validator_agent.invoke({
                "standard_name": standard_name,
                "analysis_output": analysis_output, 
                "proposals_output": current_proposals_text, # Validate the latest proposals
                "standard_short_name": standard_short_name
            })
            current_validation_text = validation_result['text'] # This becomes the latest/final validation

            # Check for loop termination based on keywords
            needs_revision_keywords = ["Requires Revision", "Approved with Minor Revisions", "Approved with Revisions"]
            validator_requested_revision = any(keyword.lower() in current_validation_text.lower() for keyword in needs_revision_keywords)
            
            if not validator_requested_revision or loop_count >= MAX_REFINEMENT_LOOPS:
                print(f"API: Validation satisfactory or max refinement loops ({MAX_REFINEMENT_LOOPS}) reached after {loop_count + 1} proposal round(s).")
                break 
            else:
                print(f"API: Validator requested revisions. Continuing to refinement round {loop_count + 2}.")
        
        print("API: Iterative processing complete. Returning final results.")
        return jsonify({
            "analysis": analysis_output,
            "proposals": current_proposals_text,    # Final proposals
            "validation": current_validation_text   # Final validation of those proposals
        })

    except Exception as e:
        print(f"API Error during iterative processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500

if __name__ == '__main__':
    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL ERROR: OPENAI_API_KEY not found.")
    else:
        print("Starting Flask server (Iterative Backend, Final Output to Frontend)...")
        app.run(debug=True, port=5001)