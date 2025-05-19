# --- START OF FILE app.py (Full Code) ---

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
import time
# --- IMPORT TYPING FOR DUMMY CLASS ---
from typing import Dict, List, Tuple
# --- END IMPORT TYPING ---

# Import specific Groq/OpenAI API error for handling
try:
    # Attempt to import Groq RateLimitError
    from groq import RateLimitError as GroqRateLimitError
    # langchain_openai often maps Groq errors to OpenAI's RateLimitError
    from openai import RateLimitError as OpenAIRateLimitError
    # Define a tuple to catch either specific error
    RateLimitError = (GroqRateLimitError, OpenAIRateLimitError)
    print("Info: Imported specific RateLimitError types for retry logic.")
except ImportError:
    # Fallback to generic exception if Groq or OpenAI client isn't fully installed
    print("Warning: Could not import specific RateLimitError types (groq/openai). Using generic Exception for retry logic.")
    RateLimitError = Exception # Catch all exceptions for retry


load_dotenv() # <<<< CRITICAL: Load environment variables FIRST

# Import necessary components from other files *after* dotenv is loaded
from langchain_openai import ChatOpenAI # LLM initialization
from utils import load_standard_from_file # Utility for file loading
from analyser import run_analyzer_agent # Function to run the analyzer
from proposers import run_proposer_agents, PROPOSER_PERSONAS # Function to run proposers and get outputs/parsed proposals, import Personas for config
from summarizer import run_summarizer_agent # Function to run the summarizer
from validators import run_validator_agents, VALIDATOR_PERSONAS # Function to run validators and get outputs, import Personas for config
from aggregation import aggregate_scores_and_verdicts # Function to aggregate results (requires personas)

# Import classifier after environment variables are loaded
try:
    # --- CORRECTED IMPORT PATH ASSUMING classifier.py IS IN THE SAME DIRECTORY ---
    from classifier import StandardsClassifier
    # --- END CORRECTED IMPORT PATH ---
except ImportError as e:
    print(f"ERROR: Could not import StandardsClassifier from 'classifier': {e}. Make sure classifier.py exists in the same directory as app.py, or adjust the import path.")
    # --- Dummy class for app to run without full classifier logic (executed if import fails) ---
    # Note: This dummy class uses the List and Tuple type hints imported at the top of this file.
    class StandardsClassifier:
        def __init__(self, standards_file: str, translations_file: str, openai_api_key: str = None):
            print("Warning: Using DUMMY StandardsClassifier due to import error.")
            # Load standards data needed for dummy classification logic and explanation info
            self.standards = {}
            try:
                 # Assuming standards_file path is relative to app.py
                 base_dir = os.path.dirname(os.path.abspath(__file__))
                 full_standards_path = os.path.join(base_dir, standards_file)
                 with open(full_standards_path, 'r', encoding='utf-8') as f:
                     self.standards = json.load(f)
                 if not self.standards:
                     print(f"Warning: Dummy classifier could not load standards from {full_standards_path}. It will not classify effectively.")
            except Exception as e:
                 print(f"Warning: Dummy classifier failed to load standards from {standards_file}: {e}")

            # Load translations data needed for dummy explanation info
            self.translations = {}
            try:
                 # Assuming translations_file path is relative to app.py
                 base_dir = os.path.dirname(os.path.abspath(__file__))
                 full_translations_path = os.path.join(base_dir, translations_file)
                 with open(full_translations_path, 'r', encoding='utf-8') as f:
                     self.translations = json.load(f)
            except Exception as e:
                 print(f"Warning: Dummy classifier failed to load translations from {translations_file}: {e}")


            # Define keyword patterns for dummy classification
            # Mapping standard ID to a list of patterns (case-insensitive, word boundaries where appropriate)
            # Using specific terms from your standards and general indicators
            self.dummy_patterns = {
                "FAS4": [
                    re.compile(r'\bFAS\s*4\b', re.IGNORECASE),
                    re.compile(r'\bMusharaka\b', re.IGNORECASE),
                    re.compile(r'\bpartnership\b', re.IGNORECASE),
                    re.compile(r'\bjoint\s*venture\b', re.IGNORECASE),
                    re.compile(r'\bMudarabah\b', re.IGNORECASE), # Often discussed alongside Musharaka
                ],
                "FAS7": [
                    re.compile(r'\bFAS\s*7\b', re.IGNORECASE),
                    re.compile(r'\bSalam\b', re.IGNORECASE),
                    re.compile(r'\bforward\s*sale\b', re.IGNORECASE),
                    re.compile(r'\bdeferred\s*delivery\b', re.IGNORECASE),
                    re.compile(r'\bupfront\s*payment\b', re.IGNORECASE),
                ],
                "FAS10": [
                    re.compile(r'\bFAS\s*10\b', re.IGNORECASE),
                    re.compile(r'\bIstisna\'a\b', re.IGNORECASE),
                    re.compile(r'\bconstruction\b', re.IGNORECASE),
                    re.compile(r'\bmanufacturing\b', re.IGNORECASE),
                    re.compile(r'\bprogress\s*payments\b', re.IGNORECASE),
                ],
                "FAS28": [ # Mapping to Real Estate based on AGENT_STANDARD_FILES_MAP
                    re.compile(r'\bFAS\s*28\b', re.IGNORECASE),
                    re.compile(r'\bReal\s*Estate\b', re.IGNORECASE),
                    re.compile(r'\bproperty\b', re.IGNORECASE),
                    re.compile(r'\binvestment\s*property\b', re.IGNORECASE),
                    # Add Murabaha patterns IF FAS28 in your standards.json *is* Murabaha
                    # re.compile(r'\bMurabaha\b', re.IGNORECASE),
                    # re.compile(r'\bcost-plus\b', re.IGNORECASE),
                ],
                "FAS32": [
                    re.compile(r'\bFAS\s*32\b', re.IGNORECASE),
                    re.compile(r'\bIjarah\b', re.IGNORECASE),
                    re.compile(r'\bLeasing\b', re.IGNORECASE),
                    re.compile(r'\bright-of-use\b', re.IGNORECASE),
                    re.compile(r'\brental\b', re.IGNORECASE),
                ],
                # Removed dummy patterns for FAS11 and FAS30
            }


        # Dummy method that classifies based on keyword counts
        # Uses List and Tuple type hints imported at the top of this file
        def classify_transaction(self, text: str, **kwargs) -> List[Tuple[str, float]]:
             print(f"Dummy classify: {text[:200]}..."); # Print snippet
             if not text or not self.dummy_patterns:
                 return []

             text_lower = text.lower()
             match_counts = defaultdict(int)
             # We don't need to store matched keywords in the dummy anymore if not used
             # matched_keywords = defaultdict(set) # To store unique keywords matched per standard

             for std_id, patterns in self.dummy_patterns.items():
                 for pattern in patterns:
                     # Use findall to count occurrences
                     found_matches = pattern.findall(text_lower)
                     if found_matches:
                         match_counts[std_id] += len(found_matches) # Count total occurrences


             # Convert counts to scores (simple normalization by max count)
             max_count = max(match_counts.values()) if match_counts else 0
             if max_count == 0:
                 scores = {}
             else:
                 scores = {std_id: count / max_count for std_id, count in match_counts.items()}

             # Filter out very low scores (e.g., score < 0.05)
             filtered_scores = {std_id: score for std_id, score in scores.items() if score >= 0.05}


             # Print debug information
             print(f"Dummy classifier matched counts: {dict(match_counts)}")
             print(f"Dummy classifier calculated scores (normalized): {dict(scores)}")
             print(f"Dummy classifier filtered scores (>=0.05): {dict(filtered_scores)}")

             # Return sorted list of (standard_id, score) tuples
             return sorted(filtered_scores.items(), key=lambda item: item[1], reverse=True)

        # Dummy method that provides explanations (optional, can be basic)
        # Uses List, Tuple, and Dict type hints imported at the top of this file
        def explain_classification(self, text: str, results: List[Tuple[str, float]], **kwargs) -> Dict[str, str]:
             print(f"Dummy explain for text: {text[:100]}... with results: {results}");
             explanations_dict = {}

             language = self.detect_language(text) # Use the dummy language detection

             for std_id, score in results:
                 # Get standard name from loaded standards data
                 std_info = self.standards.get(std_id, {})
                 std_name_en = std_info.get("name", f"Standard {std_id}")

                 # Try to get translated name
                 lang_code = language # Use detected language
                 std_translation = self.translations.get(lang_code, {}).get('standards', {}).get(std_id, {})
                 std_name_display = std_translation.get('name', std_name_en)
                 std_desc_display = std_translation.get('description', std_info.get("description", "No description available."))

                 # Get matched keywords for this standard if available (re-run pattern check in dummy explain)
                 # This is less accurate than storing them in classify_transaction, but simpler for a dummy
                 matched_terms_in_text = []
                 patterns = self.dummy_patterns.get(std_id, [])
                 text_check = text.lower() if language == 'en' else text
                 for p in patterns:
                      found_matches = p.findall(text_check) # Find all occurrences
                      if found_matches:
                          # Add the pattern string itself (cleaned) as the matched term
                          term = p.pattern.strip(r'\b').strip('\\').strip().replace('?', '').replace('*', '') # Basic cleaning
                          if term: matched_terms_in_text.append(term)


                 matched_terms_in_text = list(set(matched_terms_in_text)) # Get unique terms


                 if language == 'ar':
                    explanation_html = f"<h4>{std_name_display} ({std_id}) - الصلة (تقريبية): {score*100:.0f}%</h4>"
                    explanation_html += f"<p><strong>الوصف العام للمعييار:</strong> {std_desc_display}</p>"
                    if matched_terms_in_text:
                         explanation_html += f"<p><strong>الكلمات/العبارات المطابقة (تقريبية):</strong> {', '.join(matched_terms_in_text)}</p>"
                    explanation_html += f"<p><strong>تحليل تقريبي:</strong> تم اعتبار هذا المعيار ملائمًا بناءً على الكلمات الرئيسية الموجودة في النص.</p>"
                 else: # English
                    explanation_html = f"<h4>{std_name_display} ({std_id}) - Relevance (Approx): {score*100:.0f}%</h4>"
                    explanation_html += f"<p><strong>Standard Overview:</strong> {std_desc_display}</p>"
                    if matched_terms_in_text:
                         explanation_html += f"<p><strong>Matched Keywords/Phrases (Approx):</strong> {', '.join(matched_terms_in_text)}</p>"
                    explanation_html += f"<p><strong>Approximate Analysis:</strong> This standard was deemed relevant based on keywords found in the text.</p>"

                 explanations_dict[std_id] = explanation_html


             print(f"Dummy explanations generated for {len(explanations_dict)} standards.")
             return explanations_dict

        # Keep the dummy detect_language method
        def detect_language(self, text: str) -> str:
             if not text or not isinstance(text, str): return 'en'
             arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
             return 'ar' if arabic_chars / max(len(text), 1) > 0.05 else 'en'


    # --- END Dummy class ---
    # raise # Uncomment this line if you want the app to fail when classifier.py is not found


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # For development, adjust for production
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_default_flask_secret_key_for_development")

# --- Configuration & Initialization ---
# Get API keys AFTER load_dotenv() has run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Used by the classifier
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Used by the multi-agent LLMs

if not OPENAI_API_KEY:
    print("CRITICAL Warning: OPENAI_API_KEY environment variable is not set or loaded. OpenAI calls in classifier will likely fail.")
else:
    print(f"app.py: OpenAI API Key found (masked: sk-...{OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 4 else 'SHORT'}).")

if not GROQ_API_KEY:
     print("CRITICAL Warning: GROQ_API_KEY environment variable is not set or loaded. Groq calls for multi-agent will likely fail.")
else:
     print(f"app.py: Groq API Key found (masked: sk-...{GROQ_API_KEY[-4:] if len(GROQ_API_KEY) > 4 else 'SHORT'}).")


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STANDARDS_FILE_CLASSIFIER = os.path.join(DATA_DIR, "standards.json") # Classifier's standards file
TRANSLATIONS_FILE = os.path.join(DATA_DIR, "translations.json") # Classifier's translations file
EXAMPLES_FILE = os.path.join(DATA_DIR, "examples.json") # Classifier's examples file

# --- Paths to the standard text files for Agents ---
# This maps standard IDs (keys used by classifier and agents) to their details
# and file paths for loading the full standard text.
# **MODIFIED based on provided standards directory image**
AGENT_STANDARD_FILES_MAP = {
    "FAS4": {
        "name": "AAOIFI Financial Accounting Standard No. 4 - Musharaka Financing",
        "short_name": "Musharaka (FAS 4)",
        "path": "standards/fas4_musharaka.txt" # Assuming standards are in a 'standards' subdirectory relative to app.py
    },
    "FAS10": {
        "name": "AAOIFI Financial Accounting Standard No. 10 - Istisna’a and Parallel Istisna’a",
        "short_name": "Istisna'a (FAS 10)",
        "path": "standards/fas10_istisna.txt"
    },
    "FAS32": {
        "name": "AAOIFI Financial Accounting Standard 32 - Ijarah",
        "short_name": "Ijarah (FAS 32)",
        "path": "standards/fas32_ijarah.txt"
    },
    "FAS7": {
        "name": "AAOIFI Financial Accounting Standard No. 7 - Salam and Parallel Salam",
        "short_name": "Salam (FAS 7)",
        "path": "standards/fas7_salam.txt"
    },
    "FAS28": {
        "name": "AAOIFI Financial Accounting Standard No. 28 - Investments in Real Estate", # **NOTE**: If FAS28 in your standards.json is Murabaha, this mapping needs adjustment or clarification.
        "short_name": "Real Estate (FAS 28)",
        "path": "standards/fas28_realestate.txt"
    },
    # Removed FAS11 and FAS30 entries as per the provided standards directory image
}

# --- Multi-Agent LLM Configuration ---
# Set the model name to a Groq model name.
LLM_MODEL_NAME = "llama3-70b-8192" # Using the larger Llama 3 model (32k context)

# Groq API Base URL (OpenAI compatible endpoint)
GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Retry settings for rate limits
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1.0 # seconds

# Max characters for standard text snippet passed to agents (helps control input size)
MAX_SNIPPET_CHARS = 4000


# --- Initialize LLM Clients for Agents ---
# Initialize ChatOpenAI instances pointing to Groq API using the Groq key
llm_creative = ChatOpenAI(
    model_name=LLM_MODEL_NAME,
    temperature=0.4,
    openai_api_key=GROQ_API_KEY, # Use Groq key for these
    base_url=GROQ_API_BASE
)

llm_structured = ChatOpenAI(
    model_name=LLM_MODEL_NAME,
    temperature=0.1,
    openai_api_key=GROQ_API_KEY, # Use Groq key for these
    base_url=GROQ_API_BASE
)

# Instantiate classifier, passing the API key loaded by app.py
# The classifier uses the OPENAI_API_KEY for its internal AI calls
classifier = StandardsClassifier(
    standards_file=STANDARDS_FILE_CLASSIFIER,
    translations_file=TRANSLATIONS_FILE,
    # fine_tuned_model_id is removed as it's not used in the new classifier
    openai_api_key=OPENAI_API_KEY # Pass the OpenAI key here for the classifier's client
)

# Load example data for the /api/examples endpoint
try:
    with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
        examples_data_list = json.load(f)
except Exception as e:
    print(f"Error loading examples file ({EXAMPLES_FILE}): {e}. Examples list will be empty.")
    examples_data_list = []

# Load translations data for the /api/translations endpoint and classifier explanations/prompts
try:
    with open(TRANSLATIONS_FILE, "r", encoding="utf-8") as f:
        translations_data_all_langs = json.load(f)
except Exception as e:
    print(f"Error loading translations file ({TRANSLATIONS_FILE}): {e}. Translations functionality will be impacted.")
    translations_data_all_langs = {
        "en": {"error_loading_translations": "Critical: Main translations file could not be loaded."},
        "ar": {"error_loading_translations": "خطأ جسيم: لم يتم تحميل ملف الترجمة الرئيسي."}
    }

# --- Helper function to run the multi-agent process for a *single* standard ---
def run_multi_agent_process_for_standard(standard_key: str, news_text: str):
    """
    Orchestrates the multi-agent process (Analysis, Proposing, Summarizing, Validating, Aggregation)
    for a single specified standard and news text.
    Returns analysis output and final proposals list, or None/empty on failure for this standard.
    """
    standard_info = AGENT_STANDARD_FILES_MAP.get(standard_key)
    if not standard_info:
        print(f"Error: Standard key '{standard_key}' not configured in AGENT_STANDARD_FILES_MAP for agent processing.")
        return None, [] # Return empty list for proposals on failure

    standard_name = standard_info["name"]
    standard_short_name = standard_info["short_name"]
    standard_file_path = standard_info["path"]

    standard_text_content = load_standard_from_file(standard_file_path) # Use utility function
    if not standard_text_content:
        print(f"Error: Could not load standard text for {standard_name} from {standard_file_path}. Cannot run agents.")
        return None, [] # Return empty list for proposals on failure

    print(f"\n--- Running Multi-Agent Process for {standard_name} ('{standard_key}') ---")
    print(f"Using News Text (first 200 chars):\n{news_text[:200]}...") # Print snippet of news

    standard_text_snippet = standard_text_content[:MAX_SNIPPET_CHARS]

    try:
        # 1. Run Analyzer Agent (using function from analyser.py)
        analysis_output = run_analyzer_agent(
            llm_structured=llm_structured,
            standard_name=standard_name,
            standard_short_name=standard_short_name,
            standard_text=standard_text_content, # Analyzer gets full text
            finance_news=news_text,
            max_retries=MAX_RETRIES,
            initial_retry_delay=INITIAL_RETRY_DELAY,
            RateLimitError=RateLimitError # Pass the RateLimitError type tuple/Exception
        )
        # Check analysis output - decide if critical enough to stop.
        # If analyzer fails completely, proposer/validator prompts might also fail.
        # For now, let's allow it to continue, agents might still produce something.
        if not analysis_output or analysis_output.strip() == "":
             print(f"Warning: Analyzer returned empty output for {standard_key}.")
             # Pass an empty string or a placeholder analysis output


        # 2. Run Proposer Agents (using function from proposers.py)
        # This function returns raw outputs AND the list of parsed proposals (temp_all_proposals)
        proposer_outputs_by_persona, temp_all_proposals, proposer_personas_used = run_proposer_agents(
             llm_creative=llm_creative,
             standard_name=standard_name,
             standard_short_name=standard_short_name,
             standard_text_snippet=standard_text_snippet, # Proposers get snippet
             analysis_output=analysis_output,
             finance_news=news_text,
             max_retries=MAX_RETRIES,
             initial_retry_delay=INITIAL_RETRY_DELAY,
             RateLimitError=RateLimitError,
             proposer_personas=PROPOSER_PERSONAS # Pass the persona config
        )
        if not temp_all_proposals:
             print(f"Warning: No proposals generated by any proposer for {standard_key}.")
             # Proceed to summarizer/aggregation, which will handle empty lists


        # 3. Run Summarizer Agent (using function from summarizer.py)
        # The summarizer needs the raw outputs to build its simplified input
        all_proposals_summarized_markdown = run_summarizer_agent(
             llm_structured=llm_structured,
             standard_name=standard_name,
             standard_short_name=standard_short_name,
             raw_proposer_outputs_by_persona=proposer_outputs_by_persona,
             max_retries=MAX_RETRIES,
             initial_retry_delay=INITIAL_RETRY_DELAY,
             RateLimitError=RateLimitError
        )
        # Summarizer output will indicate if no proposals were found


        # 4. Run Validator Agents (using function from validators.py)
        # Pass the summarized markdown for validators to assess
        validator_outputs_by_name, validator_personas_used = run_validator_agents(
             llm_structured=llm_structured,
             standard_name=standard_name,
             standard_short_name=standard_short_name,
             analysis_output=analysis_output, # Validators still need analysis context
             all_proposals_summarized_markdown=all_proposals_summarized_markdown, # Pass the summary
             max_retries=MAX_RETRIES,
             initial_retry_delay=INITIAL_RETRY_DELAY,
             RateLimitError=RateLimitError,
             validator_personas=VALIDATOR_PERSONAS # Pass the persona config
        )
        # Note: Validator outputs will contain messages if summarizer found no proposals.
        # Aggregation handles this case.


        # 5. Aggregate Scores and Determine Final Verdicts (using function from aggregation.py)
        # The aggregation logic needs the raw proposer outputs (to parse full details)
        # and the raw validator outputs (to parse scores/justifications).
        final_proposals_list = aggregate_scores_and_verdicts(
            proposer_outputs_by_persona=proposer_outputs_by_persona,
            validator_outputs_by_name=validator_outputs_by_name,
            proposer_personas=PROPOSER_PERSONAS, # Pass persona configs to aggregation
            validator_personas=VALIDATOR_PERSONAS
        )

        print(f"\n--- Multi-Agent Process Complete for {standard_key} ---")
        return analysis_output, final_proposals_list # Return results for this standard

    except Exception as e:
        print(f"\n--- Multi-Agent Error for Standard {standard_key} ---")
        print(f"Error during multi-agent processing for {standard_key}: {str(e)}")
        import traceback
        traceback.print_exc()
        print("-----------------")
        # Return None analysis and empty proposals list for this standard on error
        return None, []


# --- API Endpoints ---

# Endpoint for getting translations (used by frontend for UI text)
@app.route('/api/translations/<language>')
def get_translations_for_lang_route(language):
    lang_code = language.lower()
    lang_data = translations_data_all_langs.get(lang_code)

    if lang_data:
        return jsonify(lang_data)

    # Fallback to English if requested language is not found
    print(f"Warning: Translations for '{lang_code}' not found. Falling back to 'en'.")
    fallback_lang_data = translations_data_all_langs.get('en', {})
    if not fallback_lang_data and lang_code != 'en': # If even 'en' is missing and it wasn't 'en' requested
         return jsonify({"error": f"Translations for '{lang_code}' and fallback 'en' are missing."}), 404
    return jsonify(fallback_lang_data)

# Endpoint for getting examples (used by frontend for examples list)
@app.route('/api/examples')
def get_all_examples_route():
    return jsonify(examples_data_list)

# Endpoint for classifying general text (transaction or news, does NOT run agents)
# This is distinct from /api/analyze-news which triggers the agent workflow
@app.route('/api/analyze', methods=['POST'])
def api_analyze_route():
    """
    Endpoint for classifying transaction or news text using the classifier.
    Does NOT trigger the multi-agent enhancement workflow.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload. Missing 'transaction_text'."}), 400

    text_to_classify = data.get('transaction_text', '').strip() # Renamed for clarity

    if not text_to_classify:
        return jsonify({"error": "Text must be provided for analysis."}), 400

    # Language is optional, default to 'en' and validate/fallback in classifier
    lang_code_input = data.get('language', 'en').lower()

    # The classifier's classify_transaction method now uses only AI prompting and no weights
    try:
        # --- UPDATED CALL TO CLASSIFIER ---
        # The real StandardsClassifier classify_transaction method no longer accepts weights
        classified_scores_tuples = classifier.classify_transaction(
            text_to_classify # Pass only the text
        )
        # --- END UPDATED CALL ---

        # Get explanations based on these scores
        # The explain_classification method still works with the List[Tuple[str, float]] output
        explanations_dict = classifier.explain_classification(text_to_classify, classified_scores_tuples)

    except Exception as e: # Catch broader errors from classifier methods
        print(f"Error during classification or explanation process: {e}")
        return jsonify({"error": f"An internal error occurred during classification: {str(e)}", "details": str(e) if app.debug else None}), 500

    # Load master standards data for names/descriptions for formatting the results
    # Need to load the classifier's standard file again here for display info.
    try:
        with open(STANDARDS_FILE_CLASSIFIER, 'r', encoding="utf-8") as f:
            standards_master_data_classifier = json.load(f)
    except Exception as e:
        print(f"Critical Error loading classifier standards master file ({STANDARDS_FILE_CLASSIFIER}) in api_analyze: {e}")
        standards_master_data_classifier = {}

    # Format results for the frontend
    formatted_results_list = []
    # The classifier results are already sorted
    for std_id, score_value in classified_scores_tuples:
        # Get display info from classifier's master data
        standard_info_from_classifier_master = standards_master_data_classifier.get(std_id, {})

        # Get name/short_name from the agent standards map for consistent keys/names if possible
        standard_info_from_agent_map = AGENT_STANDARD_FILES_MAP.get(std_id, {})

        # Use name from agent map if available, fallback to classifier master data, then generic
        base_name = standard_info_from_agent_map.get("name", standard_info_from_classifier_master.get("name", f"Standard {std_id}"))
        # Use short_name from agent map if available, fallback to classifier master data name, then generic
        base_short_name = standard_info_from_agent_map.get("short_name", standard_info_from_classifier_master.get("name", f"Std {std_id}")) # Use classifier name as fallback for short
        base_description = standard_info_from_classifier_master.get("description", "No description available.")


        # Attempt to get translated name/description for the current display language
        current_lang_code = classifier.detect_language(text_to_classify) # Use classifier's detection if needed, or assume 'en' if language wasn't provided
        display_lang_translations_for_standards = translations_data_all_langs.get(current_lang_code, {}).get('standards', {})
        std_specific_display_translation = display_lang_translations_for_standards.get(std_id, {})

        display_name = std_specific_display_translation.get('name', base_name)
        display_description = std_specific_display_translation.get('description', base_description)


        formatted_results_list.append({
            "standard_id": std_id,
            "name": display_name,
            "short_name": base_short_name,
            "description": display_description, # Include description for UI
            "score": round(score_value * 100, 2), # Assuming score_value is 0-1 from classifier
            "explanation": explanations_dict.get(std_id, "") # Get corresponding explanation HTML
        })

    return jsonify({
        "input_text": text_to_classify, # Echo the input text
        "results": formatted_results_list # This is the list of formatted standard objects
    }), 200


# Endpoint for analyzing news text to identify relevant standards,
# and then running the multi-agent enhancement process for each identified standard.
@app.route('/api/analyze-news', methods=['POST'])
def api_analyze_news_route():
    """
    Endpoint for classifying news text to identify relevant standards using the classifier,
    and then running the multi-agent enhancement process for each identified standard.
    Returns results for all processed standards.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload. Missing 'news_text'."}), 400

    news_text = data.get('news_text', '').strip()

    if not news_text:
        return jsonify({"error": "News text must be provided for analysis."}), 400

    print(f"\n--- Received News Analysis Request ---")
    print(f"News Text (first 200 chars):\n{news_text[:200]}...")

    # 1. Use the classifier to find relevant standards from the news text
    # The classifier's classify_transaction method now uses only AI prompting and no weights
    try:
        # --- UPDATED CALL TO CLASSIFIER ---
        # The real StandardsClassifier classify_transaction method no longer accepts weights
        classified_scores_tuples = classifier.classify_transaction(
            news_text # Pass only the text
        )
        # --- END UPDATED CALL ---

        print(f"\nClassifier identified {len(classified_scores_tuples)} relevant standards from news.")
        # print("Classifier Results for News:", relevant_standards_from_news) # Uncomment for detailed print

    except Exception as e:
        print(f"Error during news classification: {e}")
        return jsonify({"error": f"An error occurred during news classification: {str(e)}"}), 500

    # 2. Determine which of the identified standards to process with the multi-agent system
    # Only process standards that are mapped in AGENT_STANDARD_FILES_MAP
    # and have a score above a certain threshold. Process a limited number (e.g., top 3-5).
    standards_to_process = []
    # Use a threshold consistent with the classifier's output/filtering (e.g., 0.05 or slightly higher)
    processing_threshold = 0.1 # Only process if score is above this threshold (adjust as needed)
    max_standards_to_process = 5 # Limit the number of standards for efficiency (Increased to 5)

    for std_id, score in classified_scores_tuples: # Iterate over the classified results
        if score >= processing_threshold and std_id in AGENT_STANDARD_FILES_MAP: # Use >= threshold and check if it's in our map
            standards_to_process.append((std_id, score))
            if len(standards_to_process) >= max_standards_to_process:
                break # Stop if we reach the limit

    if not standards_to_process:
        print("No relevant standards found above threshold or configured for agent processing.")
        return jsonify({
            "news_text": news_text,
            # Still return all identified standards from classifier for context in the UI
            "identified_standards_scores": classified_scores_tuples, # Return all results from classifier
            "message": "No relevant standards identified for agent processing based on the provided news text (check 'identified_standards_scores').",
            "processed_standards_results": {} # Return empty results
        }), 200

    print(f"\n--- Identified {len(standards_to_process)} standards for Multi-Agent Processing: ---")
    for std_id, score in standards_to_process:
        # Look up standard name using AGENT_STANDARD_FILES_MAP
        std_name = AGENT_STANDARD_FILES_MAP.get(std_id, {}).get('name', std_id)
        print(f"- {std_name} ({std_id}) - Score: {score:.2f}")
    print("--------------------------------------------------------------------")

    # 3. Run the multi-agent process for each identified standard
    processed_standards_results = {}
    overall_errors = [] # To collect errors from individual standard processing

    for std_id, score in standards_to_process:
        try:
            # Call the reusable multi-agent function
            analysis_output, final_proposals_list = run_multi_agent_process_for_standard(std_id, news_text)

            # Get standard info from AGENT_STANDARD_FILES_MAP for display
            standard_info_for_display = AGENT_STANDARD_FILES_MAP.get(std_id, {})
            # Note: standard_info_for_display should always exist here due to the standards_to_process filter, but defensive check is okay


            processed_standards_results[std_id] = {
                "standard_key": std_id,
                "standard_info": standard_info_for_display, # Include standard name/short_name etc.
                "relevance_score": score, # Include the score from news classification
                "analysis": analysis_output, # Analysis for this standard (raw markdown)
                "final_proposals": final_proposals_list # Aggregated proposals for this standard (structured list)
            }

        except Exception as e:
            print(f"CRITICAL ERROR: Uncaught exception during multi-agent process for standard {std_id}: {e}")
            import traceback
            traceback.print_exc()
            overall_errors.append(f"Error processing standard {std_id}: {str(e)}")
            # Get standard info from AGENT_STANDARD_FILES_MAP for display in error case
            standard_info_for_display = AGENT_STANDARD_FILES_MAP.get(std_id, {})
            # Add an entry indicating failure for this standard
            processed_standards_results[std_id] = {
                "standard_key": std_id,
                 "standard_info": standard_info_for_display,
                 "relevance_score": score,
                 "error": f"Failed to complete multi-agent process for this standard: {str(e)}. Check backend logs."
            }


    print("\n--- Overall News Analysis & Agent Processing Complete ---")

    # 4. Return results for all processed standards
    response_payload = {
        "news_text": news_text, # Echo the input news text
        "identified_standards_scores": classified_scores_tuples, # Return all classifier scores for transparency
        "processed_standards_results": processed_standards_results # Results structured by standard ID
    }

    if overall_errors:
        response_payload["overall_errors"] = overall_errors
        # Return 200 even if some standards failed, as long as the overall process finished.
        # A 500 would indicate a failure *before* even starting standard processing loops.
        return jsonify(response_payload), 200
    else:
        return jsonify(response_payload), 200


# Endpoint for running agents on a pre-selected standard (manual selection)
# Keep this endpoint if you still want the manual selection option in your frontend.
@app.route('/api/enhance-standard', methods=['POST'])
def enhance_standard_api():
    """
    Endpoint to run the multi-agent process for a *single* standard manually selected by the user.
    Requires standardKey and userNewsText in the request body.
    """
    data = request.get_json()
    standard_key = data.get('standardKey')
    user_news_text = data.get('userNewsText', '').strip()

    # Use AGENT_STANDARD_FILES_MAP to validate key for agent processing
    if not standard_key or standard_key not in AGENT_STANDARD_FILES_MAP:
        return jsonify({"error": "Invalid standard key provided or standard not configured for agent processing. Available for agents: " + ", ".join(AGENT_STANDARD_FILES_MAP.keys())}), 400

    if not user_news_text:
         return jsonify({"error": "Finance news text must be provided."}), 400

    print(f"\n--- Received Enhance Standard Request (Manual Selection) ---")
    print(f"Standard Selected: {standard_key}")

    # Call the reusable multi-agent process function
    analysis_output, final_proposals_list = run_multi_agent_process_for_standard(standard_key, user_news_text)

    # Check if the process function indicated failure (returned None analysis and empty list)
    if analysis_output is None and not final_proposals_list:
         return jsonify({"error": f"Failed to complete multi-agent process for standard {standard_key}. Check backend logs."}), 500

    # Return the structured results for the single standard
    return jsonify({
        "standard": {
            "key": standard_key,
            "name": AGENT_STANDARD_FILES_MAP[standard_key]["name"],
            "short_name": AGENT_STANDARD_FILES_MAP[standard_key]["short_name"]
        },
        "analysis": analysis_output, # Analysis from the Reviewer (raw markdown)
        "finance_news": user_news_text, # Provided news text (raw markdown)
        "final_proposals": final_proposals_list # <--- Structured list of proposals with aggregated scoring
    }), 200


if __name__ == '__main__':
    # Note about using Groq API key in GROQ_API_KEY environment variable
    # Check for API keys specifically
    if os.getenv("GROQ_API_KEY") and os.getenv("OPENAI_API_KEY"):
        print("\n--- API Configuration Status ---")
        print(f"Groq Model (Agents): {LLM_MODEL_NAME}")
        print(f"Groq API Base URL: {GROQ_API_BASE}")
        print(f"Groq API Key: Loaded (via GROQ_API_KEY env var)")
        print(f"OpenAI API Key (Classifier): Loaded (via OPENAI_API_KEY env var)")
        print(f"Fine-Tuned Model ID: {'Not Set - Using default/base AI in classifier'}") # Updated status
        print("------------------------------\n")
    else:
        print("\nCRITICAL ERROR: One or more API keys (GROQ_API_KEY, OPENAI_API_KEY) not found. Please set them in your .env file or environment.")
        # Consider exiting or raising an error here in production
        # exit()


    # Ensure data and standards directories exist
    data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    standards_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'standards')

    os.makedirs(data_dir_path, exist_ok=True)
    os.makedirs(standards_dir_path, exist_ok=True)
    print(f"Ensured '{data_dir_path}' and '{standards_dir_path}' directories exist.")


    # Check if essential files exist (standards.json, translations.json) for the classifier
    classifier_standards_exists = os.path.exists(STANDARDS_FILE_CLASSIFIER)
    translations_exists = os.path.exists(TRANSLATIONS_FILE)

    if not classifier_standards_exists:
         print(f"CRITICAL WARNING: Classifier standards file '{STANDARDS_FILE_CLASSIFIER}' not found.")
         # The classifier __init__ will also print an error, but this is a heads-up.
    if not translations_exists:
         print(f"WARNING: Translations file '{TRANSLATIONS_FILE}' not found.")
         # The classifier __init__ will also print a warning.


    # Check if agent standard text files exist based on AGENT_STANDARD_FILES_MAP
    missing_agent_standard_files = []
    # Iterate through AGENT_STANDARD_FILES_MAP to check individual text files
    for key, info in AGENT_STANDARD_FILES_MAP.items():
        file_path = info["path"]
        # Use load_standard_from_file from utils to check if it can be loaded
        # Assuming utils.py is in the same directory as app.py
        full_file_path_check = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        if not os.path.exists(full_file_path_check):
             missing_agent_standard_files.append(file_path)
        else:
             # Also check if it's readable/non-empty
             content_check = load_standard_from_file(file_path)
             if not content_check or content_check.strip() == "":
                 print(f"Warning: Agent standard file '{file_path}' found but is empty or unreadable. Multi-agent process for '{key}' will likely fail.")
                 # We don't add to missing_agent_standard_files here because the run_multi_agent_process_for_standard function already handles this.


    if missing_agent_standard_files:
         print(f"\nWARNING: Missing agent standard text files:")
         for f in missing_agent_standard_files:
             print(f"- {f}")
         print(f"Please create these files and add the text for the corresponding AAOIFI standards mapped in AGENT_STANDARD_FILES_MAP.")
         print("Multi-agent process will not run for standards with missing files.")
    else:
        print("\nAll configured agent standard text files found and are readable.")


    print("\n--- Starting Flask server (AAOIFI AI Assistant) ---")
    print("API Endpoints:")
    print("- /api/analyze (POST): For classifying text (can use news or transaction text).")
    print("- /api/analyze-news (POST): For classifying news text and running multi-agent enhancement for relevant standards.")
    print("- /api/enhance-standard (POST): For running multi-agent enhancement on a manually selected standard (Optional endpoint for frontend).")
    print("- /api/translations/<language> (GET): For fetching UI translations.")
    print("- /api/examples (GET): For fetching example transactions.")
    print(f"Server will listen on http://0.0.0.0:5001") # Consolidating to one port

    # Ensure the AGENT_STANDARD_FILES_MAP keys are valid keys in the classifier's standards data if possible,
    # although the system is designed to handle cases where they differ.

    app.run(debug=True, host='0.0.0.0', port=5001) # Run on port 5001

# --- END OF FILE app.py (Full Code) ---