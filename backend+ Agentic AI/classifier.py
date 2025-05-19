# --- START OF FILE classifier.py (New - AI Prompting Only Classification) ---

import json
import re
import os
# --- Import Union for type hint compatibility (Python < 3.10) ---
from typing import Dict, List, Tuple, Union
# --- End Import Union ---
from openai import OpenAI
from dotenv import load_dotenv

# It's recommended to call load_dotenv in your main entry point (app.py)
# but keeping it here allows for standalone testing of this file.
load_dotenv()

# Initialize the global OpenAI client carefully.
# This is a fallback if the class instance isn't given a key.
# In the Flask app, the key is passed to the instance, which is preferred.
global_openai_client = None
_api_key_from_env_for_global = os.getenv("OPENAI_API_KEY")
if _api_key_from_env_for_global:
    try:
        global_openai_client = OpenAI(api_key=_api_key_from_env_for_global)
        print("Classifier Module: Global OpenAI client initialized from environment variable.")
    except Exception as e:
        print(f"Classifier Module Warning: Failed to initialize global OpenAI client from env var: {e}")
        global_openai_client = None # Ensure it's None on failure
else:
    print("Classifier Module Warning: OPENAI_API_KEY not found in environment. Global client not initialized.")


class StandardsClassifier:
    def __init__(self,
                 standards_file: str,
                 translations_file: str,
                 # fine_tuned_model_id is removed as it's not used in this AI-only approach
                 openai_api_key: str = None):
        """
        Load standards and translations, and set up OpenAI client for classification and explanations.
        Relies solely on LLM prompting for classification.
        openai_api_key: If provided, this key will be used for this instance's client.
        """
        self.standards_file_path = standards_file
        self.translations_file_path = translations_file

        # Initialize the OpenAI client using the provided key, or fallback to global
        if openai_api_key:
            try:
                self.client = OpenAI(api_key=openai_api_key)
                print("StandardsClassifier Instance: Initialized with API key provided by caller.")
            except Exception as e:
                print(f"ERROR: StandardsClassifier Instance: Failed to initialize client with provided API key: {e}")
                if global_openai_client:
                    print("StandardsClassifier Instance: Falling back to global client due to error with provided key.")
                    self.client = global_openai_client
                else:
                    raise ValueError(f"Provided OpenAI API key is invalid and no global client available: {e}") from e
        elif global_openai_client:
            self.client = global_openai_client
            print("StandardsClassifier Instance: Using globally initialized OpenAI client (no key provided by caller).")
        else:
            print("CRITICAL ERROR: StandardsClassifier Instance: No API key provided and global client failed to initialize.")
            # Allow initialization but AI calls will fail later
            self.client = None # Set client to None if initialization fails


        # Load Standards data - Needed for prompt content and explanations
        try:
            with open(self.standards_file_path, 'r', encoding='utf-8') as f:
                self.standards = json.load(f)
            if not self.standards:
                print(f"Warning: No standards loaded from {self.standards_file_path}. Classification and explanations will not work.")
        except FileNotFoundError:
            print(f"CRITICAL ERROR: Standards file not found at {self.standards_file_path}")
            self.standards = {}
        except json.JSONDecodeError:
            print(f"CRITICAL ERROR: Could not decode JSON from standards file {self.standards_file_path}")
            self.standards = {}


        # Load Translations data - Needed for prompt and explanations
        self.translations = {}
        try:
            with open(self.translations_file_path, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Translations file not found at {self.translations_file_path}. Translations in prompt/explanations will be empty.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from translations file {self.translations_file_path}. Translations might be incomplete.")

        # We don't need self.all_keywords for keyword matching anymore.
        # We'll just use the standard data directly for the prompt and explanations.
        # However, we can still load keyword lists for potential use in explanations if needed.
        self.all_keywords = {} # Keep this structure
        if self.standards:
             for std_id, std_data in self.standards.items():
                if not isinstance(std_data, dict): continue
                en_kws = [str(kw) for kw in (std_data.get('key_concepts', []) + std_data.get('transaction_indicators', []) + std_data.get('accounting_entries', [])) if kw]
                ar_kws = []
                ar_std_translation = self.translations.get('ar', {}).get('standards', {}).get(std_id, {})
                if isinstance(ar_std_translation, dict):
                     ar_kws = [str(kw) for kw in (ar_std_translation.get('key_concepts', []) + [ar_std_translation.get('name')]) if kw]

                self.all_keywords[std_id] = {
                     'en_keywords_list': en_kws, # Stored for explanations only
                     'ar_keywords_list': ar_kws, # Stored for explanations only
                     # Removed 'en_patterns', 'ar_patterns', 'weight' if not used
                }


    def detect_language(self, text: str) -> str:
        """Basic language detection (Arabic or English)."""
        # Keep this as is
        if not text or not isinstance(text, str): return 'en'
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        return 'ar' if arabic_chars / max(len(text), 1) > 0.05 else 'en' # Lowered threshold slightly


    # --- Removed keyword_match method ---

    # --- Build AI Classification Prompt ---
    def _build_ai_classification_prompt(self, text_to_classify: str, language: str) -> Tuple[str, str]:
        """Builds the prompt for AI classification."""
        if language == 'ar':
            system_message_content = (
                "أنت خبير في معايير المحاسبة المالية (FAS) الصادرة عن هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية (AAOIFI). "
                "مهمتك هي تحليل النص المقدم وتحديد المعايير الأكثر صلة به من القائمة المقدمة."
            )
            # --- IMPROVED PROMPT ---
            prompt_intro = (
                "الرجاء تحليل النص التالي، وهو عبارة عن خبر مالي حول تطورات أو معاملات في التمويل الإسلامي. "
                "حدد معايير المحاسبة المالية (FAS) من القائمة أدناه التي تبدو الأكثر ملاءمةً للموضوع أو من المحتمل أن تتأثر بالاتجاهات أو المعاملات المذكورة في النص. "
                "يجب أن يكون إخراجك كائن JSON يحتوي على معرفات المعايير (مثل 'FAS4', 'FAS28') كمفاتيح ودرجات ملاءمتها (من 0.0 إلى 1.0) كقيم. "
                "قم بتضمين فقط المعايير التي لا تقل درجة ملاءمتها عن 0.01.\n" # Lowered threshold hint
                "إذا كان النص لا يتعلق بأي من المعايير ذات الصلة، أخرج كائن JSON فارغًا {{}}.\n"
                "مثال للإخراج ذي صلة متعددة: {{ \"FAS28\": 0.8, \"FAS32\": 0.5 }}\n"
                "مثال للإخراج ذي صلة واحدة: {{ \"FAS7\": 1.0 }}\n"
                "مثال للإخراج عندما لا يكون هناك صلة كافية: {{}}\n\n"
            )
            # --- END IMPROVED PROMPT ---
            standards_list_header = "قائمة المعايير المحتملة للنظر فيها (تأكد من إخراج معرفات المعايير الصحيحة مثل 'FAS4' وليس فقط '4'):\n"
            text_header = "\nالنص للتحليل:\n'''{text}'''"
            output_json_header = "\n\nكائن JSON الخاص بالإخراج (تأكد من أن القيم عددية بين 0.0 و 1.0، والمفاتيح هي معرفات المعايير الصحيحة، وتضمين فقط المعايير ذات الصلة):\n"
        else: # Default to English
            system_message_content = (
                "You are an expert in AAOIFI Financial Accounting Standards (FAS). "
                "Your task is to analyze the provided text and identify the most relevant standards from the provided list."
            )
            # --- IMPROVED PROMPT ---
            prompt_intro = (
                "Please analyze the following text, which is a financial news article about developments or transactions in Islamic finance. "
                "Identify the AAOIFI Financial Accounting Standards (FAS) from the list below that appear most relevant to or are likely impacted by the topics discussed in the news. "
                "Your output must be a JSON object with standard IDs (like 'FAS4', 'FAS28') as keys and their relevance scores (0.0 to 1.0) as values. "
                "Only include standards with a relevance score of 0.01 or higher.\n" # Lowered threshold hint
                "If the text is not relevant to any of the standards above a certain threshold (less than 0.01), output an empty JSON object {{}}.\n"
                "Example output for multiple relevance: {{ \"FAS28\": 0.8, \"FAS32\": 0.5 }}\n"
                "Example output for single relevance: {{ \"FAS7\": 1.0 }}\n"
                "Example output for no sufficient relevance: {{}}\n\n"
            )
            # --- END IMPROVED PROMPT ---
            standards_list_header = "List of possible standards to consider (ensure you output correct standard IDs like 'FAS4', not just '4'):\n"
            text_header = "\nText to analyze:\n'''{text}'''"
            output_json_header = "\n\nYour JSON output (ensure values are numerical between 0.0 and 1.0, keys are correct standard IDs, and include only relevant standards):\n"

        standards_list_items = []
        # Include details about standards in the prompt for better AI classification
        for std_id, std_data_master in self.standards.items():
            name_en = std_data_master.get("name", f"Standard {std_id}")
            # Keep description concise for the prompt
            desc_en_short = (std_data_master.get("description", "")[:150] + '...') if std_data_master.get("description") else "No description."
            # Include key concepts if available
            key_concepts_en = ", ".join([str(c) for c in std_data_master.get("key_concepts", [])[:5]]) # Limit concepts, ensure str


            if language == 'ar':
                # Get translated name and description for the prompt if available
                ar_std_trans = self.translations.get('ar', {}).get('standards', {}).get(std_id, {})
                name_display = ar_std_trans.get('name', name_en)
                # Use translated description if available, fallback to English short
                desc_display_short = (ar_std_trans.get('description', "")[:150] + '...') if ar_std_trans.get('description') else desc_en_short
                 # Use translated key concepts if available, fallback to English
                key_concepts_list = [str(c) for c in ar_std_trans.get("key_concepts", std_data_master.get("key_concepts", [])) if c] # Ensure str
                key_concepts_display = ", ".join(key_concepts_list[:5]) if key_concepts_list else key_concepts_en


                item_str = f"- {std_id}: {name_display}. الوصف: {desc_display_short}. المفاهيم الأساسية: {key_concepts_display}."
            else: # English
                name_display = name_en
                desc_display_short = desc_en_short
                key_concepts_display = key_concepts_en
                item_str = f"- {std_id}: {name_display}. Description: {desc_display_short}. Key Concepts: {key_concepts_display}."

            standards_list_items.append(item_str)

        standards_section_str = standards_list_header + "\n".join(standards_list_items)

        user_prompt_content_template = (
            prompt_intro +
            standards_section_str +
            text_header +
            output_json_header
        )
        user_prompt_content = user_prompt_content_template.format(text=text_to_classify)

        return system_message_content, user_prompt_content


    # --- Modified _call_openai_api to handle both JSON (classification) and text (explanation) ---
    # --- FIX: Use Union for return type hint compatibility with Python < 3.10 ---
    def _call_openai_api(self, model_to_use: str, system_message: str, user_prompt: str, context_for_error_msg: str, is_explanation_call: bool = False) -> Union[Dict[str, float], str]:
        """Handles interaction with the OpenAI API."""
        if not self.client:
            print(f"Error: OpenAI client not initialized in _call_openai_api for {context_for_error_msg}. Cannot perform AI call.")
            # Return appropriate empty value based on expected return type
            return {} if not is_explanation_call else ""

        try:
            response_params = {
                "model": model_to_use,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                # Higher temperature for creativity in explanations, lower for structured classification
                "temperature": 0.1 if not is_explanation_call else 0.5
            }
            # Ensure response_format is set ONLY for classification calls where JSON is expected
            if not is_explanation_call:
                # Use "json_object" to strongly encourage JSON output
                 response_params["response_format"] = {"type": "json_object"}


            response = self.client.chat.completions.create(**response_params)
            result_content = response.choices[0].message.content

            if is_explanation_call:
                return result_content.strip() if result_content else ""

            # --- Classification Response Handling (JSON) ---
            if result_content:
                 # Attempt to find JSON object within potential markdown code blocks
                match = re.search(r"```json\s*(\{.*?\})\s*```", result_content, re.DOTALL)
                if match:
                    json_string = match.group(1)
                else:
                     # Assume the whole content is the JSON string if no code block
                    json_string = result_content.strip()

                # --- DEBUG PRINT: Raw AI Classification Response ---
                print(f"--- Debug: Raw AI Classification Response ({context_for_error_msg}, model {model_to_use}) ---")
                print(f"Content: {result_content[:500]}...") # Print snippet
                print("------------------------------------------------------------")
                # --- END DEBUG PRINT ---

                if not json_string:
                    print(f"Warning: AI ({context_for_error_msg}, model {model_to_use}) returned empty content or no JSON found.")
                    return {}

                try:
                    result_dict = json.loads(json_string)
                except json.JSONDecodeError as e:
                    print(f"AI JSON Decode Error ({context_for_error_msg}, model {model_to_use}): {e}. Received: '{json_string[:300]}...'")
                    return {} # Return empty on JSON decode error

                # Validate and filter parsed scores
                parsed_scores = {}
                for k, v in result_dict.items():
                    # Ensure key is a string, corresponds to a valid standard ID, and value is a number
                    if isinstance(k, str) and k in self.standards and isinstance(v, (int, float)):
                         score_value = float(v)
                         # Enforce score >= 0 and <= 1.0 as requested in prompt (and filter very low scores)
                         if 0.0 <= score_value <= 1.0:
                             parsed_scores[k] = score_value
                         else:
                             print(f"Warning: AI returned score {score_value} outside [0, 1] range for {k} from {context_for_error_msg}. Skipping.")
                    elif isinstance(k, str) and k not in self.standards:
                        print(f"Warning: AI returned unknown standard ID '{k}' from {context_for_error_msg}. Skipping.")
                    else:
                         # Log other unexpected formats, but don't stop
                         print(f"Warning: AI returned unexpected key-value pair format ({type(k)}:{type(v)}) for {context_for_error_msg}. Pair: '{k}': {v}. Skipping.")


                # Filter based on the minimum threshold mentioned in the prompt hint (0.01)
                # We apply this filter programmatically after parsing to be sure.
                scores_above_threshold = {k: v for k, v in parsed_scores.items() if v >= 0.01} # Filter based on prompt hint (0.01)


                if not scores_above_threshold:
                    print(f"Warning: AI ({context_for_error_msg}, model {model_to_use}) returned no relevant standard scores above threshold (0.01) after parsing/filtering from: {json_string[:200]}")
                    return {}

                # --- DEBUG PRINT: Parsed & Filtered Scores ---
                print(f"--- Debug: Parsed & Filtered AI Classification Scores ({context_for_error_msg}) ---")
                print(f"Scores: {json.dumps(scores_above_threshold, indent=2)}")
                print("------------------------------------------------------------")
                # --- END DEBUG PRINT ---

                return scores_above_threshold # Return the dictionary of validated and filtered scores

            else: # result_content was empty
                print(f"Warning: AI ({context_for_error_msg}, model {model_to_use}) returned empty content.")
                # Return appropriate empty value based on expected return type
                return {} if not is_explanation_call else ""


        except Exception as e:
            print(f"Generic AI API Error ({context_for_error_msg}, model {model_to_use}): {e}")
            # Ensure we return the correct type based on what was expected
            return {} if not is_explanation_call else ""


    # --- Primary AI Classification Method ---
    def _classify_with_prompt(self, text_to_classify: str) -> Dict[str, float]:
        """
        Classifies text using a general LLM prompt.
        Returns a dictionary {standard_id: relevance_score}.
        """
        if not self.client:
             print("Error: OpenAI client not initialized. Cannot perform AI classification.")
             return {}

        if not self.standards:
             print("Warning: No standards loaded. AI classification prompt cannot list standards.")
             return {}

        language = self.detect_language(text_to_classify)
        system_message, user_prompt = self._build_ai_classification_prompt(text_to_classify, language)

        # Using a general model ID for classification
        model_id = "gpt-3.5-turbo-1106" # Or other capable models like "gpt-4o"
        context = "General AI Model (Classification Prompt)"
        print(f"Analyzing with {context}: {model_id}")

        # Call the AI API - _call_openai_api handles parsing and validation
        ai_scores = self._call_openai_api(model_id, system_message, user_prompt, context, is_explanation_call=False)

        return ai_scores # Return the dictionary from _call_openai_api


    # --- The main public classification method ---
    def classify_transaction(
        self,
        text_to_classify: str,
        # Removed weight arguments as they are not used in this version
    ) -> List[Tuple[str, float]]:
        """
        Classifies transaction/news text against AAOIFI standards using ONLY LLM prompting.
        Returns a sorted list of (standard_id, score) tuples.
        """
        if not text_to_classify or not text_to_classify.strip():
            print("Warning: classify_transaction received empty text.")
            return []

        print(f"\n--- Classifying Text using AI Prompting Only (first 50 chars): '{text_to_classify[:50].strip()}...' ---")

        # --- Use only the AI prompt classification method ---
        ai_scores_dict = self._classify_with_prompt(text_to_classify)

        if not ai_scores_dict:
             print("No relevant standards identified by the AI classifier.")
             return []

        # Convert dictionary to list of tuples and sort by score descending
        classified_results = sorted(ai_scores_dict.items(), key=lambda item: item[1], reverse=True)

        print(f"Final AI Classification Results ({len(classified_results)} standards):")
        # Add a debug print of the final sorted results
        for std_id, score in classified_results:
            print(f"  - {std_id}: {score:.4f}") # Print with more decimal places for scores
        print("--- Classification Process Finished ---\n")


        return classified_results # Return sorted list of (id, score) tuples


    # --- Keep _get_ai_explanation_for_standard for Explanations ---
    def _get_ai_explanation_for_standard(self, text: str, std_id: str, language: str) -> str:
        """Prompts the AI for a detailed explanation of a standard's relevance to the text."""
        if std_id not in self.standards:
            return ""

        std_info = self.standards[std_id]
        std_name_en = std_info.get("name", f"Standard {std_id}")
        std_desc_en = std_info.get("description", "No description available.")
        # Use the standard's key concepts from the loaded data for the prompt
        std_concepts_list = [str(c) for c in std_info.get("key_concepts", []) if c] # Ensure str

        if language == 'ar':
            ar_std_trans = self.translations.get('ar', {}).get('standards', {}).get(std_id, {})
            std_name_display = ar_std_trans.get('name', std_name_en)
            std_desc_display = ar_std_trans.get('description', std_desc_en)
             # Use translated key concepts if available from translations, fallback to English concepts
            translated_concepts_list = [str(c) for c in ar_std_trans.get('key_concepts', []) if c]
            concepts_display = ", ".join(translated_concepts_list) if translated_concepts_list else ", ".join(std_concepts_list)


            system_msg = (
                "أنت مساعد خبير في معايير التمويل الإسلامي. قم بتقديم شرح مفصل وواضح لماذا يعتبر المعيار المحدد مناسبًا للنص المقدم."
                "ركز على ربط تفاصيل النص بالمفاهيم والوصف الخاص بالمعيار المحدد."
            )
            user_prompt = (
                f"اشرح بالتفصيل وبلغة واضحة وموجزة لماذا يعتبر المعيار '{std_name_display} ({std_id})' ملائمًا للنص التالي:\n\n"
                f"تفاصيل المعيار:\n"
                f"  الاسم: {std_name_display}\n"
                f"  الوصف: {std_desc_display}\n"
                f"  المفاهيم الأساسية: {concepts_display}\n\n" # Use concepts in prompt
                f"النص للتحليل:\n'''{text}'''\n\n"
                f"شرحك (يرجى تقديم أسباب مفصلة بناءً على عناصر النص وارتباطها بالمعيار):\n"
            )
        else: # English
            std_name_display = std_name_en
            std_desc_display = std_desc_en
            concepts_display = ", ".join(std_concepts_list) # Use concepts in prompt

            system_msg = (
                "You are an expert Islamic Finance Standards assistant. Provide a detailed and clear explanation "
                "for why the specified standard is relevant to the provided text."
                "Focus on connecting elements of the text to the standard's concepts and description."
            )
            user_prompt = (
                f"Explain in detail, using clear and concise language, why the standard '{std_name_display} ({std_id})' is relevant to the following text:\n\n"
                f"Standard Details:\n"
                f"  Name: {std_name_display}\n"
                f"  Description: {std_desc_display}\n"
                f"  Key Concepts: {concepts_display}\n\n" # Use concepts in prompt
                f"Text for Analysis:\n'''{text}'''\n\n"
                f"Your Explanation (Please provide detailed reasoning based on the text elements and their connection to the standard):\n"
            )

        # Use a general model for explanation
        model_for_explanation = "gpt-3.5-turbo-1106" # Or "gpt-4o", etc.

        print(f"Requesting AI explanation for {std_id} using model {model_for_explanation}...")
        ai_explanation_text = self._call_openai_api(
            model_to_use=model_for_explanation,
            system_message=system_msg,
            user_prompt=user_prompt,
            context_for_error_msg=f"AI Explanation for {std_id}",
            is_explanation_call=True # Indicate this is an explanation call
        )

        return ai_explanation_text

    # --- Keep explain_classification for Explanations ---
    def explain_classification(self, text: str, top_standards_scores: List[Tuple[str, float]]) -> Dict[str, str]:
        """
        Generates explanations for the given standard classifications.
        Explanation format includes standard overview and AI reasoning.
        text: The original transaction/news text.
        top_standards_scores: A list of (standard_id, score) tuples from classify_transaction.
        """
        if not text or not top_standards_scores:
            return {}

        language = self.detect_language(text)
        explanations_dict: Dict[str, str] = {}

        print("\n--- Generating Explanations ---")
        # Generate explanations for the top N standards (e.g., top 5)
        standards_to_explain = top_standards_scores[:5] # Limit explanations to top 5 or fewer

        for std_id, score_value in standards_to_explain:
            if std_id not in self.standards:
                print(f"Warning (Explain): Standard ID {std_id} not in loaded standards.")
                continue

            # Get display names/descriptions from translations for the explanation text
            master_std_data = self.standards[std_id]
            base_std_name = master_std_data.get("name", f"Standard {std_id}")
            base_std_desc = master_std_data.get("description", "No description available.")

            current_lang_std_translation = self.translations.get(language, {}).get('standards', {}).get(std_id, {})
            display_name = current_lang_std_translation.get('name', base_std_name)
            display_desc = current_lang_std_translation.get('description', base_std_desc)

            # --- The explanation HTML will only include standard overview and AI reasoning ---

            detailed_ai_reasoning = self._get_ai_explanation_for_standard(text, std_id, language)
            processed_reasoning_html = ""
            if detailed_ai_reasoning:
                # Simple replacement of newlines with <br> for basic formatting
                processed_reasoning_html = detailed_ai_reasoning.replace('\n', '<br>')


            # --- Construct the final explanation string (can use HTML formatting for frontend) ---
            if language == 'ar':
                explanation_html = f"<h4>{display_name} ({std_id}) - الصلة: {score_value*100:.0f}%</h4>"
                explanation_html += f"<p><strong>الوصف العام للمعيار:</strong> {display_desc}</p>" # Keep general description
                 # Removed keyword line from previous versions
                if detailed_ai_reasoning:
                    explanation_html += f"<p><strong>التحليل التفصيلي المستند إلى الذكاء الاصطناعي:</strong></p><p>{processed_reasoning_html}</p>"
                else:
                    explanation_html += "<p><strong>التحليل التفصيلي المستند إلى الذكاء الاصطناعي:</strong> لم يتمكن الذكاء الاصطناعي من تقديم تحليل تفصيلي لهذه الحالة.</p>"
            else: # English
                explanation_html = f"<h4>{display_name} ({std_id}) - Relevance: {score_value*100:.0f}%</h4>"
                explanation_html += f"<p><strong>Standard Overview:</strong> {display_desc}</p>" # Keep general description
                 # Removed keyword line from previous versions
                if detailed_ai_reasoning:
                    explanation_html += f"<p><strong>Detailed AI Analysis:</strong></p><p>{processed_reasoning_html}</p>"
                else:
                    explanation_html += "<p><strong>Detailed AI Analysis:</strong> The AI could not provide a detailed analysis for this case.</p>"

            explanations_dict[std_id] = explanation_html

        print("--- Explanation Generation Finished ---\n")
        return explanations_dict


# --- Updated __main__ block for testing the AI Prompting Only Classifier ---
if __name__ == "__main__":
    # Ensure load_dotenv is called here again for the __main__ block,
    # especially if you run this file directly. override=True ensures it reloads if already called.
    load_dotenv(override=True)

    main_openai_api_key = os.getenv("OPENAI_API_KEY")
    if not main_openai_api_key:
        print("CRITICAL: OPENAI_API_KEY environment variable not set for __main__ block. Cannot run example.")
    else:
        print(f"__main__: OPENAI_API_KEY is set (masked: sk-...{main_openai_api_key[-4:]}). Proceeding.")

        # Determine project root dynamically to locate data files
        # Assuming this classifier.py is in the same directory as app.py
        project_root = os.path.dirname(os.path.abspath(__file__)) # If in the same directory


        main_standards_file = os.path.join(project_root, "data", "standards.json")
        main_translations_file = os.path.join(project_root, "data", "translations.json")

        # Ensure these files exist for testing, create dummy ones if needed for __main__
        os.makedirs(os.path.join(project_root, "data"), exist_ok=True)

        # Dummy standards.json structure needed for prompt content and explanations
        dummy_standards_content = {
            "FAS4": {"name": "Musharaka Financing", "description": "Covers accounting rules for Musharaka (partnership finance).", "key_concepts": ["partnership", "profit sharing", "loss bearing", "joint venture", "Mudarabah", "Sharik"]},
            "FAS7": {"name": "Salam and Parallel Salam", "description": "Covers accounting for Salam contracts (forward sale with upfront payment).", "key_concepts": ["salam", "forward sale", "upfront payment", "deferred delivery", "agricultural finance", "commodity finance"]},
            "FAS10": {"name": "Istisna'a and Parallel Istisna'a", "description": "Covers accounting for Istisna'a contracts (manufacturing/construction finance).", "key_concepts": ["istisna'a", "manufacturing", "construction", "progress payments", "contractor", "beneficiary", "building", "project"]},
            "FAS28": {"name": "Investments in Real Estate", "description": "Covers accounting for investments in real estate properties.", "key_concepts": ["real estate", "property investment", "rental income", "fair value", "cost model", "development property"]},
            "FAS32": {"name": "Ijarah", "description": "Covers accounting for Ijarah contracts (leasing).", "key_concepts": ["ijarah", "lease", "leasing", "right-of-use asset", "rental", "lessee", "lessor", "operating lease", "finance lease", "Ijarah Muntahia Bittamleek"]},
            "FAS11": {"name": "Provisions and Reserves", "description": "Covers accounting for provisions, contingent liabilities and contingent assets.", "key_concepts": ["provision", "reserve", "contingent liability", "contingent asset", "obligation", "liability", "asset"]},
            "FAS30": {"name": "Impairment, Credit Losses and Onerous Commitments", "description": "Addresses impairment of assets, recognition and measurement of credit losses, and accounting for onerous commitments.", "key_concepts": ["impairment", "credit loss", "expected credit loss (ECL)", "onerous contract", "recoverable amount", "bad debt", "allowance"]}
        }

        if not os.path.exists(main_standards_file):
            print(f"Warning: Dummy standards.json created for __main__ test as it was not found at {main_standards_file}")
            with open(main_standards_file, 'w', encoding='utf-8') as f: json.dump(dummy_standards_content, f, indent=2, ensure_ascii=False)
        else:
            # If file exists, load it to ensure the classifier uses its content
            try:
                with open(main_standards_file, 'r', encoding='utf-8') as f:
                    loaded_standards_check = json.load(f)
                if not loaded_standards_check:
                     print(f"Warning: standards.json found at {main_standards_file} is empty or invalid. Using dummy content for __main__ test.")
                     with open(main_standards_file, 'w', encoding='utf-8') as f: json.dump(dummy_standards_content, f, indent=2, ensure_ascii=False)
                else:
                     print(f"Using standards.json found at {main_standards_file}.")
            except Exception as e:
                 print(f"Error loading standards.json at {main_standards_file}: {e}. Using dummy content for __main__ test.")
                 with open(main_standards_file, 'w', encoding='utf-8') as f: json.dump(dummy_standards_content, f, indent=2, ensure_ascii=False)


        # Dummy translations.json structure needed for prompt and explanation translations
        dummy_translations_content = {
            "en": {"title": "Test"},
            "ar": {"title": "اختبار", "standards": {
                "FAS4": {"name": "تمويل المشاركة", "description": "يغطي قواعد المحاسبة لعقود المشاركة."},
                "FAS7": {"name": "السلم والسلم الموازي", "description": "يغطي المحاسبة لعقود السلم."},
                "FAS10": {"name": "الاستصناع والاستصناع الموازي", "description": "يغطي المحاسبة لعقود الاستصناع."},
                "FAS28": {"name": "الاستثمارات العقارية", "description": "يغطي المحاسبة عن الاستثمارات في العقارات."},
                "FAS32": {"name": "الإجارة", "description": "يغطي المحاسبة لعقود الإجارة."},
                "FAS11": {"name": "المخصصات والاحتياطيات", "description": "تغطي المحاسبة عن المخصصات."},
                "FAS30": {"name": "اضمحلال القيمة والخسائر الائتمانية والالتزامات المرهقة", "description": "يتناول اضمحلال قيمة الأصول والخسائر الائتمانية."}
            }}
        }

        if not os.path.exists(main_translations_file):
            print(f"Warning: Dummy translations.json created for __main__ test as it was not found at {main_translations_file}")
            with open(main_translations_file, 'w', encoding='utf-8') as f: json.dump(dummy_translations_content, f, indent=2, ensure_ascii=False)
        else:
             # If file exists, load it to ensure the classifier uses its content
            try:
                with open(main_translations_file, 'r', encoding='utf-8') as f:
                    loaded_translations_check = json.load(f)
                if not loaded_translations_check:
                     print(f"Warning: translations.json found at {main_translations_file} is empty or invalid. Using dummy content for __main__ test.")
                     with open(main_translations_file, 'w', encoding='utf-8') as f: json.dump(dummy_translations_content, f, indent=2, ensure_ascii=False)
                else:
                    print(f"Using translations.json found at {main_translations_file}.")
            except Exception as e:
                 print(f"Error loading translations.json at {main_translations_file}: {e}. Using dummy content for __main__ test.")
                 with open(main_translations_file, 'w', encoding='utf-8') as f: json.dump(dummy_translations_content, f, indent=2, ensure_ascii=False)


        # Fine-tuned model ID is not used in this version
        main_fine_tuned_id = os.getenv("FINE_TUNED_MODEL_ID")
        if main_fine_tuned_id:
            print(f"__main__: Note - FINE_TUNED_MODEL_ID ({main_fine_tuned_id}) is set but NOT used for classification in this classifier version.")

        classifier_instance = StandardsClassifier(
            standards_file=main_standards_file,
            translations_file=main_translations_file,
            # fine_tuned_model_id=main_fine_tuned_id, # Not passed anymore
            openai_api_key=main_openai_api_key
        )

        # --- Test cases using news-like text ---
        test_news_text = {
            "English News - Digital Assets & Tokenized Real Estate": """
            Recent news indicates a surge in Islamic financial institutions exploring the issuance of Shariah-compliant digital assets,
            potentially backed by real-world assets or future revenues. Discussions are ongoing regarding the accounting treatment
            and classification of tokenized real estate and the underlying assets under existing AAOIFI standards.
            There's a need for clear guidance on the recognition and measurement of these digital tokens.
            Questions also arise about disclosure requirements and potential impairment risks.
            """,
            "Arabic News - Green Finance & Ijarah Sukuk": """
            تتزايد اهتمام البنوك الإسلامية بالتمويل الأخضر والصكوك المستندة إلى الأصول الخضراء.
            على سبيل المثال، يتم استكشاف صكوك الإجارة المدعومة بمشاريع الطاقة المتجددة أو العقارات المستدامة.
            يبرز هذا التطور الحاجة إلى توضيح المعالجة المحاسبية لهذه الهياكل الجديدة ضمن معايير AAOIFI الحالية،
            خاصة فيما يتعلق بتقييم الأصول وتوزيع الإيرادات وتحديد الالتزامات المحتملة.
            """,
             "English News - Murabaha & Provisions": """
            A recent regulatory review highlighted inconsistencies in the accounting treatment of Murabaha receivables across several Islamic banks.
            Issues were noted particularly in the calculation of profit recognition over the deferred payment period and the methods for handling potential credit losses and provisions.
            This indicates a potential need for clearer guidance or stricter interpretation of the existing Murabaha standard to ensure uniformity and transparency.
            """,
            "Arabic News - Salam in Agriculture": """
            شهد قطاع التمويل الزراعي الإسلامي نمواً ملحوظاً في استخدام عقود السلم لتمويل المزارعين لشراء البذور والمعدات، على أن يتم تسليم المحصول في تاريخ لاحق.
            يثير هذا التوسع تساؤلات حول إدارة مخاطر السلم، بما في ذلك جودة المحصول ومواعيد التسليم، والمعالجة المحاسبية الصحيحة لدفعات السلم المقدمة واستلام الأصول المؤجلة.
            """,
            # --- Test case for FAS 32 without explicit mention ---
            "English News - FAS 32 Concepts (Leasing/Usage Rights)": """
            A major Islamic bank announced a new offering allowing small businesses to access heavy machinery without outright purchase.
            The arrangement involves periodic payments over a fixed term, at the end of which the business has the option to acquire the equipment for a nominal fee.
            This innovative financing structure provides flexible usage rights for valuable assets, addressing capital constraints for entrepreneurs.
            """,
            # --- End Test case ---
            "Text with No Relevant Concepts": "This text discusses global economic trends unrelated to Islamic finance standards. Inflation is high. The stock market is volatile. There is a new report on environmental policy." # Text with no relevant terms or very general ones
        }


        print(f"\n__main__: Testing the AI Prompting Only Classifier with News Text")


        for desc, text in test_news_text.items():
            print(f"\n===== Testing News Text: {desc} =====")
            detected_lang = classifier_instance.detect_language(text)
            print(f"Detected Language: {detected_lang}")

            # Call classify_transaction (which now uses only AI prompting)
            classified_results = classifier_instance.classify_transaction(text)

            if not classified_results:
                print("No relevant standards identified for this news text by the AI classifier.")
                continue

            print("\nAI Classification Results (Sorted by Relevance Score):")
            for std_id_res, score_res in classified_results:
                 # Try to get display name from loaded standards data (classifier uses its own standards.json)
                 std_info_for_display = classifier_instance.standards.get(std_id_res, {})
                 std_name_display = std_info_for_display.get('name', std_id_res)
                 if detected_lang == 'ar' and 'ar' in classifier_instance.translations and std_id_res in classifier_instance.translations['ar'].get('standards', {}):
                      std_name_display_ar = classifier_instance.translations['ar']['standards'][std_id_res].get('name', std_name_display)
                      std_name_display = f"{std_name_display} ({std_name_display_ar})"


                 print(f"  - {std_name_display} ({std_id_res}): {(score_res*100):.1f}%")

            # Get explanations for top N standards (e.g., top 5)
            explanations_for_results = classifier_instance.explain_classification(text, classified_results[:5])
            print("\nExplanations for top identified standards:")
            for std_id_exp, explanation_html in explanations_for_results.items():
                 # Basic cleaning for console output
                console_explanation = re.sub(r'<br\s*/?>', '\n    ', explanation_html)
                console_explanation = re.sub(r'<[^<]+?>', '', console_explanation)
                print(f"  --- For {std_id_exp} --- \n    {console_explanation.strip()}\n")
            print("=" * 30)
# --- END OF FILE classifier.py (New - AI Prompting Only Classification) ---