import os
from dotenv import load_dotenv
import requests
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
import traceback

load_dotenv()

# --- Global Configuration ---
LLM_PROVIDER = "OPENAI"
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GNEWS_API_URL = "https://gnews.io/api/v4/search"

# --- LLM Initialization based on Provider ---
if LLM_PROVIDER == "OPENAI":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY: print("CRITICAL: OPENAI_API_KEY missing."); exit(1)
    from langchain_openai import ChatOpenAI
    LLM_MODEL_NAME = "gpt-4o"
    LLM_MODEL_FAST = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.2, openai_api_key=OPENAI_API_KEY)
    llm_structured = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.1, openai_api_key=OPENAI_API_KEY)
    llm_reviewer = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.2, openai_api_key=OPENAI_API_KEY)
    llm_extractor = ChatOpenAI(model_name=LLM_MODEL_FAST, temperature=0.0, openai_api_key=OPENAI_API_KEY)
elif LLM_PROVIDER == "GROQ":
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY: print("CRITICAL: GROQ_API_KEY missing."); exit(1)
    from langchain_groq import ChatGroq
    GROQ_MODEL_GENERAL = "llama3-8b-8192"
    GROQ_MODEL_STRUCTURED = "llama3-8b-8192"
    GROQ_MODEL_EXTRACTOR = "llama3-8b-8192"
    llm = ChatGroq(temperature=0.2, groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL_GENERAL)
    llm_structured = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL_STRUCTURED)
    llm_reviewer = ChatGroq(temperature=0.2, groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL_GENERAL)
    llm_extractor = ChatGroq(temperature=0.0, groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL_EXTRACTOR)
else:
    print(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}"); exit(1)

def load_standard_from_file(file_path):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, file_path)
        with open(full_path, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError: print(f"ERROR: File not found: {full_path}"); return None
    except Exception as e: print(f"ERROR: Reading file {full_path}: {e}"); return None

STANDARD_FILES = {
    "FAS4": { "name": "AAOIFI FAS No. 4 - Musharaka Financing", "short_name": "Musharaka (FAS 4)", "path": "standards/fas4_musharaka.txt"},
    "FAS7": { "name": "AAOIFI FAS No. 7 - Salam", "short_name": "Salam (FAS 7)", "path": "standards/fas7_salam.txt"},
    "FAS10": { "name": "AAOIFI FAS No. 10 - Istisna’a", "short_name": "Istisna'a (FAS 10)", "path": "standards/fas10_istisna.txt"},
    "FAS28": { "name": "AAOIFI FAS No. 28 - Real Estate Inv.", "short_name": "Real Estate (FAS 28)", "path": "standards/fas28_realestate.txt"},
    "FAS32": { "name": "AAOIFI FAS 32 - Ijarah", "short_name": "Ijarah (FAS 32)", "path": "standards/fas32_ijarah.txt"}
}
THEMATIC_AGENT_TYPES = {
    "Foundations_Compliance": {"name": "Foundations & Compliance", "description": "Focuses on Riba, Gharar, transparency, Shariah governance."},
    "Asset_Risk_Accounting": {"name": "Asset, Risk & Accounting", "description": "Focuses on asset-backing, profit/risk recognition."}
}
def get_mock_news(reason="Test"):
    return f"""--- Recent Finance News (MOCK DATA - Reason: {reason}) ---
1. Title: AI in Islamic Finance Audit. Summary: AI for auditing Shariah compliance. URL: #mock1
2. Title: Digital Transformation. Summary: Tech for transparency. URL: #mock2
-------------------------"""

def fetch_finance_news_gnews(query, num_articles=3):
    MOCK_NEWS_ON_ERROR = True
    if not GNEWS_API_KEY: return get_mock_news("GNEWS_API_KEY missing.") if MOCK_NEWS_ON_ERROR else "Error: Key missing."
    params = {"q": query, "lang": "en", "country": "us,gb,ae,my,sa", "max": num_articles, "apikey": GNEWS_API_KEY, "sortby": "publishedAt"}
    # print(f"GNews Query (raw): '{query}'")
    try:
        r = requests.get(GNEWS_API_URL, params=params, timeout=20); print(f"GNews Status: {r.status_code}"); r.raise_for_status()
        data, nt = r.json(), "--- Recent Finance News ---\n"; articles = data.get('articles', [])
        if not articles: nt += f"No news for: '{query}'.\n"
        else:
            for i, a in enumerate(articles): nt += f"{i+1}. {a.get('title','N/A')}\n   Desc: {a.get('description','N/A')}\n   URL: {a.get('url', '#')}\n"
        return nt + "-------------------------\n"
    except requests.exceptions.HTTPError as e: print(f"GNews HTTP Err: {e}. Resp: {e.response.text}"); return get_mock_news(f"GNews Err {e.response.status_code}") if MOCK_NEWS_ON_ERROR else f"Err: GNews HTTP {e.response.status_code}"
    except Exception as e: print(f"GNews Ex: {e}"); return get_mock_news(f"GNews Ex {type(e).__name__}") if MOCK_NEWS_ON_ERROR else f"Err: GNews Ex {type(e).__name__}"

analyzer_template_str = """**Expert AAOIFI Analyst:** Analyze '{standard_name}'. TEXT: --- {standard_text} --- INSTR: 1. Summary. 2. Key Elements (for '{standard_short_name}'). 3. Enhancements (AI for '{standard_short_name}'). OUTPUT: Markdown."""
analyzer_prompt = ChatPromptTemplate.from_template(analyzer_template_str)
analyzer_agent: RunnableSequence = analyzer_prompt | llm_structured

proposer_template_str = """**You are an Innovative Islamic Finance & AI Strategist (Theme: {theme_description}).**
Your task is to propose specific, actionable enhancements for '{standard_name}'.
**Inputs:** Analysis: --- {analysis_output} --- News: --- {finance_news} ---
**Instructions (Focus: {theme_description}):** Propose 3-5 modifications for '{standard_short_name}'. For EACH proposal:
1.  **Proposal Title:** Create a concise title (e.g., "Enhancing X for Y"). This exact title MUST be the H2 heading.
2.  **Proposed Change:** ... 3. **Relevant Section(s):** ... 4. **Rationale:** ... 5. **AI/Tech Link:** ... 6. **Expected Benefit:** ...
**OUTPUT Format Example (Strictly Follow for EACH proposal):**
## [Your Specific Proposal Title From Point 1]
**Proposed Change Description:** ...
(Continue with points 2-6)

## [Your Next Specific Proposal Title From Point 1]
**Proposed Change Description:** ...
(Continue with points 2-6 for 3-5 proposals)"""
proposer_prompt = ChatPromptTemplate.from_template(proposer_template_str)
proposer_agent: RunnableSequence = proposer_prompt | llm

thematic_validator_template_str = """**Esteemed Shariah Scholar & AAOIFI Compliance Officer (Theme: {theme_description}):** Validate the single proposal provided for '{standard_name}'.
PROPOSAL TO VALIDATE:
---
{single_proposal_markdown}
---
ANALYSIS CONTEXT: --- {analysis_output} --- NEWS CONTEXT: --- {finance_news} ---
INSTRUCTIONS: Provide your validation for THE ONE proposal above.
For each point below, provide your assessment. **YOU MUST USE THE EXACT FORMATTING SHOWN for points 3 and 4, including the exact labels and placeholders. DO NOT ADD ANY EXTRA TEXT ON LINES 3 AND 4 OTHER THAN THE REQUESTED SCORE OR VERDICT KEYWORD WITHIN THE BRACKETS.**
1.  **Shariah Compliance Assessment** ({theme_description} view): *   Alignment with Shariah: [Your assessment text here] *   AI & Shariah Risk: [Your assessment text here]
2.  **Compliance & Practicality Assessment** ({theme_description} view): *   AAOIFI Alignment: [Your assessment text here] *   Implementability: [Your assessment text here] *   Clarity: [Your assessment text here]
**BEGIN REQUIRED FORMAT SECTION - FOLLOW EXACTLY:**
3.  **Numerical Score (1-10):** [YOUR_SINGLE_NUMBER_SCORE_HERE_AS_A_DIGIT_e.g.,_8_NO_OTHER_TEXT_ON_THIS_LINE_EXCEPT_THE_NUMBER_IN_BRACKETS]
4.  **Overall Verdict:** [CHOOSE_ONE_KEYWORD_ONLY_FROM_THE_FOLLOWING:_Approved_OR_Approved with Revisions_OR_Rejected_NO_OTHER_TEXT_ON_THIS_LINE_EXCEPT_THE_KEYWORD_IN_BRACKETS]
**END REQUIRED FORMAT SECTION.**
5.  **Justification for Verdict** ({theme_description} view): [Your brief justification here.]
OUTPUT: Your entire output for this single proposal validation MUST start with the heading:
`## Validation by {theme_description} for: [Exact ## Proposal Title from Input Proposal To Validate]`
Then, list your detailed responses for points 1 through 5 using the exact bolded labels as shown above, especially for points 3 and 4.
"""
thematic_validator_prompt = ChatPromptTemplate.from_template(thematic_validator_template_str)
thematic_validator_agent: RunnableSequence = thematic_validator_prompt | llm_structured

# UPDATED META-REVIEWER PROMPT
meta_reviewer_template_str = """
**You are a Chief Review Officer.** Your task is to provide a STRICT final assessment and **reasoned justification** for a given proposal, based on its average score and, more importantly, the qualitative feedback and specific concerns or strengths identified by multiple thematic validators.

**Original Proposal Being Assessed:**
Title: {proposal_title}
Full Text (body of the proposal):
{proposal_full_text_body}
---
**Aggregated Validation Summary:**
Average Score (out of 10): {average_score}

**Summary of Thematic Validator Feedback (Scores, Verdicts, and Key Justification Points/Snippets):**
{detailed_validations_summary_for_meta}
---
**Instructions for STRICT Assessment & Reasoned Justification:**

1.  **Analyze Validator Feedback:** Focus on the *substance* of the "Key Justification Points" provided by each thematic validator. Identify common themes, critical endorsements, or overriding concerns (e.g., Shariah issues, practical flaws). An "N/A" for a score, verdict, or justification snippet from a validator means that information was not clearly provided or parsed and should be treated as incomplete validation from that validator.
2.  **Decision Criteria (use your judgment based on feedback provided):**
    *   **Accepted:** Requires a high average score (e.g., >= 7.5). More importantly, the *justifications* from validators must be predominantly positive, indicating strong alignment with Shariah, AAOIFI objectives, and practicality. All critical thematic validators (e.g., "Foundations & Compliance") must have provided a positive ("Approved" or "Approved with Revisions") verdict with supporting justification. Any revisions mentioned should be clearly minor.
    *   **Potentially Accepted with Revisions:** Average score might be moderate (e.g., 5.0-7.9). The validators' justifications indicate the core idea is sound but point to specific, actionable areas for improvement. There should be no fundamental "Rejected" verdicts on Shariah grounds unless the justification for revision clearly shows how it can be overcome.
    *   **Rejected:** Low average score (e.g., < 5.0), OR if any validator's justification reveals a critical, unresolvable Shariah non-compliance, major practical flaw, or fundamental disagreement with the proposal's core. If validator justifications are largely "N/A" or crucial validators provided "N/A" verdicts/scores, lean towards rejection due to insufficient validation for a high-stakes standard enhancement.
3.  Based on your analysis of the validators' *reasoning* and the criteria above, decide an **Overall Status**. Choose **ONLY ONE** from: "Accepted", "Potentially Accepted with Revisions", "Rejected".
4.  Provide a **Consolidated Justification** (3-5 sentences). **This is the most important part.** Your justification MUST:
    *   Explain the *primary reasons* for the chosen status by synthesizing the key arguments (pro and con) from the thematic validators' justifications.
    *   **Focus on what *is* available in the justifications.** If a validator's justification is missing or "N/A", acknowledge that its full reasoning isn't available, but still base your decision on the overall picture from other validators and the proposal itself. Do not invent reasons for the "N/A".
    *   If "Potentially Accepted with Revisions," briefly state the *nature* of the most critical revisions needed, drawing directly from the validators' justifications.
    *   If "Rejected," clearly state the main overriding concern(s) highlighted in the validators' justifications.
    *   Connect your reasoning back to the overall goal of enhancing the AAOIFI standard effectively and responsibly.

**Output Format (Strictly follow this):**
**Overall Status:** [Your Chosen Status]
**Consolidated Justification:** [Your detailed, reasoned justification, 3-5 sentences, based on the thematic validators' REASONS]
"""
meta_reviewer_prompt = ChatPromptTemplate.from_template(meta_reviewer_template_str)
meta_reviewer_agent: RunnableSequence = meta_reviewer_prompt | llm_reviewer

score_verdict_extractor_template_str = """
You are an expert data extraction tool. From the provided VALIDATION TEXT block, extract the numerical score and the single keyword verdict.
VALIDATION TEXT:
---
{validation_text_block}
---
Identify:
1.  **Numerical Score:** The number after a label like "Numerical Score (1-10):". If not found, return null for the score.
2.  **Overall Verdict:** One of "Approved", "Approved with Revisions", or "Rejected". If not clear, return "N/A" for the verdict.
Your ENTIRE output MUST be a single, valid JSON string formatted EXACTLY like this:
{{
  "score": [the extracted numerical score OR null if not found],
  "verdict": "[the extracted verdict keyword string OR "N/A" if not found]"
}}
Example 1: If score is 8 and verdict is Approved, output: {{"score": 8, "verdict": "Approved"}}
Example 2: If score is not found and verdict is Rejected, output: {{"score": null, "verdict": "Rejected"}}
Example 3: If score is 7.5 and verdict is Approved with Revisions, output: {{"score": 7.5, "verdict": "Approved with Revisions"}}
"""
score_verdict_extractor_prompt = ChatPromptTemplate.from_template(score_verdict_extractor_template_str)
score_verdict_extractor_agent: RunnableSequence = score_verdict_extractor_prompt | llm_extractor

app = Flask(__name__)
CORS(app)

def parse_proposals_from_markdown(markdown_text, source_theme_name):
    proposals = []; parts = re.split(r'(?=##\s+)', markdown_text)
    intro_text = ""
    if parts and parts[0] and not parts[0].strip().startswith("##"): intro_text = parts[0].strip(); parts = parts[1:]
    elif parts and parts[0].strip() == "": parts = parts[1:]
    for block in parts:
        if not block.strip().startswith("## "): continue
        nl_idx = block.find('\n')
        title_l, body = (block.strip(), "") if nl_idx == -1 else (block[:nl_idx].strip(), block[nl_idx+1:].strip())
        clean_t = title_l.replace("## ", "", 1).strip()
        if not clean_t: clean_t = f"Unnamed {len(proposals)+1} from {source_theme_name}"; print(f"WARN: Proposer {source_theme_name} empty title.")
        proposals.append({"id":f"p_{source_theme_name.replace(' ','_')}_{len(proposals)}","title":clean_t,"full_text_markdown":body,"source_theme":source_theme_name})
    if not proposals and markdown_text.strip() and not intro_text : print(f"WARN: No '## Title' proposals from {source_theme_name}.")
    return proposals

def parse_validations_with_scores(validation_md, validator_theme, original_proposal_title_expected):
    assessments = []
    header_match = re.search(r'(?i)##\s*Validation by.*?\s*for:\s*(.*)', validation_md)
    val_prop_title_parsed_from_header = header_match.group(1).strip() if header_match else original_proposal_title_expected
    if not header_match:
        print(f"WARN Val-{validator_theme}: Output for '{original_proposal_title_expected}' missing '## Validation by...for:' header. Content: {validation_md[:250]}...")
    elif val_prop_title_parsed_from_header.strip().lower() != original_proposal_title_expected.strip().lower():
         print(f"WARN Val-{validator_theme}: Title mismatch! Expected '{original_proposal_title_expected}', Validator Header FOR '{val_prop_title_parsed_from_header}'.")

    score = None; verdict = "N/A"; justification = "N/A"
    score_match_prompted = re.search(r"(?i)3\.\s*\*\*Numerical Score \(1-10\):\*\*\s*\[(\d+(?:\.\d+)?)\]", validation_md)
    if score_match_prompted:
        try: score = float(score_match_prompted.group(1))
        except ValueError: print(f"WARN ScoreConvertErr (Strict): {score_match_prompted.group(1)}")

    verdict_match_prompted = re.search(r"(?i)4\.\s*\*\*Overall Verdict:\*\*\s*\[(Approved(?: with Revisions)?|Rejected|Revisions)\]", validation_md)
    if verdict_match_prompted:
        verdict_candidate = verdict_match_prompted.group(1).strip()
        if "approved with revisions" in verdict_candidate.lower() or "revisions" in verdict_candidate.lower(): verdict = "Approved with Revisions"
        elif "approved" in verdict_candidate.lower(): verdict = "Approved"
        elif "rejected" in verdict_candidate.lower(): verdict = "Rejected"
        else: verdict = verdict_candidate

    if score is None or verdict == "N/A":
        print(f"INFO: Regex failed for score ({score}) or verdict ('{verdict}') for '{val_prop_title_parsed_from_header}' by {validator_theme}. Using LLM Extractor.")
        try:
            extraction_result = score_verdict_extractor_agent.invoke({"validation_text_block": validation_md})
            extraction_content = extraction_result.content if hasattr(extraction_result, 'content') else str(extraction_result)
            json_data = None
            json_match = re.search(r"{\s*\"score\":.*?\s*}", extraction_content, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                try: json_data = json.loads(json_string)
                except json.JSONDecodeError as je: print(f"ERROR LLM Extractor: Could not parse extracted JSON: '{json_string}'. Error: {je}")
            else:
                clean_extraction_content = re.sub(r"```json\n?([\s\S]*?)\n?```", r"\1", extraction_content.strip(), flags=re.DOTALL)
                if clean_extraction_content.startswith("{") and clean_extraction_content.endswith("}"):
                    try: json_data = json.loads(clean_extraction_content)
                    except json.JSONDecodeError as je_clean: print(f"ERROR LLM Extractor (cleaned) invalid JSON. Cleaned: '{clean_extraction_content}'. Error: {je_clean}")
                else: print(f"ERROR LLM Extractor: No recognizable JSON block. Full Output: {extraction_content}")
            if json_data:
                if score is None and "score" in json_data and json_data["score"] is not None:
                    try: score = float(json_data["score"]); print(f"  LLM Extracted Score: {score}")
                    except (ValueError, TypeError): print(f"WARN LLM Extractor: Non-numeric score: {json_data.get('score')}")
                if verdict == "N/A" and "verdict" in json_data and json_data["verdict"] not in [None, "N/A", ""]:
                    verdict_llm = json_data.get("verdict", "N/A").strip()
                    if "approved with revisions" in verdict_llm.lower() or "revisions" in verdict_llm.lower(): verdict = "Approved with Revisions"
                    elif "approved" in verdict_llm.lower(): verdict = "Approved"
                    elif "rejected" in verdict_llm.lower(): verdict = "Rejected"
                    else: verdict = verdict_llm
                    print(f"  LLM Extracted Verdict: {verdict}")
        except Exception as e: print(f"ERROR during LLM Extractor call for '{val_prop_title_parsed_from_header}': {e}")

    if score is None: print(f"WARN FINAL: Score NOT parsed for '{val_prop_title_parsed_from_header}' by {validator_theme}.")
    if verdict == "N/A": print(f"WARN FINAL: Verdict NOT parsed for '{val_prop_title_parsed_from_header}' by {validator_theme}.")

    just_match = re.search(r"(?i)5\.\s*\*\*Justification for Verdict(?:.*?):\*\*\s*(.*)", validation_md, re.DOTALL)
    if just_match: justification = just_match.group(1).strip()
    else:
        just_match_alt = re.search(r"(?i)\*\*Justification for Verdict(?:.*?):\*\*\s*(.*)", validation_md, re.DOTALL)
        if just_match_alt: justification = just_match_alt.group(1).strip()

    print(f"FINAL PARSED (Val {validator_theme}) - For Orig Title '{original_proposal_title_expected}': Validator Assessed Title '{val_prop_title_parsed_from_header}', Parsed Score={score}, Parsed Verdict='{verdict}'. Justification found: {justification != 'N/A'}")
    assessments.append({"validated_proposal_title_text": val_prop_title_parsed_from_header, "original_input_title_for_validator": original_proposal_title_expected, "validator_theme": validator_theme, "score_given": score, "verdict_text": verdict, "justification_text": justification, "full_validation_markdown": validation_md})
    return assessments


@app.route('/api/enhance-standard', methods=['POST'])
def enhance_standard_api():
    # --- MODIFICATION START ---
    # Always process FAS32, ignoring any standardKey provided in the request
    std_key = "FAS32"
    print(f"API: Ignoring request body standardKey. FORCING standardKey='{std_key}' (FAS 32)")
    # --- MODIFICATION END ---

    std_info = STANDARD_FILES.get(std_key);
    if not std_info:
        # This case should theoretically not happen if FAS32 is in STANDARD_FILES
        # but kept as a safeguard.
        return jsonify({"error": f"Internal Server Error: Could not find configuration for hardcoded standard key '{std_key}'"}), 500

    std_text = load_standard_from_file(std_info["path"])
    if not std_text: return jsonify({"error": f"Could not load {std_info['name']} file from disk."}), 500

    print(f"API: Processing {std_info['name']} ({std_key})")

    try:
        analysis_res = analyzer_agent.invoke({"standard_name": std_info["name"], "standard_text": std_text, "standard_short_name": std_info["short_name"]})
        analysis_output = analysis_res.content if hasattr(analysis_res, 'content') else str(analysis_res); print("API: Analyzer complete.")

        # Use standard short name for news query
        news_q = f'("{std_info["short_name"]}" OR "Islamic Finance") AND (AI OR technology OR fintech)'; print(f"GNews Query: {news_q}")
        news_text = fetch_finance_news_gnews(query=news_q); print("API: News complete.")

        all_proposals_structured, raw_proposer_outputs_for_ui = [], {}
        for theme_k_internal, agent_c_config in THEMATIC_AGENT_TYPES.items():
            proposer_display_name = agent_c_config['name']
            print(f"API: Proposer {proposer_display_name} running...")
            prop_res = proposer_agent.invoke({"standard_name": std_info["name"], "analysis_output": analysis_output,"standard_text_snippet": std_text[:7000], "standard_short_name": std_info["short_name"], "theme_description": agent_c_config["description"], "finance_news": news_text})
            prop_md = (prop_res.content if hasattr(prop_res,'content') else str(prop_res)).strip(); raw_proposer_outputs_for_ui[proposer_display_name] = prop_md
            parsed = parse_proposals_from_markdown(prop_md, proposer_display_name); all_proposals_structured.extend(parsed); print(f"API: Proposer {proposer_display_name} done. Found {len(parsed)} proposals.")

        results_by_prop_id = {p['id']: {"original_proposal": p, "validations_by_theme": {}, "average_score": None, "all_scores": [], "meta_review": None} for p in all_proposals_structured}
        raw_val_outputs_for_ui_display = {}

        for theme_k_v_internal, agent_c_v_config in THEMATIC_AGENT_TYPES.items(): # theme_k_v_internal = "Foundations_Compliance"
            validator_display_name = agent_c_v_config['name'] # "Foundations & Compliance"
            print(f"API: Validator {validator_display_name} running (per proposal)...")
            concatenated_raw_validations_for_this_theme = []
            if not all_proposals_structured: raw_val_outputs_for_ui_display[validator_display_name] = "No proposals."; continue
            for original_prop_item in all_proposals_structured:
                single_proposal_md_for_validator = f"## {original_prop_item['title']}\n{original_prop_item['full_text_markdown']}"
                val_res = thematic_validator_agent.invoke({"standard_name": std_info["name"], "analysis_output": analysis_output, "single_proposal_markdown": single_proposal_md_for_validator, "standard_short_name": std_info["short_name"], "theme_description": agent_c_v_config["description"], "finance_news": news_text})
                val_md_for_one_proposal_raw = (val_res.content if hasattr(val_res,'content') else str(val_res)).strip()
                concatenated_raw_validations_for_this_theme.append(val_md_for_one_proposal_raw)
                assessments = parse_validations_with_scores(val_md_for_one_proposal_raw, validator_display_name, original_prop_item['title'])
                if assessments:
                    assessment_for_this_prop = assessments[0]
                    matched_id = original_prop_item['id']
                    results_by_prop_id[matched_id]['validations_by_theme'][validator_display_name] = assessment_for_this_prop
                    if assessment_for_this_prop['score_given'] is not None: results_by_prop_id[matched_id]['all_scores'].append(assessment_for_this_prop['score_given'])
                else: print(f"    WARN Val-{validator_display_name}: No assessment parsed for prop '{original_prop_item['title']}'. Raw: {val_md_for_one_proposal_raw[:100]}")
            raw_val_outputs_for_ui_display[validator_display_name] = "\n\n---\n\n".join(concatenated_raw_validations_for_this_theme)
            print(f"API: Validator {validator_display_name} done.")

        for p_id in results_by_prop_id:
            scores = results_by_prop_id[p_id]['all_scores']
            results_by_prop_id[p_id]['average_score'] = round(sum(scores)/len(scores),2) if scores else "N/A"
            avg_s_disp = results_by_prop_id[p_id]['average_score']
            orig_p_item = results_by_prop_id[p_id]['original_proposal']
            val_summary_for_meta = "" # Renamed for clarity
            for validator_disp_name_key, val_detail in results_by_prop_id[p_id]['validations_by_theme'].items():
                just_snippet = val_detail.get('justification_text', 'N/A')
                if just_snippet != 'N/A' and len(just_snippet) > 250: just_snippet = just_snippet[:250] + "..." # Provide more justification text
                val_summary_for_meta += f"\n- Validator ({validator_disp_name_key}): Score={val_detail.get('score_given','N/A')}, Verdict='{val_detail.get('verdict_text','N/A')}', Justification Snippet='{just_snippet}'"

            if not val_summary_for_meta.strip() and avg_s_disp == "N/A":
                results_by_prop_id[p_id]['meta_review'] = {"status": "Review Error", "justification": "No validation data (scores, verdicts, or justifications) parsed from thematic validators.", "raw_meta_output":""}; continue

            print(f"API: Meta-Reviewer for ID: {p_id} ('{orig_p_item['title']}')")
            # print(f"DEBUG MetaReviewer Input Summary:\n{val_summary_for_meta.strip() if val_summary_for_meta else 'No validation details.'}") # Uncomment for heavy debug
            meta_res = meta_reviewer_agent.invoke({"proposal_title": orig_p_item['title'], "proposal_full_text_body": orig_p_item['full_text_markdown'], "average_score": avg_s_disp, "detailed_validations_summary_for_meta": val_summary_for_meta.strip() if val_summary_for_meta else "No specific scores/verdicts/justifications available from thematic validators."})
            meta_out = (meta_res.content if hasattr(meta_res,'content') else str(meta_res)).strip()
            stat_m = re.search(r'\*\*Overall Status:\*\*\s*(.*)', meta_out, re.IGNORECASE)
            just_m = re.search(r'\*\*Consolidated Justification:\*\*\s*(.*)', meta_out, re.DOTALL | re.IGNORECASE)
            results_by_prop_id[p_id]['meta_review'] = {"status": stat_m.group(1).strip() if stat_m else "Meta Status Parse Err", "justification": just_m.group(1).strip() if just_m else "Meta Justification Parse Err", "raw_meta_output": meta_out}

        final_list_ui = list(results_by_prop_id.values())
        combo_prop_ui = ""
        for th_disp_name, md in raw_proposer_outputs_for_ui.items():
            combo_prop_ui += f"\n\n--- Proposals from {th_disp_name} ---\n{md}"

        print("API: Multi-agent processing complete.")
        return jsonify({"standard": std_info, "analysis_markdown": analysis_output, "proposals_combined_markdown_for_ui": combo_prop_ui.strip(), "validator_raw_outputs_by_theme_for_ui": raw_val_outputs_for_ui_display, "final_enhancement_proposals_with_validations": final_list_ui, "debug_news_fetched": news_text})
    except Exception as e:
        print(f"API TopLvl Err: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred: {e}"}), 500 # Changed error message for user facing API

if __name__ == '__main__':
    print(f"Starting AAOIFI Backend (Using LLM Provider: {LLM_PROVIDER})...")
    if LLM_PROVIDER == "OPENAI" and not os.getenv("OPENAI_API_KEY"): print("CRITICAL: OPENAI_API_KEY missing."); exit(1)
    if LLM_PROVIDER == "GROQ" and not os.getenv("GROQ_API_KEY"): print("CRITICAL: GROQ_API_KEY missing."); exit(1)
    if not GNEWS_API_KEY: print("\n!! WARN: GNEWS_API_KEY missing !!\n")
    st_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'standards')
    if not os.path.exists(st_dir): os.makedirs(st_dir); print(f"Created '{st_dir}'.")
    for k, si_main in STANDARD_FILES.items():
        fp_main = os.path.join(os.path.dirname(os.path.abspath(__file__)), si_main["path"])
        if not os.path.exists(fp_main): print(f"WARN: Standard file '{si_main['path']}' for {si_main['name']} not found.")
    print(f"Flask server starting on http://127.0.0.1:5001"); app.run(debug=True, port=5001)