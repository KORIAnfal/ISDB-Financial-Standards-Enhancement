import os
from dotenv import load_dotenv
# import requests # No longer needed for news fetching from API
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
from collections import defaultdict # Useful for grouping scores
import statistics # For calculating the mean score
import time # Import time for sleep

# Import specific Groq/OpenAI API error for handling
try:
    # Attempt to import Groq RateLimitError
    from groq import RateLimitError as GroqRateLimitError
    # langchain_openai often maps Groq errors to OpenAI's RateLimitError
    from openai import RateLimitError as OpenAIRateLimitError
    # Define a tuple to catch either specific error
    RateLimitError = (GroqRateLimitError, OpenAIRateLimitError)
    print("Info: Imported specific RateLimitError types.")
except ImportError:
    # Fallback to generic exception if Groq or OpenAI client isn't fully installed
    print("Warning: Could not import specific RateLimitError types (groq/openai). Using generic Exception for retry logic.")
    RateLimitError = Exception # Catch all exceptions for retry


load_dotenv()

# --- Configuration ---
# Set the model name to a Groq model name.
# Choose one based on your Groq account access and needs.
# llama3-8b-8192 (8k context), llama3-70b-8192 (32k context), mixtral-8x7b-32768 (32k context)
# We are optimizing for smaller validator context by summarizing inputs.
# Let's use the 70b model which has a larger context window (32k),
# which gives more room for the summarized proposals and analysis.
LLM_MODEL_NAME = "llama3-70b-8192" # Using the larger Llama 3 model (32k context)


# Groq API Base URL (OpenAI compatible endpoint)
GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Retry settings for rate limits
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1.0 # seconds

# Max characters for standard text snippet passed to agents (helps control input size)
# Analyzer gets full text. Proposers get snippet. Validators get NO snippet in this version.
MAX_SNIPPET_CHARS = 4000


# --- Function to load standard text from a file (Keep this) ---
def load_standard_from_file(file_path):
    try:
        # Adjust path for clarity or if files are in a subdir
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found at {full_path}")
        return None
    except Exception as e:
        print(f"ERROR: Could not read file {full_path}: {e}")
        return None

# --- Paths to the standard text files (Add FAS 7, FAS 28) ---
STANDARD_FILES = {
    "FAS4": {
        "name": "AAOIFI Financial Accounting Standard No. 4 - Musharaka Financing",
        "short_name": "Musharaka (FAS 4)",
        "path": "standards/fas4_musharaka.txt" # Assuming standards are in a 'standards' subdirectory
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
    # Add new standards
    "FAS7": {
        "name": "AAOIFI Financial Accounting Standard No. 7 - Salam and Parallel Salam",
        "short_name": "Salam (FAS 7)",
        "path": "standards/fas7_salam.txt" # Create this file
    },
    "FAS28": {
        "name": "AAOIFI Financial Accounting Standard No. 28 - Investments in Real Estate",
        "short_name": "Real Estate (FAS 28)",
        "path": "standards/fas28_realestate.txt" # Create this file
    }
}

# --- Define Agent Personas/Types ---

# Proposer Agent Personas - Each will generate proposals from their perspective based on Analyzer's news-driven gaps
PROPOSER_PERSONAS = {
    "Accountant_Auditor_Proposer": {
        "name": "Islamic Finance Accountant/Auditor",
        "description": "Focuses on practical accounting treatment, recognition, measurement, journal entries, and disclosure requirements."
    },
    "Shariah_Scholar_Proposer": {
        "name": "Shari’ah Scholar",
        "description": "Focuses on the Shari’ah basis, justification, and alignment of proposals."
    },
    "Academic_Legal_Proposer": {
        "name": "Academic/Legal Expert",
        "description": "Focuses on theoretical soundness, legal implications, and alignment with academic principles."
    },
    "Regulatory_Industry_Proposer": {
         "name": "Regulator/Industry Representative",
         "description": "Focuses on practical implementability, regulatory compliance, and industry best practices."
    }
}

# Validator Agent Personas - A fixed set that score all proposals based on specific criteria
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


# --- Initialize LLM ---
# Initialize ChatOpenAI instances pointing to Groq API
# They use the GROQ_API_KEY environment variable which now holds the Groq key
llm_creative = ChatOpenAI(
    model_name=LLM_MODEL_NAME, # Using the specified Groq model name
    temperature=0.4,
    openai_api_key=os.getenv("GROQ_API_KEY"), # Read Groq key from this variable
    base_url=GROQ_API_BASE # Point to Groq API
)

llm_structured = ChatOpenAI(
    model_name=LLM_MODEL_NAME, # Using the specified Groq model name
    temperature=0.1,
    openai_api_key=os.getenv("GROQ_API_KEY"), # Read Groq key from this variable
    base_url=GROQ_API_BASE # Point to Groq API
)


# --- AGENT PROMPT TEMPLATES ---

# Analyzer Agent (Reviewer - MODIFIED to focus analysis on PROVIDED NEWS GAPS)
analyzer_template_str = """
**You are an Expert AAOIFI Standard Analyst (Reviewer).** Your primary task is to meticulously analyze the provided AAOIFI Standard text and the provided finance news to identify existing elements and pinpoint areas needing clarification or enhancement *within that specific standard, specifically focusing on issues or opportunities for enhancement highlighted by the news*. Your output should bridge the gap between the standard text and the news. Be precise and direct.

**Standard Name:** '{standard_name}'
**Specific Task:** Review THE FOLLOWING STANDARD TEXT for '{standard_name}' and the provided Finance News Text. Perform the analysis and identification tasks below, focusing on how the news indicates potential areas for enhancement in the standard.

**PROVIDED STANDARD TEXT for '{standard_name}':**
---
{standard_text}
---

**PROVIDED FINANCE NEWS TEXT:**
---
{finance_news}
---

**Instructions - Perform these actions *directly* on the standard text provided above, informed by the PROVIDED FINANCE NEWS TEXT. Be concise and direct:**
1.  **Core Summary:** *Based on the provided standard text*, summarize the main purpose, scope, and core principles of this standard ('{standard_short_name}').
2.  **Key Elements Extraction:** *From the provided standard text*, systematically extract and list:
    *   Major sections/topics and their primary rules (e.g., Scope, Definitions, Recognition, Measurement, Disclosure).
    *   Critical definitions *explicitly stated or clearly implied for '{standard_short_name}' in the text*.
    *   Core accounting treatment principles (initial/subsequent recognition & measurement for assets, liabilities, revenue, expenses) *as detailed in the text for '{standard_short_name}'*.
    *   Any specific conditions or criteria *mentioned in the text* for applying certain rules for '{standard_short_name}'.
3.  **Identified Areas for Enhancement *Specifically Highlighted by the Provided NEWS TEXT*:**
    *   **Relevance of News:** Briefly explain how the provided news text is relevant to this specific standard ('{standard_short_name}'), identifying the key trends or topics mentioned in the news that could impact the standard.
    *   **Specific Impact/Gaps highlighted by News:** Based on the *provided news text*, identify specific areas (sections, rules, lack of guidance) *within the provided standard text* that might need enhancement, clarification, or updating *because of the trends or issues mentioned in the news*. For each identified area, provide:
        *   A brief description of the potential issue or need.
        *   Reference to relevant sections in the *standard text* if applicable.
        *   Reference to the part of the *news text* that highlights this issue.

**Output Format:**
Present your findings as a structured markdown report. Be precise and direct. Focus section 3 *entirely* on news-driven enhancement areas.
"""
analyzer_prompt = ChatPromptTemplate.from_template(analyzer_template_str)
# LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
analyzer_agent = LLMChain(llm=llm_structured, prompt=analyzer_prompt)


# Proposer Agent (Persona-based - Creates proposals based on Analyzer's NEWS-DRIVEN gaps)
proposer_template_str = """
**You are an Innovative Islamic Finance & AI Strategist, acting as an {persona_name} ({persona_description}).** Your task is to propose specific, actionable enhancements or clarifications for the AAOIFI Standard: '{standard_name}' ('{standard_short_name}'). You will focus specifically on the *Identified Areas for Enhancement Specifically Highlighted by the Provided NEWS TEXT* from the Analyzer's report. Use the Analyzer's findings, the standard text, and the provided news. Propose 1-3 distinct, concrete modifications *relevant to your role ({persona_name}) and the identified news-driven gaps*.

**Standard Name:** '{standard_name}' ('{standard_short_name}')

**Input: Analysis of '{standard_name}' from the Standard Analyst:**
---
{analysis_output}
---

**Input: Original Standard Text Snippet (for contextual reference regarding '{standard_short_name}'):**
---
{standard_text_snippet}
---

**Input: Provided Finance News Text:**
---
{finance_news}
---

**Instructions - Based on the Analyst's *Identified Areas for Enhancement Specifically Highlighted by the Provided NEWS TEXT*, standard context, and provided news. Focus as an {persona_name}. Be concise and direct:**
Address the specific enhancement areas identified by the Analyst based on the news. For *each* proposal, provide the following details from your perspective as an {persona_name}:

1.  **Proposal Title:** Concise title.
2.  **Proposed Change Description:** Clear, specific modification/addition to the standard text for '{standard_short_name}'.
3.  **Relevant Original Section(s):** Indicate the existing section(s) of the standard text ('{standard_short_name}') that this proposal would amend, extend, or where a new section would logically fit. Reference section numbers if available in the text.
4.  **Rationale (linking to analysis/news):** Explain *why* this specific enhancement is necessary, explicitly referencing the *news-driven gaps identified by the Analyst* and relevant insights from the 'Provided Finance News Text' that support this proposal. Frame this from your perspective as an {persona_name}.
5.  **Proposal Details (from {persona_name} Perspective):** Provide details relevant to your role, including (where applicable to this proposal and relevant to the news-driven gap):
    *   Accounting principles affected or introduced.
    *   Recognition & measurement rules impacted.
    *   Implications for Journal entries.
    *   Requirements for Disclosure.
    *   Shari’ah basis and justification.
    *   Implications for Regulators, Financial Institutions, or Audit Firms.
    *   AI/Technology Link.
6.  **Expected Benefit:** Positive outcome for IFIs applying '{standard_short_name}', framed by your role.

**Output Format:** Structure using markdown.

---
## Proposed Enhancement: [Proposal Title]
**Proposed Change Description:** ...
**Relevant Original Section\(s\):** ...
**Rationale:** ... (Link clearly to Analyst's news-driven gap and news text)
**Details from {persona_name} Perspective:**
*   Accounting principles: ... (If relevant)
*   Recognition & measurement: ... (If relevant)
*   Journal entries: ... (If relevant)
*   Disclosure: ... (If relevant)
*   Shari’ah basis/justification: ... (If relevant)
*   Regulatory/Industry implications: ... (If relevant)
*   AI/Technology Link: ... (If relevant)
**Expected Benefit:** ... (From {persona_name} standpoint)
---
... (Continue for 1-3 proposals relevant to news-driven gaps) ...
"""
proposer_prompt = ChatPromptTemplate.from_template(proposer_template_str)
# LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
proposer_agent = LLMChain(llm=llm_creative, prompt=proposer_prompt) # Use creative LLM for proposing


# Summarizer Agent - Takes raw proposals and produces a concise summary list
# MODIFIED prompt to simplify input structure expected
summarizer_template_str = """
**You are an AAOIFI Proposal Summarizer.** Your task is to review the proposed enhancements provided below. For each distinct proposed enhancement, create a concise summary entry containing *only* the Proposal Title and a brief description of the proposed change.

**Standard Name:** '{standard_name}' ('{standard_short_name}')

**Input: Proposed Enhancements from Proposers (Simplified Markdown):**
---
{all_proposals_simplified_markdown}
---

**Instructions:**
Review the provided list of proposals (Title and Description only). Identify each unique proposal. For each unique proposal, create a concise summary entry.

**Output Format:**
Provide a list of proposal summaries using markdown.

---
## Summarized Proposals for '{standard_name}'
*   **[Proposal Title 1]:** [Concise Description of Proposed Change 1]
*   **[Proposal Title 2]:** [Concise Description of Proposed Change 2]
*   ... (Continue for all unique proposals found in the input)
---
"""
summarizer_prompt = ChatPromptTemplate.from_template(summarizer_template_str)
# LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
summarizer_agent = LLMChain(llm=llm_structured, prompt=summarizer_prompt)


# Validator Agent (Specific - Reviews ALL proposals and gives a SCORE out of 100)
# MODIFIED prompt to reflect reduced input context and REMOVE {finance_news} placeholder
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

{{!-- REMOVED: No longer providing full news text to validators for brevity --}}
{{!--
**Input: Provided Finance News Text (for context):**
---
{finance_news}
---
--}}


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

# Define concrete validator agents based on the new list
# LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
technical_validator_agent = LLMChain(llm=llm_structured, prompt=validator_prompt)
# LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
shariah_validator_agent = LLMChain(llm=llm_structured, prompt=validator_prompt)
# LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
global_consistency_validator_agent = LLMChain(llm=llm_structured, prompt=validator_prompt)
# LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
clarity_auditability_validator_agent = LLMChain(llm=llm_structured, prompt=validator_prompt)


VALIDATOR_AGENTS = {
    "Technical_Validator": {
        "agent": technical_validator_agent,
        "persona": VALIDATOR_PERSONAS["Technical_Validator"]
    },
    "Shariah_Validator": {
        "agent": shariah_validator_agent,
        "persona": VALIDATOR_PERSONAS["Shariah_Validator"]
    },
    "Global_Consistency_Validator": {
         "agent": global_consistency_validator_agent,
         "persona": VALIDATOR_PERSONAS["Global_Consistency_Validator"]
    },
    "Clarity_Auditability_Validator": {
         "agent": clarity_auditability_validator_agent,
         "persona": VALIDATOR_PERSONAS["Clarity_Auditability_Validator"]
    }
}


# --- Helper Functions for Parsing LLM Output ---

# Keep parse_proposals_markdown to parse raw proposer outputs for aggregation
def parse_proposals_markdown(markdown_text):
    """
    Parses markdown output from Proposer agents, extracting proposal details.
    Handles potential variations in headings like '## Proposal 1 Title' or '## Proposed Enhancement: Title'.
    """
    proposals = []
    # Regex to find proposal sections starting with ## followed by anything, non-greedily
    # Assumes each proposal starts with a Level 2 markdown heading (##)
    # Adjusted regex to be more specific about the title pattern
    proposal_sections = re.split(r'##\s*Proposed Enhancement:\s*(.+)', markdown_text, re.DOTALL)

    # The first element of split is usually before the first ##, ignore if empty
    # Also handle cases where the heading might just be ## Title (less likely with the prompt, but defensive)
    if not proposal_sections or (len(proposal_sections) > 0 and proposal_sections[0].strip() == ''):
         # Try splitting by a simpler ## followed by anything if the first split failed
         temp_sections = re.split(r'##\s*(.+)', markdown_text, re.DOTALL)
         if not temp_sections or (len(temp_sections) > 0 and temp_sections[0].strip() == ''):
              proposal_sections = temp_sections[1:]
         else: # Use the result of the first split if it had non-empty content before the first match
             proposal_sections = proposal_sections[1:]


    # Process in pairs (Title line, Content block) from the potentially adjusted split
    # Need to be careful if the simpler split also resulted in empty starting element
    processed_sections = []
    if proposal_sections and proposal_sections[0].strip() == '':
        processed_sections = proposal_sections[1:]
    else:
        processed_sections = proposal_sections

    for i in range(0, len(processed_sections), 2):
        if i+1 < len(processed_sections):
            # The title line is processed_sections[i]
            title = processed_sections[i].strip() # Use the captured group as title

            content = processed_sections[i+1].strip()

            proposal_data = {"title": title}
            # Basic extraction of key-value pairs within the proposal content
            # Make patterns non-greedy .*? and look for end of line or next markdown key (e.g., **Something:)
            # Updated regex to be more robust across different fields
            desc_match = re.search(r'\*\*Proposed Change Description:\*\*\s*(.*?)(?=\n\*\*[^:]+?:|$)', content, re.DOTALL)
            if desc_match: proposal_data["description"] = desc_match.group(1).strip()

            sections_match = re.search(r'\*\*Relevant Original Section\(s\):\*\*\s*(.*?)(?=\n\*\*[^:]+?:|$)', content, re.DOTALL)
            if sections_match: proposal_data["relevantSections"] = sections_match.group(1).strip()

            rationale_match = re.search(r'\*\*Rationale\s*\(linking to analysis/news\):\*\*\s*(.*?)(?=\n\*\*[^:]+?:|$)', content, re.DOTALL) # Look for specific Rationale title
            if not rationale_match: # Fallback if Rationale title is simpler
                 rationale_match = re.search(r'\*\*Rationale:\*\*\s*(.*?)(?=\n\*\*[^:]+?:|$)', content, re.DOTALL)
            if rationale_match: proposal_data["rationale"] = rationale_match.group(1).strip()

            # Capture the entire "Details from X Perspective" block
            details_match = re.search(r'\*\*Details from (.+?) Perspective:\*\*\s*(.*?)(?=\n\*\*Expected Benefit:|$)', content, re.DOTALL)
            if details_match:
                # Store the entire markdown block under a key
                proposal_data["personaDetailsMarkdown"] = details_match.group(2).strip()
                # Optionally, try to extract specific points if needed later, but keeping it as raw markdown is safer initially
                # For now, just capture the block

            ai_link_match = re.search(r'\*\*AI/Technology Link\s*\(linking to news\):\*\*\s*(.*?)(?=\n\*\*[^:]+?:|$)', content, re.DOTALL) # Look for specific AI Link title
            if not ai_link_match: # Fallback
                 ai_link_match = re.search(r'\*\*AI/Technology Link:\*\*\s*(.*?)(?=\n\*\*[^:]+?:|$)', content, re.DOTALL)
            if ai_link_match: proposal_data["aiTechnologyLink"] = ai_link_match.group(1).strip()


            benefits_match = re.search(r'\*\*Expected Benefit:\*\*\s*(.*?)(?=\n##|$)', content, re.DOTALL) # Benefits until next ## or end
            if benefits_match: proposal_data["expectedBenefits"] = benefits_match.group(1).strip()

            proposals.append(proposal_data)
    return proposals


def parse_validator_scores_and_justifications(markdown_text, validator_name):
    """
    Parses a single specific validator's markdown output, extracting proposal titles, scores, and justifications.
    Scores are expected 0-100 and are normalized to 0.0-1.0.
    """
    validations = []
    # Regex to find validation sections based on the expected markdown structure
    # Looking for "## Validation(?: by [Validator Name])? for: [Proposal Title]" or "## Validation for: [Proposal Title]"
    # The regex captures (?: by (.*?))? to optionally get the validator name from the text,
    # and (.*) to get the proposal title.
    # Adjusted regex to match the expected "Validation by {persona_name} for:" structure
    validation_sections = re.split(r'##\s*Validation by .+? for: (.*)', markdown_text, re.DOTALL)

    # Expected structure after split: ['', proposal_title, content, proposal_title, content, ...]
    if not validation_sections or validation_sections[0].strip() == '':
        validation_sections = validation_sections[1:] # Remove empty start

    # Process in pairs (Proposal Title, Content)
    for i in range(0, len(validation_sections), 2):
        if i + 1 < len(validation_sections):
            title = validation_sections[i].strip() # This is the Proposal Title being validated
            content = validation_sections[i+1].strip()

            validation_data = {
                "proposalTitle": title, # Use the exact title from the validator's output
                "validatingValidator": validator_name, # Use the known validator name from config
                "score": None, # Initialize score (will store normalized 0.0-1.0)
                "rawScore": None, # Store the raw 0-100 score
                "assessment": None, # Capture the assessment text
                "justification": None,
                "individualVerdict": None # Capture any individual verdict if present
            }

            # Extract the Assessment (comes after title, before Score or Justification)
            # Adjusted regex to be more flexible for different validator names in the assessment header
            assessment_match = re.search(r'\*\*Assessment \(.+?\s*Perspective\):\*\*\s*(.*?)(?=\n\*\*Score:|\n\*\*Justification:|$)', content, re.DOTALL) # Try specific assessment header
            if not assessment_match: # Fallback to simpler Assessment header
                 assessment_match = re.search(r'\*\*Assessment:\*\*\s*(.*?)(?=\n\*\*Score:|\n\*\*Justification:|$)', content, re.DOTALL)
            if assessment_match:
                 validation_data["assessment"] = assessment_match.group(1).strip()


            # Extract the Score (looking for a number, possibly with decimal, up to 100)
            score_match = re.search(r'\*\*Score:\*\*\s*([\d.]+)', content) # Look for "**Score:** " followed by numbers/decimal
            if score_match:
                try:
                    raw_score = float(score_match.group(1).strip())
                    # Clamp raw score between 0 and 100
                    raw_score = max(0.0, min(100.0, raw_score))
                    validation_data["rawScore"] = raw_score
                    # Normalize score to 0.0-1.0 scale for mean calculation
                    validation_data["score"] = raw_score / 100.0
                except ValueError:
                    print(f"Warning: Could not parse score as float for proposal '{title}' validated by '{validator_name}'. Content snippet: {content[:100]}...")
                    validation_data["score"] = None # Keep as None if parsing fails
                    validation_data["rawScore"] = None # Also clear raw score

            # Extract Justification (comes after Score if present, or after Assessment)
            # Adjusted regex to be more flexible for different validator names in the justification header
            justification_match = re.search(r'\*\*Justification \(.+?\s*Perspective\):\*\*\s*(.*?)(?=\n##|$)', content, re.DOTALL) # Try specific justification header
            if not justification_match: # Fallback to simpler justification header
                 justification_match = re.search(r'\*\*Justification:\*\*\s*(.*?)(?=\n##|$)', content, re.DOTALL)
            if justification_match:
                validation_data["justification"] = justification_match.group(1).strip()

            # Extract individual verdict if present (Optional based on prompt, but captured defensively)
            individual_verdict_match = re.search(r'\*\*Overall Verdict:\*\*\s*(.*?)(?:\s*\([^)]*\))?\n', content)
            if individual_verdict_match: validation_data["individualVerdict"] = individual_verdict_match.group(1).strip()

            validations.append(validation_data)
    return validations


# --- Function to Aggregate Scores and Determine Final Verdicts ---

def aggregate_scores_and_verdicts(proposer_outputs_by_persona, validator_outputs_by_name, proposer_personas, validator_personas):
    """
    Aggregates proposals from all proposers, collects scores for each proposal
    from specific validators, calculates mean scores, and determines final verdicts.
    """
    all_proposals_dict = {} # Dict to store unique proposals {title: {details}}

    # 1. Collect all proposed enhancements and identify unique ones (using title as key)
    # This uses the parse_proposals_markdown function
    for persona_key, proposals_markdown in proposer_outputs_by_persona.items():
        parsed_proposals = parse_proposals_markdown(proposals_markdown)
        persona_name = proposer_personas.get(persona_key, {}).get("name", persona_key)
        for proposal in parsed_proposals:
            # Use title as the primary key for uniqueness
            # Clean up title potentially generated differently by LLM
            title = proposal.get("title", f"Untitled Proposal from {persona_name}").strip()
            # Basic cleaning: remove potential trailing markdown
            title = re.sub(r'\*\*:?$', '', title).strip()


            if title not in all_proposals_dict:
                 # Store the first instance found, add source persona
                 all_proposals_dict[title] = proposal
                 all_proposals_dict[title]["sourcePersonas"] = [persona_name]
            else:
                 # If already exists, just note the additional source persona
                 source_name = proposer_personas.get(persona_key, {}).get("name", persona_key)
                 if source_name not in all_proposals_dict[title].get("sourcePersonas", []):
                      all_proposals_dict[title].setdefault("sourcePersonas", []).append(source_name)

    # 2. Collect all validation results (including scores) for each proposal, grouped by proposal title
    validations_by_proposal_title = defaultdict(list) # {proposal_title: [list of validation_data]}

    for validator_key, validations_markdown in validator_outputs_by_name.items():
        validator_name = validator_personas.get(validator_key, {}).get("name", validator_key)
        # Only process if the validator actually ran and produced output that isn't just the default empty message
        if validations_markdown and not validations_markdown.startswith("No proposals were generated") and not validations_markdown.startswith("No proposals were generated or summarized"):
            parsed_validations = parse_validator_scores_and_justifications(validations_markdown, validator_name)
            for validation in parsed_validations:
                # Ensure the proposal title from validation matches one we collected
                # Clean up validation proposal title for matching
                validation_proposal_title = validation.get("proposalTitle", "").strip()
                validation_proposal_title = re.sub(r'\*\*:?$', '', validation_proposal_title).strip()


                # Find the matching title in our collected proposals dictionary keys
                # Use a case-insensitive or fuzzy match here if exact string matching proves brittle
                # For now, using exact match on the cleaned title
                if validation_proposal_title in all_proposals_dict:
                    validations_by_proposal_title[validation_proposal_title].append(validation)
                else:
                    # Log if a validation refers to a proposal title we didn't find
                    print(f"Warning: Validation from '{validator_name}' found for unknown proposal title: '{validation_proposal_title}'. Skipping.")


    # 3. Calculate mean scores and determine final verdict for each unique proposal
    final_proposals_list = []
    num_validators_configured = len(VALIDATOR_AGENTS) # Total number of *specific* validators

    for proposal_title, proposal_details in all_proposals_dict.items():
        # Get scores for this proposal from *all* specific validators that provided a score
        validations_for_this_proposal = validations_by_proposal_title.get(proposal_title, [])
        scores_normalized = [v['score'] for v in validations_for_this_proposal if v.get('score') is not None]

        # Calculate mean score on the normalized scores (0.0-1.0)
        mean_score_normalized = statistics.mean(scores_normalized) if scores_normalized else 0.0
        # Convert mean score back to 0-100 scale for display and comparison
        mean_score_hundred = round(mean_score_normalized * 100.0, 2)

        num_scores_received = len(scores_normalized) # Count only valid numerical scores received


        # Determine final verdict based on mean score > 90
        final_verdict = "Rejected" # Default
        # Initial justification using the mean score out of 100
        final_justification = f"Mean score ({mean_score_hundred:.2f}/100) is below the threshold (90/100)."

        if mean_score_hundred > 90.0 and num_scores_received > 0: # Must exceed 90 and have at least one score
            final_verdict = "Approved"
            final_justification = f"Mean score ({mean_score_hundred:.2f}/100) meets or exceeds the threshold (90/100)."
        elif num_scores_received == 0 and num_validators_configured > 0:
            final_verdict = "Undetermined"
            final_justification = f"No valid scores received from any validator out of {num_validators_configured} configured validators."
        elif mean_score_hundred <= 90.0 and num_scores_received > 0:
             # Prioritize rejection justification based on validator type if available and score is low (<= 90)
             # Find the validation with the lowest *raw* score <= 90 that has a justification
             low_scoring_validations = sorted([v for v in validations_for_this_proposal if v.get('rawScore', 100.0) <= 90.0 and v.get('justification')],
                                               key=lambda x: x.get('rawScore', 100.0)) # Sort by lowest raw score first

             if low_scoring_validations:
                 # Take the justification from the validator with the lowest score <= 90
                 main_concern_val = low_scoring_validations[0]
                 final_justification = f"Rejected due to {main_concern_val['validatingValidator']} concern: {main_concern_val['justification']}"
             else:
                  # Default rejection reason if specific low-score justification isn't clear
                  general_reasons = [v.get("justification", "No justification provided") for v in validations_for_this_proposal if v.get("justification")]
                  final_justification = f"Rejected due to low mean score ({mean_score_hundred:.2f}/100). " + (f"Reasons from validators: {'; '.join(general_reasons)}" if general_reasons else "No specific reasons provided by validators.")


        # Combine original proposal details with aggregated results
        final_proposal_data = {
            **proposal_details, # Include original title, description, rationale, etc.
            "sourcePersonas": proposal_details.get("sourcePersonas", []), # Which proposer personas suggested this idea
            "individualValidations": validations_for_this_proposal, # List of all validator results for this proposal
            "meanScore": mean_score_hundred, # Store mean score out of 100 for frontend
            "numScoresReceived": num_scores_received,
            "totalValidatorsConfigured": num_validators_configured,
            "finalVerdict": final_verdict,
            "finalVerdictJustification": final_justification
        }
        final_proposals_list.append(final_proposal_data)

    # Sort proposals (optional, e.g., by mean score descending)
    # Use a default score like -1 for Undetermined proposals to put them at the end
    # Need to handle None scores for sorting correctly - sort by 'meanScore' if it exists, otherwise put None values at the end (-1)
    final_proposals_list.sort(key=lambda x: (x.get('meanScore') is not None, x.get('meanScore', -1)), reverse=True)

    return final_proposals_list


# --- Helper function to invoke LLM agent with retries ---
def invoke_agent_with_retry(agent_chain, inputs, max_retries=MAX_RETRIES, initial_delay=INITIAL_RETRY_DELAY):
    """Invokes an LLMChain agent with retry logic for RateLimitError."""
    delay = initial_delay
    for i in range(max_retries):
        try:
            return agent_chain.invoke(inputs)
        except RateLimitError as e:
            print(f"Rate limit error encountered (Attempt {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                # Check if error message contains specific retry-after header info
                retry_after_match = re.search(r'Please try again in ([\d.]+)s', str(e))
                if retry_after_match:
                    suggested_delay = float(retry_after_match.group(1))
                    # Use the suggested delay if it's larger than our current delay
                    delay = max(delay, suggested_delay)
                    print(f"API suggested retry after {suggested_delay:.2f}s. Using delay {delay:.2f}s.")
                else:
                     print(f"No specific retry-after header found. Using default delay {delay:.2f}s.")

                time.sleep(delay)
                delay *= 2 # Exponential backoff
            else:
                print(f"Max retries ({max_retries}) reached. Failing.")
                raise # Re-raise the last exception
        except Exception as e:
            # Catch other potential errors during invoke
            print(f"An unexpected error occurred during agent invocation (Attempt {i+1}/{max_retries}): {e}")
            # For other errors, we don't necessarily want to retry unless it's a transient connection issue
            # For simplicity, we'll just re-raise them.
            raise


# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)


@app.route('/api/enhance-standard', methods=['POST'])
def enhance_standard_api():
    data = request.get_json()
    standard_key = data.get('standardKey')
    # Receive the news text directly from the frontend input
    user_news_text = data.get('userNewsText', '').strip()

    if not standard_key or standard_key not in STANDARD_FILES:
        return jsonify({"error": "Invalid standard key provided. Available: " + ", ".join(STANDARD_FILES.keys())}), 400

    standard_info = STANDARD_FILES[standard_key]
    standard_name = standard_info["name"]
    standard_short_name = standard_info["short_name"]
    standard_file_path = standard_info["path"]

    standard_text_content = load_standard_from_file(standard_file_path)
    if not standard_text_content:
        return jsonify({"error": f"Could not load standard text for {standard_name} from {standard_file_path}."}), 500

    # Require news text input for this workflow
    if not user_news_text:
         return jsonify({"error": "Finance news text must be provided."}), 400

    print(f"\n--- Processing Request for {standard_name} ---")
    print(f"Provided News Text:\n{user_news_text}")


    # Limit standard text snippet for proposer/validator context
    # Keep this reasonable to avoid exceeding context windows, especially for validators receiving combined info.
    max_snippet_chars = 4000

    standard_text_snippet = standard_text_content[:max_snippet_chars]

    # Initialize temp_all_proposals list *before* the proposer loop
    temp_all_proposals = []

    try:
        # News is provided by the user, no API fetch here.
        finance_news_text = user_news_text # Use the provided news text


        # 1. Analyzer Agent (Reviewer) - Analyze standard based on PROVIDED news gaps
        print("\n--- Step 1: Running Analyzer Agent (Identifiying News-driven Gaps) ---")
        # Use retry function for the agent invocation
        analysis_result = invoke_agent_with_retry(analyzer_agent, {
            "standard_name": standard_name,
            "standard_text": standard_text_content,
            "standard_short_name": standard_short_name,
            "finance_news": finance_news_text # Pass the provided news text to analyzer
        })
        analysis_output = analysis_result['text']
        print("\nAnalyzer Output:")
        print(analysis_output) # Print Analyzer output to terminal


        # 2. Multiple Proposer Agents (Persona-based) - Propose based on Analyzer's news-driven gaps
        print(f"\n--- Step 2: Running {len(PROPOSER_PERSONAS)} Proposer Agents (Persona-based) ---")
        proposer_outputs_by_persona = {}

        for persona_key, persona_config in PROPOSER_PERSONAS.items():
            print(f"\nRunning Proposer Agent: {persona_config['name']} ({persona_key})...")
            # Use retry function for the agent invocation
            proposals_result = invoke_agent_with_retry(proposer_agent, {
                "standard_name": standard_name,
                "analysis_output": analysis_output, # Provide analysis context (including news gaps)
                "standard_text_snippet": standard_text_snippet, # Provide standard text snippet context
                "standard_short_name": standard_short_name,
                "persona_name": persona_config["name"], # Pass the persona name
                "persona_description": persona_config["description"], # Pass the persona description
                "finance_news": finance_news_text # Pass the provided news text for context
            })
            proposer_output = proposals_result['text'].strip()
            proposer_outputs_by_persona[persona_key] = proposer_output # Store raw output by persona

            # --- MODIFIED: Parse proposals here and add to temp_all_proposals ---
            parsed_proposals = parse_proposals_markdown(proposer_output) # Parse raw output from this proposer
            temp_all_proposals.extend(parsed_proposals) # Add to the temporary list for aggregation later
            # --- END MODIFIED ---


            print(f"\nProposer Output ({persona_config['name']}):")
            print(proposer_output) # Print Proposer output to terminal


        # 3. Summarizer Agent - Summarize ALL raw proposals
        print("\n--- Step 3: Running Summarizer Agent ---")

        # --- MODIFIED: Create a lighter, combined input for the Summarizer from parsed proposals ---
        all_proposals_simplified_markdown = ""

        # Build simplified markdown for the summarizer from the collected parsed proposals
        if temp_all_proposals:
            # Use a dictionary to track unique titles to avoid duplication in simplified list
            unique_proposals_for_summarizer = {}
            for prop in temp_all_proposals:
                 title = prop.get("title", "Untitled").strip()
                 title = re.sub(r'\*\*:?$', '', title).strip() # Clean title
                 if title and title not in unique_proposals_for_summarizer:
                     unique_proposals_for_summarizer[title] = prop # Store the first occurrence

            if unique_proposals_for_summarizer:
                all_proposals_simplified_markdown += "## Proposed Enhancements Summary for LLM\n\n" # Add a clear header
                for title, prop in unique_proposals_for_summarizer.items():
                    description = prop.get("description", "No description provided.").strip()
                    all_proposals_simplified_markdown += f"*   **{title}:** {description}\n"
            else:
                 all_proposals_simplified_markdown = "No distinct proposals were identified."

        # If no proposals were parsed/simplified, handle that case
        if not all_proposals_simplified_markdown.strip() or all_proposals_simplified_markdown == "No distinct proposals were identified.":
             print("API: Skipping summarizer as no significant proposals were generated or parsed.")
             all_proposals_summarized_markdown = "No proposals were generated by any agent."
        else:
             # Use retry function for the agent invocation
             summarizer_result = invoke_agent_with_retry(summarizer_agent, {
                 "standard_name": standard_name,
                 "standard_short_name": standard_short_name,
                 # Pass the *simplified* combined markdown to the summarizer using the CORRECT key name
                 "all_proposals_simplified_markdown": all_proposals_simplified_markdown # <--- CORRECT KEY NAME
                 # Keep prompt token size low for summarizer - remove unnecessary context
                 # "analysis_output": analysis_output, # Removed
                 # "finance_news": finance_news_text # Removed
             })
             all_proposals_summarized_markdown = summarizer_result['text'].strip()
        # --- END MODIFIED Summarizer Input ---

        print("\nSummarizer Output:")
        print(all_proposals_summarized_markdown) # Print Summarizer output to terminal


        # 4. Multiple Validator Agents (Specific Fixed Set) - Validate and Score ALL proposals (based on SUMMARY)
        print(f"\n--- Step 4: Running {len(VALIDATOR_AGENTS)} Validator Agents (Specific) ---")
        validator_outputs_by_name = {}

        # Only run validators if there was a summary generated that contains actual items
        # Check if the summarized markdown looks like a summary list
        if all_proposals_summarized_markdown.startswith("## Summarized Proposals for") and "*" in all_proposals_summarized_markdown:
             for validator_key, validator_config in VALIDATOR_AGENTS.items():
                print(f"\nRunning Validator Agent: {validator_config['persona']['name']} ({validator_key})...")
                # Pass specific persona name and description to the validator prompt
                persona_name_for_prompt = validator_config['persona']['name']
                persona_description_for_prompt = validator_config['persona']['description']

                # Use retry function for the agent invocation
                # Pass the *summarized* proposals + minimal context to the validator
                validation_result = invoke_agent_with_retry(validator_config['agent'], {
                    "standard_name": standard_name, # Keep standard name
                    "standard_short_name": standard_short_name, # Keep short name
                    # Keep analysis output as it provides the news-driven gaps context
                    "analysis_output": analysis_output,
                    # Pass the *summarized* proposals using the CORRECT key name
                    "all_proposals_summarized": all_proposals_summarized_markdown, # <-- CORRECT KEY NAME

                    # Standard text snippet and original news text REMOVED from validator invoke call
                    # because the placeholder is removed from the validator template.
                    # "standard_text_snippet": standard_text_snippet, # Removed
                    # "finance_news": finance_news_text, # <-- REMOVED FROM INVOKE AS WELL

                    "persona_name": persona_name_for_prompt,
                    "persona_description": persona_description_for_prompt,
                })
                # --- END MODIFIED Validator Input ---

                validator_output = validation_result['text'].strip()
                validator_outputs_by_name[validator_key] = validator_output # Store raw output by validator name

                print(f"\nValidator Output ({validator_config['persona']['name']}):")
                print(validator_output) # Print Validator output to terminal
        else:
             # If summary was empty or just the "no proposals" message, set validator outputs accordingly
             print("API: Skipping validators as no significant proposals were generated or summarized.")
             validator_outputs_by_name = {vk: f"No proposals were generated or summarized, so validation by {vc['persona']['name']} was not performed." for vk, vc in VALIDATOR_AGENTS.items()} # Populate with message
             for vk, output in validator_outputs_by_name.items():
                 print(f"\nValidator Output ({VALIDATOR_AGENTS[vk]['persona']['name']}):\n{output}")


        # 5. Aggregate Scores and Determine Final Verdicts
        print("\n--- Step 5: Aggregating Scores and Determining Final Verdicts ---")
        # Ensure this step runs even if no proposals were generated, to return an empty list
        # We use temp_all_proposals which was populated during the proposer step
        if not temp_all_proposals:
             final_proposals_list = []
             print("\n*No unique proposals to aggregate.*") # Updated message
        else:
            # Aggregate using the *original parsed proposals* (from temp_all_proposals)
            # and the *raw* validator outputs (to get scores/justifications)
            # Need to map the temporary list back to the dictionary structure expected by aggregate_scores_and_verdicts
            # A simpler way is to repopulate all_proposals_dict from the parsed proposals
            all_proposals_dict_for_aggregation = {}
            for prop in temp_all_proposals:
                title = prop.get("title", "Untitled").strip()
                title = re.sub(r'\*\*:?$', '', title).strip()
                if title and title not in all_proposals_dict_for_aggregation:
                     all_proposals_dict_for_aggregation[title] = prop
                     # Note: Source personas might be lost here if not added during parsing,
                     # but aggregate_scores_and_verdicts doesn't strictly need it for core logic.
                     # Let's pass proposer_outputs_by_persona as before for consistency.


            final_proposals_list = aggregate_scores_and_verdicts(
                proposer_outputs_by_persona, # Pass individual proposer outputs for parsing (needed for full details)
                validator_outputs_by_name, # Pass individual validator outputs for parsing
                PROPOSER_PERSONAS, # Pass proposer config for names
                VALIDATOR_PERSONAS # Pass validator config for names
            )

        print("\n--- Final Aggregated Results Summary ---")
        if final_proposals_list:
            for prop in final_proposals_list:
                 # Get source persona names for summary print
                 source_personas_summary = ", ".join(prop.get("sourcePersonas", ["Unknown"]))
                 print(f"- {prop.get('title', 'Untitled')} (from {source_personas_summary}): Verdict='{prop.get('finalVerdict')}', Mean Score={prop.get('meanScore', 'N/A')}, From {prop.get('numScoresReceived', 0)}/{prop.get('totalValidatorsConfigured', len(VALIDATOR_AGENTS))} Validators")
                 if prop.get('finalVerdict') == 'Rejected':
                     print(f"  Reason: {prop.get('finalVerdictJustification', 'N/A')}")

        else:
            print("*No unique proposals generated or scored across all steps.*") # Refine message

        print("\n--- Process Complete ---")
        # Return the structured list of final proposals with aggregated scoring
        # Frontend needs analysis, news, and the structured final proposals list
        return jsonify({
            "standard": {
                "key": standard_key,
                "name": standard_name,
                "short_name": standard_short_name
            },
            "analysis": analysis_output, # Analysis from the Reviewer (raw markdown)
            "finance_news": finance_news_text, # Provided news text (raw markdown)
            "final_proposals": final_proposals_list # <--- Structured list of proposals with aggregated scoring
        })

    except Exception as e:
        print(f"\n--- API Error ---")
        print(f"API Error during multi-agent processing: {str(e)}")
        import traceback
        traceback.print_exc()
        print("-----------------")
        return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500

if __name__ == '__main__':
    # Note about using Groq API key in GROQ_API_KEY environment variable
    # Check for GROQ_API_KEY specifically, as that's the variable used now
    if os.getenv("GROQ_API_KEY"):
        print("\n--- Groq API Configuration ---")
        print(f"Using model: {LLM_MODEL_NAME}")
        print(f"API Base URL: {GROQ_API_BASE}")
        print("Reading Groq API key from GROQ_API_KEY environment variable.")
        print("------------------------------\n")
    else:
        print("CRITICAL ERROR: GROQ_API_KEY not found. Please set it in your .env file or environment.")


    # Ensure standards directory exists
    standards_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'standards')
    if not os.path.exists(standards_dir):
        os.makedirs(standards_dir)
        print(f"Created '{standards_dir}' directory. Please add your AAOIFI standard text files here.")
    # Check if standard files exist (basic check)
    for key in STANDARD_FILES:
        file_path = STANDARD_FILES[key]["path"]
        full_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        if not os.path.exists(full_file_path):
             print(f"WARNING: Standard file '{file_path}' not found. Please create this file and add the text for {STANDARD_FILES[key]['name']}.")

    print("Server will listen on http://127.0.0.1:5001")
    print("-----------------------------\n")
    # Note: You might want to disable debug=True in production
    app.run(debug=True, port=5001)