from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from utils import invoke_agent_with_retry

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

def run_analyzer_agent(llm_structured: ChatOpenAI, standard_name: str, standard_short_name: str, standard_text: str, finance_news: str, max_retries: int, initial_retry_delay: float, RateLimitError):
    """Initializes and runs the Analyzer agent."""
    analyzer_chain = LLMChain(llm=llm_structured, prompt=analyzer_prompt)
    print("\n--- Step 1: Running Analyzer Agent (Identifiying News-driven Gaps) ---")
    inputs = {
        "standard_name": standard_name,
        "standard_text": standard_text,
        "standard_short_name": standard_short_name,
        "finance_news": finance_news
    }
    analysis_result = invoke_agent_with_retry(analyzer_chain, inputs, max_retries, initial_retry_delay, RateLimitError)
    analysis_output = analysis_result['text']
    print("\nAnalyzer Output:")
    print(analysis_output)
    return analysis_output