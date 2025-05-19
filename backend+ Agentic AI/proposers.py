import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from utils import invoke_agent_with_retry

# Proposer Agent Personas (kept here as it's config related to proposers)
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

# Proposer Agent Prompt Template (kept here)
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


def parse_proposals_markdown(markdown_text):
    """
    Parses markdown output from Proposer agents, extracting proposal details.
    Handles potential variations in headings like '## Proposed Enhancement: Title'.
    Returns a list of dictionaries.
    """
    proposals = []
    # Regex to find proposal sections starting with ## Proposed Enhancement:
    # Captures the title in group 1
    proposal_sections_split = re.split(r'##\s*Proposed Enhancement:\s*(.+)', markdown_text, re.DOTALL)

    # The split results in: ['', Title1, Content1, Title2, Content2, ...]
    # We need to process in pairs (Title, Content)
    # Ensure we handle potential empty first element if markdown starts correctly
    processed_sections = proposal_sections_split
    if processed_sections and processed_sections[0].strip() == '':
        processed_sections = processed_sections[1:]

    for i in range(0, len(processed_sections), 2):
        if i+1 < len(processed_sections):
            title = processed_sections[i].strip()
            content = processed_sections[i+1].strip()

            proposal_data = {"title": title}

            # Use regex with lookahead to find key-value pairs
            desc_match = re.search(r'\*\*Proposed Change Description:\*\*\s*(.*?)(?=\n\*\*[^:]+?:|$)', content, re.DOTALL)
            if desc_match: proposal_data["description"] = desc_match.group(1).strip()

            sections_match = re.search(r'\*\*Relevant Original Section\(s\):\*\*\s*(.*?)(?=\n\*\*[^:]+?:|$)', content, re.DOTALL)
            if sections_match: proposal_data["relevantSections"] = sections_match.group(1).strip()

            rationale_match = re.search(r'\*\*Rationale\s*\(linking to analysis/news\):\*\*\s*(.*?)(?=\n\*\*[^:]+?:|$)', content, re.DOTALL)
            if not rationale_match: # Fallback
                 rationale_match = re.search(r'\*\*Rationale:\*\*\s*(.*?)(?=\n\*\*[^:]+?:|$)', content, re.DOTALL)
            if rationale_match: proposal_data["rationale"] = rationale_match.group(1).strip()

            details_match = re.search(r'\*\*Details from (.+?) Perspective:\*\*\s*(.*?)(?=\n\*\*Expected Benefit:|$)', content, re.DOTALL)
            if details_match:
                proposal_data["personaDetailsMarkdown"] = details_match.group(2).strip()

            ai_link_match = re.search(r'\*\*AI/Technology Link\s*\(linking to news\):\*\*\s*(.*?)(?=\n\*\*[^:]+?:|$)', content, re.DOTALL)
            if not ai_link_match: # Fallback
                 ai_link_match = re.search(r'\*\*AI/Technology Link:\*\*\s*(.*?)(?=\n\*\*[^:]+?:|$)', content, re.DOTALL)
            if ai_link_match: proposal_data["aiTechnologyLink"] = ai_link_match.group(1).strip()

            benefits_match = re.search(r'\*\*Expected Benefit:\*\*\s*(.*?)(?=\n## Proposed Enhancement:|$)', content, re.DOTALL) # Look until next heading or end
            if not benefits_match: # Fallback to end of content if no next heading
                 benefits_match = re.search(r'\*\*Expected Benefit:\*\*\s*(.*)', content, re.DOTALL)
            if benefits_match: proposal_data["expectedBenefits"] = benefits_match.group(1).strip()
            else: # If even the last fallback fails
                 proposal_data["expectedBenefits"] = ""


            proposals.append(proposal_data)

    # --- DEBUG PRINT ---
    # print(f"\n--- Debug: Parsed {len(proposals)} proposals from raw proposer output ---")
    # for prop in proposals:
    #     print(f"  - Parsed Title: '{prop.get('title')}' (Description starts: '{prop.get('description', '')[:50]}...')")
    # print("---------------------------------------------------------")
    # --- END DEBUG PRINT ---

    return proposals


def run_proposer_agents(llm_creative: ChatOpenAI, standard_name: str, standard_short_name: str, standard_text_snippet: str, analysis_output: str, finance_news: str, max_retries: int, initial_retry_delay: float, RateLimitError, proposer_personas):
    """Initializes and runs all Proposer agents."""
    proposer_chain = LLMChain(llm=llm_creative, prompt=proposer_prompt)
    proposer_outputs_by_persona = {}
    temp_all_proposals = [] # To collect parsed proposals across all proposers

    print(f"\n--- Step 2: Running {len(proposer_personas)} Proposer Agents (Persona-based) ---")

    for persona_key, persona_config in proposer_personas.items():
        print(f"\nRunning Proposer Agent: {persona_config['name']} ({persona_key})...")
        inputs = {
            "standard_name": standard_name,
            "analysis_output": analysis_output,
            "standard_text_snippet": standard_text_snippet,
            "standard_short_name": standard_short_name,
            "persona_name": persona_config["name"],
            "persona_description": persona_config["description"],
            "finance_news": finance_news
        }
        proposals_result = invoke_agent_with_retry(proposer_chain, inputs, max_retries, initial_retry_delay, RateLimitError)
        proposer_output = proposals_result['text'].strip()
        proposer_outputs_by_persona[persona_key] = proposer_output

        # Parse and collect proposals here immediately after getting output from one proposer
        parsed_proposals = parse_proposals_markdown(proposer_output)
        temp_all_proposals.extend(parsed_proposals)

        print(f"\nRaw Proposer Output ({persona_config['name']}):\n{proposer_output}") # Keep raw output print for full debugging

    # Pass proposer_personas back as well, needed for aggregation
    return proposer_outputs_by_persona, temp_all_proposals, proposer_personas