import React, { useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './App.css'; // Ensure you have App.css with styles

const API_BASE_URL = "http://localhost:5001";

// Removed STANDARDS_OPTIONS as the dropdown is removed.
// The backend will now process FAS 32 by default for manual enhancement.


const THEMATIC_AGENT_TYPES = { // Used for display names in UI
  "Foundations_Compliance": { name: "Foundations & Compliance" },
  "Asset_Risk_Accounting": { name: "Asset, Risk & Accounting" }
};


function AgentOutput({ title, content, isLoading, stageActive }) {
  if (isLoading && stageActive && (content === null || content === undefined)) {
    return (
      <div className="agent-output-container loading-message">
        <h2>{title}</h2>
        {/* Updated loading message - Standard name is now fixed to FAS 32 conceptually */}
        <p>Loading {title.split('(')[0].trim().toLowerCase()} for FAS 32...</p>
      </div>
    );
  }
  if (content === null || content === undefined ) {
     return (
       <div className="agent-output-container">
         <h2>{title}</h2>
         <p>No data generated for this step, or step not yet reached.</p>
       </div>
     );
  }
  if (typeof content === 'string') {
    return (
      <div className="agent-output-container">
        <h2>{title}</h2>
        <div className="markdown-content">
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
      </div>
    );
  }
  return null;
}


function App() {
  // Removed selectedStandard state

  const [analysisOutput, setAnalysisOutput] = useState(null);
  const [proposalsOutput, setProposalsOutput] = useState(null);
  const [validationOutput, setValidationOutput] = useState(null);
  const [overallLoading, setOverallLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentStage, setCurrentStage] = useState(0);
  const [structuredResults, setStructuredResults] = useState([]);

  // Removed resetOutputs handler

  // Removed handleStandardChange handler

  // --- NEW resetOutputs handler ---
  const resetOutputs = () => {
    setAnalysisOutput(null);
    setProposalsOutput(null);
    setValidationOutput(null);
    setStructuredResults([]);
    setError(null);
    setCurrentStage(0);
  };
  // --- END NEW resetOutputs handler ---


  // --- MODIFIED processStandard handler (always uses FAS 32) ---
  const processStandard = async () => {
    // Removed check for selectedStandard

    resetOutputs(); // Reset outputs before starting
    setOverallLoading(true);

    try {
      setCurrentStage(1);
      // --- Hardcode standardKey to "FAS32" ---
      const response = await axios.post(`${API_BASE_URL}/api/enhance-standard`, { standardKey: "FAS32" });
      // --- END Hardcode ---
      console.log("Full API Response:", response.data);

      setAnalysisOutput(response.data.analysis_markdown || "No analysis output received from backend.");
      setCurrentStage(2);

      setProposalsOutput(response.data.proposals_combined_markdown_for_ui || "No proposals output received from backend.");
      setCurrentStage(3);

      let combinedValMd = "";
      if (response.data.validator_raw_outputs_by_theme_for_ui && Object.keys(response.data.validator_raw_outputs_by_theme_for_ui).length > 0) {
        for (const themeDisplayKey in response.data.validator_raw_outputs_by_theme_for_ui) { // themeDisplayKey is already "Foundations & Compliance"
            combinedValMd += `\n\n## Validation Report by ${themeDisplayKey}\n\n${response.data.validator_raw_outputs_by_theme_for_ui[themeDisplayKey]}`;
        }
        setValidationOutput(combinedValMd.trim() || "No raw validation output (empty after combining).");
      } else {
        setValidationOutput("No raw validation output received from backend.");
      }

      setCurrentStage(4);

      if (response.data.final_enhancement_proposals_with_validations) {
        setStructuredResults(response.data.final_enhancement_proposals_with_validations);
      } else {
        setStructuredResults([]);
      }
      setCurrentStage(5);

    } catch (err) {
      console.error("Error:", err);
      setError(err.response?.data?.error || err.message || "Processing failed. Check backend logs.");
      setCurrentStage(0);
    } finally {
      setOverallLoading(false);
    }
  };
  // --- END MODIFIED processStandard handler ---


  return (
    <div className="App">
      <h1>AAOIFI Standard Enhancement AI Assistant</h1>
      <div className="standard-selector">
        {/* --- Removed Standard Selection Dropdown --- */}
        {/* <label htmlFor="standard-select">Choose FAS:</label>
        <select id="standard-select" value={selectedStandard} onChange={handleStandardChange}>
          {STANDARDS_OPTIONS.map(opt => (<option key={opt.key} value={opt.key}>{opt.name}</option>))}
        </select> */}
        {/* --- End Removed Standard Selection Dropdown --- */}

        {/* --- Display the default standard --- */}
        <p>Start Enhancing FAS: </p>
        {/* --- End Display default standard --- */}


        <button onClick={processStandard} disabled={overallLoading}>
          {overallLoading ? "Processing..." : "Start"} {/* Updated button text */}
        </button>
      </div>
      {error && <div className="error-message">Error: {error}</div>}

      {currentStage >= 1 && <AgentOutput title="1. Standard Review (Analyzer)" content={analysisOutput} isLoading={overallLoading} stageActive={currentStage === 1} />}
      {currentStage >= 2 && <AgentOutput title="2. AI-Driven Enhancements (Proposers - Raw Combined)" content={proposalsOutput} isLoading={overallLoading} stageActive={currentStage === 2} />}
      {currentStage >= 3 && <AgentOutput title="3. Thematic Validations (Validators - Raw Combined)" content={validationOutput} isLoading={overallLoading} stageActive={currentStage === 3} />}

      {currentStage >= 4 && structuredResults && structuredResults.length > 0 && !overallLoading && (
        <div className="agent-output-container">
          <h2>4. Final Proposals with Aggregated Validations & Meta-Review</h2>
          {structuredResults.map((item, index) => (
            <div key={item.original_proposal.id || index} className="proposal-validation-item">
              <h3>Proposal: {item.original_proposal.title}
                {item.meta_review && item.meta_review.status && (
                  <span className={`status-badge status-${item.meta_review.status.toLowerCase().replace(/\s+/g, '-').replace(/[()]/g, '')}`}>
                    ({item.meta_review.status})
                  </span>
                )}
              </h3>
              {/* To get display name for original_proposal.source_theme, need to map it from THEMATIC_AGENT_TYPES keys if source_theme stores "Foundations_Compliance" */}
              <p><em>(Source: {item.original_proposal.source_theme})</em></p> {/* This currently shows display name because that's what was stored in proposal dict */}
              <p><strong>Average Score: {item.average_score}</strong></p>
              {item.meta_review && item.meta_review.justification && (
                <div className="meta-review-section">
                  <h4>Meta-Review Justification:</h4>
                  <p>{item.meta_review.justification}</p>
                </div>
              )}
              <details>
                <summary>Show Original Proposal Text & Thematic Validations</summary>
                <h4>Original Proposal Body:</h4>
                <div className="markdown-content-tight"><ReactMarkdown>{item.original_proposal.full_text_markdown}</ReactMarkdown></div>
                <h4>Thematic Validations:</h4>
                {item.validations_by_theme && Object.keys(item.validations_by_theme).length > 0 ? (
                  Object.entries(item.validations_by_theme).map(([themeDisplayKey, validation]) => (  // themeDisplayKey is already "Foundations & Compliance"
                    <div key={themeDisplayKey} className="thematic-validation-detail">
                      <h5>Validation by: {themeDisplayKey}</h5>
                      <p><strong>Score:</strong> {validation.score_given ?? "N/A"} | <strong>Verdict:</strong> {validation.verdict_text}</p>
                      <details className="sub-details">
                          <summary>Show full validation text by this theme</summary>
                          <div className="markdown-content-tight"><ReactMarkdown>{validation.full_validation_markdown}</ReactMarkdown></div>
                      </details>
                    </div>
                  ))
                ) : (
                  <p>No thematic validations available for this proposal.</p>
                )}
              </details>
              <hr style={{ margin: "20px 0"}}/>
            </div>
          ))}
        </div>
      )}
      {currentStage === 5 && !error && !overallLoading && <div className="process-complete-message">Process Complete!</div>}
    </div>
  );
}

export default App;