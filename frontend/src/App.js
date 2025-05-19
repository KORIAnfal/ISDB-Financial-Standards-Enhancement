import React, { useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_BASE_URL = "http://localhost:5001"; // Ensure this matches your Flask port

// Removed STANDARDS_OPTIONS as the dropdown is being removed.
// The backend's AGENT_STANDARD_FILES_MAP will dictate which are processable.


// Component to display raw markdown output (Analysis, News)
function AgentOutput({ title, content, isLoading }) {
    // Only render the container if there's content or it's loading for this specific output
    const showLoading = isLoading && !content;

    if (!content && !showLoading) {
        return null;
    }

    if (showLoading) {
        return (
            <div className="agent-output-container loading-message">
                <h2>{title}</h2>
                <p>Loading {title.toLowerCase()}...</p>
            </div>
        );
    }

    // Render content when available
    return (
        <div className="agent-output-container">
            <h2>{title}</h2>
            <div className="markdown-content">
                <ReactMarkdown>{content}</ReactMarkdown>
            </div>
        </div>
    );
}

// Component to display structured proposals with validation
function ProposalCard({ proposal }) {
    return (
        <div className={`proposal-card ${proposal.finalVerdict?.toLowerCase()}`}>
            <h3>{proposal.title || 'Untitled Proposal'}</h3>
            <p>
                <strong>Verdict:</strong> <span className={`verdict ${proposal.finalVerdict?.toLowerCase()}`}>{proposal.finalVerdict || 'Undetermined'}</span>
                {proposal.meanScore !== undefined && proposal.meanScore !== null && (
                   <>
                     <strong> Mean Score:</strong> {proposal.meanScore.toFixed(2)}/100
                     <span className="score-count"> ({proposal.numScoresReceived} / {proposal.totalValidatorsConfigured} scored)</span>
                   </>
                )}
                 {proposal.finalVerdictJustification && proposal.finalVerdictJustification.trim() !== '' && (
                     <span className="verdict-justification"> ({proposal.finalVerdictJustification})</span>
                )}
            </p>

            {proposal.sourcePersonas && proposal.sourcePersonas.length > 0 && (
               <p className="source-themes"><strong>Suggested by:</strong> {proposal.sourcePersonas.join(', ')}</p>
            )}

            {proposal.individualValidations && proposal.individualValidations.length > 0 && proposal.numScoresReceived > 0 && (
                <div className="individual-validations">
                    <h4>Individual Validator Assessments:</h4>
                    <ul>
                        {proposal.individualValidations.filter(val => val.score !== null).map((val, index) => (
                            <li key={index}>
                                <strong>{val.validatingValidator || 'Validator'}:</strong>
                                Score {val.rawScore !== null && val.rawScore !== undefined ? val.rawScore.toFixed(2) : 'N/A'}/100
                                {val.assessment && val.assessment.trim() !== '' && ` - Assessment: ${val.assessment}`}
                                {val.justification && val.justification.trim() !== '' && ` - Justification: ${val.justification}`}
                            </li>
                        ))}
                         {/* Optional: Add a note if some validators didn't produce valid scores, but this is often covered by the main score count */}
                         {/* {proposal.numScoresReceived < proposal.totalValidatorsConfigured && (
                              <li className="scoring-note">Note: Valid scores received from {proposal.numScoresReceived} out of {proposal.totalValidatorsConfigured} configured validators.</li>
                         )} */}
                    </ul>
                </div>
            )}

             {proposal.personaDetailsMarkdown && proposal.personaDetailsMarkdown.trim() !== '' && (
                 <div className="proposer-details-markdown">
                    <h4>Details from Proposer Perspectives:</h4>
                    <ReactMarkdown>{proposal.personaDetailsMarkdown}</ReactMarkdown>
                 </div>
              )}
        </div>
    );
}

// Component to display results for a single standard (Analysis, Proposals, Validation)
function StandardResultsBlock({ standardResult, overallLoading }) {
    // Use standard_info from the result payload for display name/short name
    const standardInfo = standardResult.standard_info || { name: 'Unknown Standard', short_name: 'Unknown' };
    const isLoading = overallLoading && (!standardResult.analysis && !standardResult.final_proposals); // Loading state for this specific block

    if (!standardResult.analysis && !standardResult.final_proposals?.length && !isLoading && !standardResult.error) {
        return null; // Don't render if no content and not loading
    }

    if (isLoading) {
        return (
             <div className="agent-output-container loading-message standard-results-block">
                <h3>{standardInfo.name} ({standardInfo.short_name})</h3>
                <p>Loading results for this standard...</p>
            </div>
        );
    }

    return (
        <div className="agent-output-container standard-results-block">
            <h3>
                {standardInfo.name} ({standardInfo.short_name})
                {/* Display relevance score from classification */}
                {standardResult.relevance_score !== undefined && standardResult.relevance_score !== null && (
                    <span className="standard-relevance-score"> (Relevance: {(standardResult.relevance_score * 100).toFixed(0)}%)</span>
                )}
            </h3>

             {/* Display errors for this specific standard */}
             {standardResult.error && (
                 <div className="error-message">Error processing this standard: {standardResult.error}</div>
             )}

             {/* Display Analysis Output for this standard */}
             {standardResult.analysis && (
                  <>
                    <h4>Analysis (Analyzer Agent)</h4>
                     <div className="markdown-content">
                        <ReactMarkdown>{standardResult.analysis}</ReactMarkdown>
                     </div>
                  </>
             )}

             {/* Display Proposals & Validation Results for this standard */}
             {standardResult.final_proposals && standardResult.final_proposals.length > 0 && (
                 <>
                    <h4>Proposed Enhancements & Validation Results</h4>
                     <div className="proposals-list">
                         {standardResult.final_proposals.map((proposal, index) => (
                              <ProposalCard key={index} proposal={proposal} />
                          ))}
                     </div>
                 </>
             )}

             {/* Message if no analysis or proposals generated for this specific standard */}
              {!standardResult.analysis && (!standardResult.final_proposals || standardResult.final_proposals.length === 0) && !standardResult.error && (
                  <p>*No analysis or proposals generated for this standard.*</p>
              )}
        </div>
    );
}


function App() {
  // Removed selectedStandard state and related handler

  const [userNewsText, setUserNewsText] = useState("");

  // Removed manualEnhanceResults state

  // State for results from the analyze news endpoint - this is now the PRIMARY state
  const [newsAnalysisResults, setNewsAnalysisResults] = useState(null); // Null initially, becomes object { news_text, identified_standards_scores, processed_standards_results, overall_errors }

  const [overallLoading, setOverallLoading] = useState(false);
  const [error, setError] = useState(null);

  const resetOutputs = () => {
    // Removed manualEnhanceResults reset
    setNewsAnalysisResults(null); // Reset news analysis results
    setError(null);
    // Keep userNewsText value to allow re-running with the same text
    // setUserNewsText(""); // Uncomment if you want to clear text on reset
  };

  // Removed handleStandardChange handler

  const handleUserNewsTextChange = (event) => {
    setUserNewsText(event.target.value);
    // Reset outputs if news text is changed after a previous run
    if (newsAnalysisResults) { // Check if there are any results displayed
        resetOutputs();
    }
  };

  // Removed handleEnhanceStandard function

  // Handler for the "Analyze News" button (this is now the only main action)
  const handleAnalyzeNews = async () => {
      if (!userNewsText.trim()) {
          setError("Please provide the finance news text to analyze.");
          return;
      }

      resetOutputs(); // Reset outputs before starting
      setOverallLoading(true);

      try {
          console.log(`Sending analyze news request with provided news text.`);
          const response = await axios.post(`${API_BASE_URL}/api/analyze-news`, {
              news_text: userNewsText // Backend expects 'news_text'
          });
          console.log("Analyze News Response received:", response.data);

          setNewsAnalysisResults(response.data); // Store the entire response object

          // Display any overall errors from the backend
           if (response.data.overall_errors && response.data.overall_errors.length > 0) {
               setError("Backend encountered errors processing some standards. Check console for details. Errors: " + response.data.overall_errors.join("; "));
           } else if (response.data.processed_standards_results && Object.keys(response.data.processed_standards_results).length === 0 &&
                      response.data.identified_standards_scores && response.data.identified_standards_scores.length > 0) {
                // Special case: classifier found standards, but none were processed (e.g., below threshold or not configured)
                setError("Classifier identified relevant standards, but none met the criteria for agent processing.");
           } else if (response.data.identified_standards_scores && response.data.identified_standards_scores.length === 0) {
                // Special case: Classifier found no relevant standards at all
                 setError("Classifier found no relevant standards in the provided news text.");
           }


      } catch (err) {
          console.error("Error processing news analysis:", err);
          let errorMessage = "Failed to analyze news. Check the backend console for details.";
          if (err.response && err.response.data && err.response.data.error) {
              errorMessage = err.response.data.error;
          } else if (err.message) {
              errorMessage = err.message;
          }
          setError(errorMessage);

      } finally {
          setOverallLoading(false);
      }
  };


  // Simplify check for having any results to display
  const hasNewsAnalysisResultsToDisplay = newsAnalysisResults !== null &&
                                          (newsAnalysisResults.news_text ||
                                           (newsAnalysisResults.identified_standards_scores && newsAnalysisResults.identified_standards_scores.length > 0) ||
                                           (newsAnalysisResults.processed_standards_results && Object.keys(newsAnalysisResults.processed_standards_results).length > 0));


  return (
    <div className="App">
      <h1>AAOIFI Standard Enhancement AI Assistant</h1>

      <div className="standard-selector-container"> {/* Container for input and buttons */}
          {/* Removed standard selector dropdown */}

          <div className="news-query-input">
              <label htmlFor="user-news-text">News Text (Change Trigger):</label>
              <textarea
                  id="user-news-text"
                  placeholder="Paste or type recent finance news text relevant to Islamic Finance, tech, or digital assets that may highlight gaps in existing standards..."
                  value={userNewsText}
                  onChange={handleUserNewsTextChange} // Corrected typo here if it existed
                  disabled={overallLoading}
                  rows="6" // Slightly more height
              />
          </div>

          <div className="action-buttons"> {/* Wrapper for button */}
              {/* Removed manual enhance button */}
               {/* The single "Analyze News" button */}
              <button onClick={handleAnalyzeNews} disabled={overallLoading || !userNewsText.trim()}> {/* Disable if loading or text is empty */}
                 {overallLoading ? "Analyzing News..." : "Analyze News & Enhance Relevant Standards"}
              </button>
          </div>
      </div>


      {error && <div className="error-message">Error: {error}</div>}

      {/* Overall loading message */}
      {overallLoading && (
          <div className="loading-message">
               {newsAnalysisResults === null ? // Check if we are waiting for *any* news analysis results
                   "Analyzing News and Identifying Standards..."
                   : "Processing identified standards through AI agents..." // Message once standards are identified
               }
          </div>
      )}

      {/* Display results only if newsAnalysisResults has data */}
      {hasNewsAnalysisResultsToDisplay && !overallLoading && (
           <>
              {/* Provided News Text (Echo) */}
               {newsAnalysisResults.news_text && (
                   <AgentOutput
                     title="Provided News Text" // Simpler title
                     content={newsAnalysisResults.news_text}
                     isLoading={false}
                   />
               )}


              {/* Standards Identified by Classifier */}
               {newsAnalysisResults.identified_standards_scores && newsAnalysisResults.identified_standards_scores.length > 0 && (
                   <div className="agent-output-container">
                       <h2>Standards Identified by Classifier</h2>
                       <p>Based on the news text, the classifier identified the following relevant standards:</p>
                       <ul>
                            {newsAnalysisResults.identified_standards_scores.map(([std_id, score]) => (
                                <li key={std_id}>
                                    <strong>{std_id}:</strong> {(score * 100).toFixed(1)}%
                                     {/* Optional: Add standard name if available from backend */}
                                     {newsAnalysisResults.processed_standards_results?.[std_id]?.standard_info?.name &&
                                        ` - ${newsAnalysisResults.processed_standards_results[std_id].standard_info.name}`
                                     }
                                </li>
                            ))}
                       </ul>
                       {/* Add message if only a subset was processed */}
                        {newsAnalysisResults.processed_standards_results && Object.keys(newsAnalysisResults.processed_standards_results).length > 0 &&
                         Object.keys(newsAnalysisResults.processed_standards_results).length < newsAnalysisResults.identified_standards_scores.length && (
                           <p className="scoring-note">Showing detailed agent results below for the top {Object.keys(newsAnalysisResults.processed_standards_results).length} relevant standards configured for processing.</p>
                        )}
                         {newsAnalysisResults.processed_standards_results && Object.keys(newsAnalysisResults.processed_standards_results).length === 0 && (
                            <p className="scoring-note">None of the identified standards met the criteria for detailed agent processing.</p>
                         )}
                   </div>
               )}


              {/* Analysis & Proposals for Each Processed Standard */}
               {newsAnalysisResults.processed_standards_results && Object.keys(newsAnalysisResults.processed_standards_results).length > 0 && (
                   <div className="standards-analysis-results"> {/* Container for multiple standard results */}
                        <h2>Detailed Agent Analysis & Proposals</h2>
                        {Object.entries(newsAnalysisResults.processed_standards_results).map(([std_id, standardResult]) => (
                            <StandardResultsBlock key={std_id} standardResult={standardResult} overallLoading={overallLoading} />
                        ))}
                   </div>
               )}

              {/* Final Completion Message */}
               {!overallLoading && !error && newsAnalysisResults !== null && <div className="completion-message">Process Complete!</div>}
           </>
      )}

      {/* Initial state message */}
       {!overallLoading && !error && !hasNewsAnalysisResultsToDisplay && (
            <div className="completion-message">Enter news text above and click "Analyze News & Enhance Relevant Standards".</div>
       )}


    </div>
  );
}

export default App;