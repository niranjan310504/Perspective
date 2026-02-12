import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import InputSection from './components/InputSection';
import ResultsSection from './components/ResultsSection';
import FactCheckSection from './components/FactCheckSection';
import BiasExplanations from './components/BiasExplanations';
import NewsFeed from './components/NewsFeed';
import Footer from './components/Footer';
import ErrorBoundary from './components/ErrorBoundary';
import { analyzeText, checkHealth, API_BASE_URL } from './services/api';

// Constants
const MIN_TEXT_LENGTH = 50;
const MAX_TEXT_LENGTH = 50000;

/**
 * Main Application Component
 * 
 * Perspective - Indian Media Bias Detection
 */
function App() {
  // State management
  const [inputText, setInputText] = useState('');
  const [inputUrl, setInputUrl] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('text'); // 'text' or 'url'
  const [serverStatus, setServerStatus] = useState('checking'); // 'checking', 'online', 'offline'
  const [viewMode, setViewMode] = useState('feed'); // 'feed' or 'analyze'
  const [pendingAnalysis, setPendingAnalysis] = useState(null); // URL to auto-analyze

  /**
   * Check server health on mount
   */
  useEffect(() => {
    const checkServerHealth = async () => {
      try {
        const health = await checkHealth();
        setServerStatus(health.status === 'healthy' ? 'online' : 'offline');
      } catch {
        setServerStatus('offline');
      }
    };
    
    checkServerHealth();
    
    // Check health every 30 seconds
    const interval = setInterval(checkServerHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  /**
   * Validate input before submission
   */
  const validateInput = useCallback(() => {
    if (activeTab === 'text') {
      if (!inputText.trim()) {
        return 'Please enter some text to analyze';
      }
      if (inputText.length < MIN_TEXT_LENGTH) {
        return `Text must be at least ${MIN_TEXT_LENGTH} characters for accurate analysis`;
      }
      if (inputText.length > MAX_TEXT_LENGTH) {
        return `Text exceeds maximum length of ${MAX_TEXT_LENGTH.toLocaleString()} characters`;
      }
    } else {
      if (!inputUrl.trim()) {
        return 'Please enter a URL to analyze';
      }
      if (!/^https?:\/\/.+\..+/.test(inputUrl)) {
        return 'Please enter a valid URL starting with http:// or https://';
      }
    }
    return null;
  }, [activeTab, inputText, inputUrl]);

  /**
   * Handle form submission
   */
  const handleAnalyze = async () => {
    // Check server status
    if (serverStatus === 'offline') {
      setError('Server is offline. Please ensure the backend is running.');
      return;
    }

    // Validate input
    const validationError = validateInput();
    if (validationError) {
      setError(validationError);
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const payload = activeTab === 'text' 
        ? { text: inputText }
        : { url: inputUrl };

      const response = await analyzeText(payload);
      
      if (response.success) {
        setResults(response.data);
      } else {
        setError(response.error || 'Analysis failed');
      }
    } catch (err) {
      setError(err.message || 'Failed to connect to the server');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Clear all inputs and results
   */
  const handleClear = () => {
    setInputText('');
    setInputUrl('');
    setResults(null);
    setError(null);
  };

  /**
   * Load sample text for demo
   */
  const handleLoadSample = () => {
    const sampleText = `The government's visionary policies have transformed the nation's economy, marking a new era of unprecedented growth. The Prime Minister's bold decisions have earned praise from experts worldwide, while opposition leaders continue their baseless criticism and disruptive protests. The ruling party's revolutionary reforms are expected to benefit millions, despite attempts by certain communities to spread misinformation. SHOCKING revelations emerge daily about the opposition's failed leadership, proving once again that only the current government can lead the nation to glory!`;
    
    setInputText(sampleText);
    setActiveTab('text');
    setError(null);
  };

  /**
   * Handle analyze from news feed
   */
  const handleAnalyzeFromFeed = (article) => {
    setInputUrl(article.url);
    setActiveTab('url');
    setViewMode('analyze');
    setResults(null);
    setError(null);
    // Set pending analysis to trigger useEffect
    setPendingAnalysis(article.url);
  };

  /**
   * Auto-analyze when URL is set from news feed
   */
  useEffect(() => {
    if (pendingAnalysis && viewMode === 'analyze' && !loading) {
      setPendingAnalysis(null);
      handleAnalyze();
    }
  }, [pendingAnalysis, viewMode]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        <Header />
        
        {/* Server Status Banner */}
        {serverStatus === 'offline' && (
          <div className="bg-yellow-50 border-b border-yellow-200 px-4 py-2">
            <div className="container mx-auto max-w-4xl flex items-center justify-center text-yellow-800 text-sm">
              <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              Backend server is offline. Please start the server to use the analyzer.
            </div>
          </div>
        )}
        
        <main className="flex-grow container mx-auto px-4 py-8 max-w-6xl">
          {/* View Mode Toggle */}
          <div className="flex justify-center mb-8">
            <div className="inline-flex bg-gray-100 rounded-xl p-1">
              <button
                onClick={() => setViewMode('feed')}
                className={`px-6 py-3 rounded-lg font-medium transition-all ${
                  viewMode === 'feed'
                    ? 'bg-white text-indigo-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <span className="flex items-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
                  </svg>
                  Live Feed
                </span>
              </button>
              <button
                onClick={() => setViewMode('analyze')}
                className={`px-6 py-3 rounded-lg font-medium transition-all ${
                  viewMode === 'analyze'
                    ? 'bg-white text-indigo-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <span className="flex items-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                  </svg>
                  Analyze Custom
                </span>
              </button>
            </div>
          </div>

          {/* News Feed View */}
          {viewMode === 'feed' && (
            <NewsFeed 
              apiBaseUrl={API_BASE_URL}
              onAnalyzeArticle={handleAnalyzeFromFeed}
            />
          )}

          {/* Analyze View */}
          {viewMode === 'analyze' && (
            <>
              {/* Input Section */}
              <InputSection
                activeTab={activeTab}
                setActiveTab={setActiveTab}
                inputText={inputText}
                setInputText={setInputText}
                inputUrl={inputUrl}
                setInputUrl={setInputUrl}
                onAnalyze={handleAnalyze}
                onClear={handleClear}
                onLoadSample={handleLoadSample}
                loading={loading}
              />

              {/* Error Display */}
              {error && (
                <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 fade-in">
                  <div className="flex items-center">
                    <svg className="w-5 h-5 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                    <span>{error}</span>
                    <button 
                      onClick={() => setError(null)}
                      className="ml-auto text-red-500 hover:text-red-700"
                      aria-label="Dismiss error"
                    >
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </div>
                </div>
              )}

              {/* Loading State */}
              {loading && (
                <div className="mt-8 text-center">
                  <div className="inline-flex items-center px-6 py-3 bg-primary-50 rounded-lg">
                    <svg className="animate-spin h-5 w-5 mr-3 text-primary-600" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    <span className="text-primary-700 font-medium">Analyzing article for bias and fact-checking...</span>
                  </div>
                </div>
              )}

              {/* Results Section */}
              {results && !loading && (
                <>
                  {/* Fact Check Results - Show First */}
                  <FactCheckSection factCheck={results.fact_check} />
                  
                  {/* Bias Analysis Results */}
                  <ResultsSection results={results} />
                </>
              )}
            </>
          )}

          {/* Bias Explanations */}
          <BiasExplanations />
        </main>

        <Footer />
      </div>
    </ErrorBoundary>
  );
}

export default App;
