import React from 'react';
import PropTypes from 'prop-types';

// Constants for validation
const MAX_TEXT_LENGTH = 50000;
const MIN_TEXT_LENGTH = 50;

/**
 * Input Section Component
 * 
 * Handles text/URL input with tabs
 */
function InputSection({
  activeTab,
  setActiveTab,
  inputText,
  setInputText,
  inputUrl,
  setInputUrl,
  onAnalyze,
  onClear,
  onLoadSample,
  loading
}) {
  // Validation helpers
  const textLength = inputText.length;
  const wordCount = inputText.split(/\s+/).filter(Boolean).length;
  const isTextTooShort = textLength > 0 && textLength < MIN_TEXT_LENGTH;
  const isTextTooLong = textLength > MAX_TEXT_LENGTH;
  const isValidUrl = inputUrl === '' || /^https?:\/\/.+\..+/.test(inputUrl);
  
  // Handle text input with length limit
  const handleTextChange = (e) => {
    const value = e.target.value;
    if (value.length <= MAX_TEXT_LENGTH) {
      setInputText(value);
    }
  };

  return (
    <div id="analyzer" className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      {/* Tab Headers */}
      <div className="flex border-b border-gray-200">
        <button
          className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
            activeTab === 'text'
              ? 'text-primary-600 border-b-2 border-primary-600 bg-primary-50'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('text')}
        >
          <svg className="w-4 h-4 inline-block mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Paste Text
        </button>
        <button
          className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
            activeTab === 'url'
              ? 'text-primary-600 border-b-2 border-primary-600 bg-primary-50'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('url')}
        >
          <svg className="w-4 h-4 inline-block mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
          </svg>
          Enter URL
        </button>
      </div>

      {/* Tab Content */}
      <div className="p-6">
        {activeTab === 'text' ? (
          <div>
            <label htmlFor="text-input" className="block text-sm font-medium text-gray-700 mb-2">
              Article Text
            </label>
            <textarea
              id="text-input"
              className={`w-full h-48 p-4 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 resize-none text-gray-900 ${
                isTextTooShort || isTextTooLong ? 'border-red-300' : 'border-gray-300'
              }`}
              placeholder="Paste the news article text here (minimum 50 characters)..."
              value={inputText}
              onChange={handleTextChange}
              disabled={loading}
              aria-describedby="text-help"
            />
            <div id="text-help" className="mt-2 flex justify-between text-sm">
              <div className="flex gap-4">
                <span className={textLength > 0 ? (isTextTooShort ? 'text-red-500' : 'text-gray-500') : 'text-gray-500'}>
                  {wordCount} words
                </span>
                <span className={isTextTooLong ? 'text-red-500' : 'text-gray-500'}>
                  {textLength.toLocaleString()}/{MAX_TEXT_LENGTH.toLocaleString()} chars
                </span>
              </div>
              <button
                className="text-primary-600 hover:text-primary-700 underline"
                onClick={onLoadSample}
                disabled={loading}
                type="button"
              >
                Load sample text
              </button>
            </div>
            {isTextTooShort && (
              <p className="mt-1 text-sm text-red-500">
                Please enter at least {MIN_TEXT_LENGTH} characters for accurate analysis.
              </p>
            )}
          </div>
        ) : (
          <div>
            <label htmlFor="url-input" className="block text-sm font-medium text-gray-700 mb-2">
              Article URL
            </label>
            <input
              id="url-input"
              type="url"
              className={`w-full p-4 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-gray-900 ${
                inputUrl && !isValidUrl ? 'border-red-300' : 'border-gray-300'
              }`}
              placeholder="https://example.com/news-article"
              value={inputUrl}
              onChange={(e) => setInputUrl(e.target.value)}
              disabled={loading}
              aria-describedby="url-help"
            />
            {inputUrl && !isValidUrl && (
              <p className="mt-1 text-sm text-red-500">
                Please enter a valid URL starting with http:// or https://
              </p>
            )}
            <p id="url-help" className="mt-2 text-sm text-gray-500">
              We'll extract the article content automatically
            </p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="mt-6 flex flex-wrap gap-3">
          <button
            className={`flex-1 py-3 px-6 rounded-lg font-medium transition-colors ${
              loading
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-primary-600 text-white hover:bg-primary-700'
            }`}
            onClick={onAnalyze}
            disabled={loading}
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Analyzing...
              </span>
            ) : (
              'Analyze for Bias'
            )}
          </button>
          <button
            className="py-3 px-6 border border-gray-300 rounded-lg font-medium text-gray-700 hover:bg-gray-50 transition-colors"
            onClick={onClear}
            disabled={loading}
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}

InputSection.propTypes = {
  activeTab: PropTypes.oneOf(['text', 'url']).isRequired,
  setActiveTab: PropTypes.func.isRequired,
  inputText: PropTypes.string.isRequired,
  setInputText: PropTypes.func.isRequired,
  inputUrl: PropTypes.string.isRequired,
  setInputUrl: PropTypes.func.isRequired,
  onAnalyze: PropTypes.func.isRequired,
  onClear: PropTypes.func.isRequired,
  onLoadSample: PropTypes.func.isRequired,
  loading: PropTypes.bool.isRequired
};

export default InputSection;
