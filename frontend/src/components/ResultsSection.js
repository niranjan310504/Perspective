import React from 'react';
import PropTypes from 'prop-types';

/**
 * Bias type display names and colors
 */
const BIAS_CONFIG = {
  political: { name: 'Political', color: 'bg-red-500', lightColor: 'bg-red-100', textColor: 'text-red-700' },
  gender: { name: 'Gender', color: 'bg-purple-500', lightColor: 'bg-purple-100', textColor: 'text-purple-700' },
  entity: { name: 'Entity', color: 'bg-blue-500', lightColor: 'bg-blue-100', textColor: 'text-blue-700' },
  racial: { name: 'Racial', color: 'bg-orange-500', lightColor: 'bg-orange-100', textColor: 'text-orange-700' },
  religious: { name: 'Religious', color: 'bg-green-500', lightColor: 'bg-green-100', textColor: 'text-green-700' },
  regional: { name: 'Regional', color: 'bg-yellow-500', lightColor: 'bg-yellow-100', textColor: 'text-yellow-700' },
  sensationalism: { name: 'Sensationalism', color: 'bg-pink-500', lightColor: 'bg-pink-100', textColor: 'text-pink-700' }
};

/**
 * Results Section Component
 * 
 * Displays bias analysis results with visual bars
 */
function ResultsSection({ results }) {
  if (!results) return null;

  const { biases, detected_biases, summary } = results;

  // Sort biases by score (highest first)
  const sortedBiases = Object.entries(biases)
    .sort(([, a], [, b]) => b.score - a.score);

  return (
    <div className="mt-8 fade-in">
      {/* Summary Card */}
      <div className={`p-6 rounded-xl mb-6 ${
        detected_biases.length > 0 
          ? 'bg-amber-50 border border-amber-200' 
          : 'bg-green-50 border border-green-200'
      }`}>
        <div className="flex items-start">
          {detected_biases.length > 0 ? (
            <svg className="w-6 h-6 text-amber-500 mt-0.5 mr-3 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          ) : (
            <svg className="w-6 h-6 text-green-500 mt-0.5 mr-3 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          )}
          <div>
            <h3 className={`font-semibold ${
              detected_biases.length > 0 ? 'text-amber-800' : 'text-green-800'
            }`}>
              Analysis Complete
            </h3>
            <p className={`mt-1 ${
              detected_biases.length > 0 ? 'text-amber-700' : 'text-green-700'
            }`}>
              {summary}
            </p>
            {detected_biases.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {detected_biases.map(bias => (
                  <span
                    key={bias}
                    className={`px-3 py-1 rounded-full text-sm font-medium ${BIAS_CONFIG[bias].lightColor} ${BIAS_CONFIG[bias].textColor}`}
                  >
                    {BIAS_CONFIG[bias].name}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Detailed Scores */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Bias Scores</h3>
          <p className="text-sm text-gray-500 mt-1">
            Scores above 0.5 (50%) indicate detected bias
          </p>
        </div>

        <div className="p-6 space-y-5">
          {sortedBiases.map(([biasType, data]) => (
            <BiasScoreBar
              key={biasType}
              biasType={biasType}
              score={data.score}
              detected={data.detected}
            />
          ))}
        </div>
      </div>

      {/* Score Legend */}
      <div className="mt-4 flex items-center justify-center text-sm text-gray-500 space-x-6">
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
          <span>Low (0-30%)</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-yellow-500 mr-2"></div>
          <span>Medium (30-50%)</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
          <span>High (50%+)</span>
        </div>
      </div>
    </div>
  );
}

/**
 * Individual Bias Score Bar
 */
function BiasScoreBar({ biasType, score, detected }) {
  const config = BIAS_CONFIG[biasType];
  const percentage = Math.round(score * 100);
  
  // Determine bar color based on score
  let barColor = 'bg-green-500';
  if (score >= 0.5) {
    barColor = 'bg-red-500';
  } else if (score >= 0.3) {
    barColor = 'bg-yellow-500';
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center">
          <span className="font-medium text-gray-900">{config.name}</span>
          {detected && (
            <span className="ml-2 px-2 py-0.5 bg-red-100 text-red-700 text-xs font-medium rounded-full">
              Detected
            </span>
          )}
        </div>
        <span className={`font-semibold ${detected ? 'text-red-600' : 'text-gray-600'}`}>
          {percentage}%
        </span>
      </div>
      <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full ${barColor} bias-bar rounded-full`}
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
}

/**
 * PropTypes for BiasScoreBar
 */
BiasScoreBar.propTypes = {
  biasType: PropTypes.oneOf([
    'political', 'gender', 'entity', 'racial', 'religious', 'regional', 'sensationalism'
  ]).isRequired,
  score: PropTypes.number.isRequired,
  detected: PropTypes.bool.isRequired
};

/**
 * PropTypes for ResultsSection
 */
ResultsSection.propTypes = {
  results: PropTypes.shape({
    biases: PropTypes.objectOf(
      PropTypes.shape({
        score: PropTypes.number.isRequired,
        detected: PropTypes.bool.isRequired
      })
    ).isRequired,
    detected_biases: PropTypes.arrayOf(PropTypes.string).isRequired,
    summary: PropTypes.string.isRequired
  })
};

ResultsSection.defaultProps = {
  results: null
};

export default ResultsSection;
