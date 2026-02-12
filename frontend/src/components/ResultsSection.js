import React, { useState } from 'react';
import PropTypes from 'prop-types';

/**
 * Bias type configuration with detailed explanations
 */
const BIAS_CONFIG = {
  political: { 
    name: 'Political Bias', 
    color: 'bg-red-500', 
    lightColor: 'bg-red-100', 
    textColor: 'text-red-700',
    icon: '🏛️',
    whatItMeans: 'This article appears to favor or oppose a specific political party, leader, or ideology.',
    signs: [
      'Uses loaded language about political figures',
      'Presents one-sided view of political events',
      'Missing context from opposing viewpoints'
    ],
    recommendation: 'Read coverage of this story from media outlets with different political leanings to get a balanced view.'
  },
  gender: { 
    name: 'Gender Bias', 
    color: 'bg-purple-500', 
    lightColor: 'bg-purple-100', 
    textColor: 'text-purple-700',
    icon: '⚧️',
    whatItMeans: 'This article contains stereotyping or unequal representation based on gender.',
    signs: [
      'Unnecessary mentions of gender (e.g., "lady doctor")',
      'Gender-based stereotypes or assumptions',
      'Different treatment of achievements based on gender'
    ],
    recommendation: 'Be aware that gender framing can subtly influence perception. Focus on the facts, not gendered descriptions.'
  },
  entity: { 
    name: 'Entity Bias', 
    color: 'bg-blue-500', 
    lightColor: 'bg-blue-100', 
    textColor: 'text-blue-700',
    icon: '🏢',
    whatItMeans: 'This article shows undue favor or criticism toward specific companies, organizations, or public figures.',
    signs: [
      'Excessive praise or criticism of an entity',
      'Missing perspectives from affected parties',
      'Promotional or hit-piece style language'
    ],
    recommendation: 'Check if this outlet has any business relationship with the entity mentioned. Look for more neutral coverage.'
  },
  racial: { 
    name: 'Racial/Caste Bias', 
    color: 'bg-orange-500', 
    lightColor: 'bg-orange-100', 
    textColor: 'text-orange-700',
    icon: '👥',
    whatItMeans: 'This article contains discrimination or stereotyping based on race, caste, or ethnicity.',
    signs: [
      'Unnecessary racial/caste descriptors',
      'Ethnic or caste-based stereotypes',
      'Associating crime/negative traits with specific groups'
    ],
    recommendation: 'This is a serious form of bias. Question why ethnic/caste identity is being highlighted if irrelevant to the story.'
  },
  religious: { 
    name: 'Religious Bias', 
    color: 'bg-green-500', 
    lightColor: 'bg-green-100', 
    textColor: 'text-green-700',
    icon: '🛕',
    whatItMeans: 'This article favors or targets specific religious communities or beliefs.',
    signs: [
      'Religious identity mentioned when irrelevant',
      'Generalizations about religious groups',
      'Unequal framing of religious events or practices'
    ],
    recommendation: 'Be cautious of content that frames religious groups negatively or positively without factual basis. Verify from secular sources.'
  },
  regional: { 
    name: 'Regional Bias', 
    color: 'bg-yellow-500', 
    lightColor: 'bg-yellow-100', 
    textColor: 'text-yellow-700',
    icon: '🗺️',
    whatItMeans: 'This article shows bias toward or against specific states, regions, or linguistic groups.',
    signs: [
      'Regional stereotypes (e.g., "as expected from that state")',
      'Urban vs rural bias',
      'State-based generalizations'
    ],
    recommendation: 'India is diverse. Be wary of articles that paint entire regions with broad strokes.'
  },
  sensationalism: { 
    name: 'Sensationalism', 
    color: 'bg-pink-500', 
    lightColor: 'bg-pink-100', 
    textColor: 'text-pink-700',
    icon: '📢',
    whatItMeans: 'This article uses exaggeration, clickbait, or emotional manipulation to get reactions.',
    signs: [
      'ALL CAPS or excessive punctuation!!!',
      'Emotional, fear-inducing language',
      'Exaggerated claims without evidence'
    ],
    recommendation: 'Sensational articles often sacrifice accuracy for clicks. Look for the same story from more measured sources.'
  }
};

/**
 * Get severity level from score
 */
const getSeverity = (score) => {
  if (score >= 0.7) return { level: 'High', color: 'text-red-600', bg: 'bg-red-100', icon: '🔴' };
  if (score >= 0.5) return { level: 'Moderate', color: 'text-orange-600', bg: 'bg-orange-100', icon: '🟠' };
  if (score >= 0.3) return { level: 'Low', color: 'text-yellow-600', bg: 'bg-yellow-100', icon: '🟡' };
  return { level: 'Minimal', color: 'text-green-600', bg: 'bg-green-100', icon: '🟢' };
};

/**
 * Results Section Component
 */
function ResultsSection({ results }) {
  const [expandedBias, setExpandedBias] = useState(null);
  
  if (!results) return null;

  const { biases, detected_biases } = results;

  // Sort biases by score (highest first)
  const sortedBiases = Object.entries(biases)
    .sort(([, a], [, b]) => b.score - a.score);

  // Get the highest detected biases for the summary
  const topBiases = sortedBiases
    .filter(([, data]) => data.detected)
    .slice(0, 3);

  return (
    <div className="mt-8 fade-in">
      {/* Main Summary Card */}
      <div className={`p-6 rounded-xl mb-6 ${
        detected_biases.length > 0 
          ? 'bg-amber-50 border-2 border-amber-300' 
          : 'bg-green-50 border-2 border-green-300'
      }`}>
        {detected_biases.length > 0 ? (
          <>
            <div className="flex items-center mb-4">
              <span className="text-3xl mr-3">⚠️</span>
              <div>
                <h3 className="text-xl font-bold text-amber-800">
                  Bias Detected - Read With Caution
                </h3>
                <p className="text-amber-700 text-sm">
                  This article shows signs of biased reporting
                </p>
              </div>
            </div>
            
            {/* What was detected */}
            <div className="bg-white/60 rounded-lg p-4 mb-4">
              <h4 className="font-semibold text-gray-800 mb-2">What we found:</h4>
              <div className="space-y-2">
                {topBiases.map(([biasType, data]) => {
                  const config = BIAS_CONFIG[biasType];
                  const severity = getSeverity(data.score);
                  return (
                    <div key={biasType} className="flex items-center">
                      <span className="mr-2">{severity.icon}</span>
                      <span className="font-medium">{config.name}:</span>
                      <span className={`ml-2 ${severity.color} font-semibold`}>
                        {severity.level} ({Math.round(data.score * 100)}%)
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* What should you do */}
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
              <h4 className="font-semibold text-blue-800 flex items-center mb-2">
                <span className="mr-2">💡</span>
                What should you do?
              </h4>
              <ul className="text-blue-700 text-sm space-y-1">
                <li>• <strong>Don't form opinions yet</strong> - This is one perspective</li>
                <li>• <strong>Cross-check</strong> this story with 2-3 other news sources</li>
                <li>• <strong>Look for primary sources</strong> - official statements, documents</li>
                <li>• <strong>Check the facts</strong> - Are claims backed by evidence?</li>
              </ul>
            </div>
          </>
        ) : (
          <div className="flex items-center">
            <span className="text-3xl mr-3">✅</span>
            <div>
              <h3 className="text-xl font-bold text-green-800">
                Low Bias Detected
              </h3>
              <p className="text-green-700">
                This article appears to be relatively balanced. However, always verify important claims from multiple sources.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Detailed Breakdown */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-5 border-b border-gray-200 bg-gray-50">
          <h3 className="text-lg font-semibold text-gray-900">Detailed Analysis</h3>
          <p className="text-sm text-gray-500 mt-1">
            Click on any bias type to learn more about what was detected
          </p>
        </div>

        <div className="divide-y divide-gray-100">
          {sortedBiases.map(([biasType, data]) => (
            <BiasDetailRow
              key={biasType}
              biasType={biasType}
              score={data.score}
              detected={data.detected}
              isExpanded={expandedBias === biasType}
              onToggle={() => setExpandedBias(expandedBias === biasType ? null : biasType)}
            />
          ))}
        </div>
      </div>

      {/* Understanding the scores */}
      <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
        <h4 className="font-medium text-gray-700 mb-2">Understanding the scores:</h4>
        <div className="flex flex-wrap gap-4 text-sm">
          <div className="flex items-center">
            <span className="mr-1">🟢</span>
            <span className="text-gray-600">0-30%: Minimal bias</span>
          </div>
          <div className="flex items-center">
            <span className="mr-1">🟡</span>
            <span className="text-gray-600">30-50%: Low bias</span>
          </div>
          <div className="flex items-center">
            <span className="mr-1">🟠</span>
            <span className="text-gray-600">50-70%: Moderate bias</span>
          </div>
          <div className="flex items-center">
            <span className="mr-1">🔴</span>
            <span className="text-gray-600">70%+: High bias</span>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Individual Bias Detail Row with expandable explanation
 */
function BiasDetailRow({ biasType, score, detected, isExpanded, onToggle }) {
  const config = BIAS_CONFIG[biasType];
  const severity = getSeverity(score);
  const percentage = Math.round(score * 100);

  return (
    <div className={`${detected ? 'bg-red-50/30' : ''}`}>
      {/* Main Row - Clickable */}
      <button
        onClick={onToggle}
        className="w-full p-4 flex items-center justify-between hover:bg-gray-50 transition-colors text-left"
      >
        <div className="flex items-center flex-grow">
          <span className="text-xl mr-3">{config.icon}</span>
          <div className="flex-grow">
            <div className="flex items-center">
              <span className="font-medium text-gray-900">{config.name}</span>
              {detected && (
                <span className="ml-2 px-2 py-0.5 bg-red-100 text-red-700 text-xs font-bold rounded-full">
                  DETECTED
                </span>
              )}
            </div>
            {/* Progress bar */}
            <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden w-full max-w-xs">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  score >= 0.7 ? 'bg-red-500' : 
                  score >= 0.5 ? 'bg-orange-500' : 
                  score >= 0.3 ? 'bg-yellow-500' : 'bg-green-500'
                }`}
                style={{ width: `${percentage}%` }}
              />
            </div>
          </div>
        </div>
        
        <div className="flex items-center ml-4">
          <span className={`font-bold text-lg ${severity.color}`}>
            {percentage}%
          </span>
          <svg 
            className={`w-5 h-5 ml-2 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="px-4 pb-4 bg-gray-50 border-t border-gray-100">
          <div className="ml-9 space-y-4">
            {/* What it means */}
            <div>
              <h5 className="font-semibold text-gray-800 mb-1">What this means:</h5>
              <p className="text-gray-600 text-sm">{config.whatItMeans}</p>
            </div>

            {/* Signs to look for */}
            <div>
              <h5 className="font-semibold text-gray-800 mb-1">Signs in the article:</h5>
              <ul className="text-sm text-gray-600 space-y-1">
                {config.signs.map((sign, idx) => (
                  <li key={idx} className="flex items-start">
                    <span className="text-gray-400 mr-2">•</span>
                    {sign}
                  </li>
                ))}
              </ul>
            </div>

            {/* Recommendation */}
            {detected && (
              <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
                <h5 className="font-semibold text-blue-800 flex items-center mb-1">
                  <span className="mr-1">💡</span> Recommendation:
                </h5>
                <p className="text-blue-700 text-sm">{config.recommendation}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * PropTypes
 */
BiasDetailRow.propTypes = {
  biasType: PropTypes.oneOf([
    'political', 'gender', 'entity', 'racial', 'religious', 'regional', 'sensationalism'
  ]).isRequired,
  score: PropTypes.number.isRequired,
  detected: PropTypes.bool.isRequired,
  isExpanded: PropTypes.bool.isRequired,
  onToggle: PropTypes.func.isRequired
};

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
