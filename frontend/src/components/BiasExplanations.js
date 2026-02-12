import React, { useState } from 'react';
import PropTypes from 'prop-types';

/**
 * Bias type explanations
 */
const BIAS_EXPLANATIONS = {
  political: {
    name: 'Political Bias',
    description: 'Favoring or opposing political parties, ideologies, leaders, or policies.',
    examples: [
      'The government\'s visionary policies have transformed the nation...',
      'Opposition\'s failed policies continue to harm citizens...'
    ],
    indicators: [
      'Loaded language about political entities',
      'One-sided coverage of political events',
      'Missing context for opposing viewpoints'
    ]
  },
  gender: {
    name: 'Gender Bias',
    description: 'Stereotyping, unequal representation, or discrimination based on gender.',
    examples: [
      'Lady doctor saves patient...',
      'The emotional woman leader...'
    ],
    indicators: [
      'Unnecessary gender mentions',
      'Gender-based stereotypes',
      'Unequal coverage of achievements'
    ]
  },
  entity: {
    name: 'Entity Bias',
    description: 'Undue favor or criticism toward specific organizations, companies, or public figures.',
    examples: [
      'The star-studded gala organized by [Company]...',
      'The controversial businessman\'s empire...'
    ],
    indicators: [
      'Excessive praise or criticism',
      'Missing balanced perspectives',
      'Promotional or hit-piece language'
    ]
  },
  racial: {
    name: 'Racial/Caste Bias',
    description: 'Discrimination or stereotyping based on race, caste, or ethnicity.',
    examples: [
      'The dark-skinned suspect...',
      'People from that community are known to...'
    ],
    indicators: [
      'Unnecessary racial descriptors',
      'Ethnic stereotypes',
      'Caste-based assumptions'
    ]
  },
  religious: {
    name: 'Religious Bias',
    description: 'Favoring or targeting specific religious groups or beliefs.',
    examples: [
      'The peace-loving majority community...',
      'Members of the minority community allegedly...'
    ],
    indicators: [
      'Religious identity when irrelevant',
      'Generalizations about religious groups',
      'Unequal framing of religious events'
    ]
  },
  regional: {
    name: 'Regional Bias',
    description: 'Bias toward or against specific states, regions, or linguistic groups.',
    examples: [
      'As expected from people of that state...',
      'The backward regions of...'
    ],
    indicators: [
      'Regional stereotypes',
      'State-based generalizations',
      'Urban vs rural bias'
    ]
  },
  sensationalism: {
    name: 'Sensationalism',
    description: 'Exaggeration, clickbait, emotional manipulation, or fear-mongering.',
    examples: [
      'SHOCKING revelations emerge...',
      'You won\'t BELIEVE what happened next!'
    ],
    indicators: [
      'All-caps or excessive punctuation',
      'Emotional manipulation',
      'Exaggerated claims without evidence'
    ]
  }
};

/**
 * Bias Explanations Component
 * 
 * Collapsible section explaining each bias type
 */
function BiasExplanations() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedBias, setSelectedBias] = useState(null);

  return (
    <div id="about" className="mt-12">
      {/* Section Header */}
      <button
        className="w-full flex items-center justify-between p-4 bg-white rounded-xl shadow-sm border border-gray-200 hover:bg-gray-50 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center">
          <svg className="w-6 h-6 text-primary-600 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="text-left">
            <h3 className="font-semibold text-gray-900">Understanding Bias Types</h3>
            <p className="text-sm text-gray-500">Learn what each bias category means</p>
          </div>
        </div>
        <svg
          className={`w-5 h-5 text-gray-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="mt-4 bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden fade-in">
          {/* Bias Type Tabs */}
          <div className="flex flex-wrap border-b border-gray-200 bg-gray-50">
            {Object.entries(BIAS_EXPLANATIONS).map(([key, bias]) => (
              <button
                key={key}
                className={`px-4 py-3 text-sm font-medium transition-colors ${
                  selectedBias === key
                    ? 'text-primary-600 bg-white border-b-2 border-primary-600'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
                onClick={() => setSelectedBias(selectedBias === key ? null : key)}
              >
                {bias.name}
              </button>
            ))}
          </div>

          {/* Bias Detail */}
          {selectedBias ? (
            <BiasDetail bias={BIAS_EXPLANATIONS[selectedBias]} />
          ) : (
            <div className="p-6 text-center text-gray-500">
              Click on a bias type above to learn more
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Detailed view for a single bias type
 */
function BiasDetail({ bias }) {
  return (
    <div className="p-6 fade-in">
      <h4 className="text-lg font-semibold text-gray-900 mb-2">{bias.name}</h4>
      <p className="text-gray-600 mb-6">{bias.description}</p>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Examples */}
        <div>
          <h5 className="font-medium text-gray-900 mb-3 flex items-center">
            <svg className="w-4 h-4 mr-2 text-amber-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            Examples
          </h5>
          <ul className="space-y-2">
            {bias.examples.map((example, idx) => (
              <li key={idx} className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg italic">
                "{example}"
              </li>
            ))}
          </ul>
        </div>

        {/* Indicators */}
        <div>
          <h5 className="font-medium text-gray-900 mb-3 flex items-center">
            <svg className="w-4 h-4 mr-2 text-primary-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            Key Indicators
          </h5>
          <ul className="space-y-2">
            {bias.indicators.map((indicator, idx) => (
              <li key={idx} className="text-sm text-gray-600 flex items-start">
                <span className="w-1.5 h-1.5 bg-primary-500 rounded-full mt-1.5 mr-2 flex-shrink-0"></span>
                {indicator}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

/**
 * PropTypes for BiasDetail
 */
BiasDetail.propTypes = {
  bias: PropTypes.shape({
    name: PropTypes.string.isRequired,
    description: PropTypes.string.isRequired,
    examples: PropTypes.arrayOf(PropTypes.string).isRequired,
    indicators: PropTypes.arrayOf(PropTypes.string).isRequired
  }).isRequired
};

export default BiasExplanations;
