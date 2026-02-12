import React from 'react';
import PropTypes from 'prop-types';

/**
 * Status configuration for fact-check results
 */
const STATUS_CONFIG = {
  disputed: {
    icon: '❌',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
    textColor: 'text-red-800',
    badgeColor: 'bg-red-100 text-red-700',
    title: 'Disputed Content'
  },
  mixed: {
    icon: '⚠️',
    bgColor: 'bg-yellow-50',
    borderColor: 'border-yellow-200',
    textColor: 'text-yellow-800',
    badgeColor: 'bg-yellow-100 text-yellow-700',
    title: 'Mixed Ratings'
  },
  verified: {
    icon: '✅',
    bgColor: 'bg-green-50',
    borderColor: 'border-green-200',
    textColor: 'text-green-800',
    badgeColor: 'bg-green-100 text-green-700',
    title: 'Verified Content'
  },
  found: {
    icon: '🔍',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
    textColor: 'text-blue-800',
    badgeColor: 'bg-blue-100 text-blue-700',
    title: 'Fact-Checks Found'
  },
  not_found: {
    icon: '❓',
    bgColor: 'bg-gray-50',
    borderColor: 'border-gray-200',
    textColor: 'text-gray-700',
    badgeColor: 'bg-gray-100 text-gray-600',
    title: 'Not Fact-Checked Yet'
  },
  unavailable: {
    icon: '⚙️',
    bgColor: 'bg-gray-50',
    borderColor: 'border-gray-200',
    textColor: 'text-gray-600',
    badgeColor: 'bg-gray-100 text-gray-500',
    title: 'Fact-Check Unavailable'
  },
  no_claims: {
    icon: '📝',
    bgColor: 'bg-gray-50',
    borderColor: 'border-gray-200',
    textColor: 'text-gray-600',
    badgeColor: 'bg-gray-100 text-gray-500',
    title: 'No Verifiable Claims'
  }
};

/**
 * Fact Check Section Component
 * 
 * Displays Google Fact Check API results
 */
function FactCheckSection({ factCheck }) {
  if (!factCheck) return null;

  const config = STATUS_CONFIG[factCheck.status] || STATUS_CONFIG.unavailable;
  const hasFactChecks = factCheck.fact_checks && factCheck.fact_checks.length > 0;

  return (
    <div className="mt-6 fade-in">
      {/* Main Status Card */}
      <div className={`p-5 rounded-xl ${config.bgColor} border ${config.borderColor}`}>
        <div className="flex items-start">
          <span className="text-2xl mr-3">{config.icon}</span>
          <div className="flex-grow">
            <div className="flex items-center justify-between">
              <h3 className={`font-semibold ${config.textColor}`}>
                {config.title}
              </h3>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${config.badgeColor}`}>
                Fact Check
              </span>
            </div>
            <p className={`mt-1 text-sm ${config.textColor} opacity-90`}>
              {factCheck.message}
            </p>
            
            {/* Warning for not found */}
            {factCheck.status === 'not_found' && (
              <p className="mt-2 text-xs text-gray-500 italic">
                ⚠️ No fact-check found doesn't mean the content is accurate. 
                We recommend verifying from multiple trusted sources.
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Fact Check Details */}
      {hasFactChecks && (
        <div className="mt-4 bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <div className="p-4 border-b border-gray-200 bg-gray-50">
            <h4 className="font-medium text-gray-900">
              Related Fact-Checks ({factCheck.fact_checks.length})
            </h4>
            <p className="text-xs text-gray-500 mt-1">
              From professional fact-checking organizations
            </p>
          </div>
          
          <div className="divide-y divide-gray-100">
            {factCheck.fact_checks.map((fc, index) => (
              <FactCheckItem key={index} item={fc} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Individual Fact Check Item
 */
function FactCheckItem({ item }) {
  // Determine rating color
  const getRatingColor = (rating) => {
    const ratingLower = rating.toLowerCase();
    if (['false', 'fake', 'misleading', 'incorrect', 'pants on fire'].some(r => ratingLower.includes(r))) {
      return 'bg-red-100 text-red-700';
    }
    if (['true', 'correct', 'accurate', 'verified'].some(r => ratingLower.includes(r))) {
      return 'bg-green-100 text-green-700';
    }
    if (['partly', 'partial', 'mixed', 'half'].some(r => ratingLower.includes(r))) {
      return 'bg-yellow-100 text-yellow-700';
    }
    return 'bg-gray-100 text-gray-700';
  };

  return (
    <div className="p-4 hover:bg-gray-50 transition-colors">
      <div className="flex items-start justify-between">
        <div className="flex-grow pr-4">
          <p className="text-sm text-gray-800 font-medium line-clamp-2">
            "{item.claim}"
          </p>
          {item.claimant && item.claimant !== 'Unknown' && (
            <p className="text-xs text-gray-500 mt-1">
              — {item.claimant}
            </p>
          )}
        </div>
        <span className={`px-2 py-1 rounded text-xs font-medium whitespace-nowrap ${getRatingColor(item.rating)}`}>
          {item.rating}
        </span>
      </div>
      
      <div className="mt-2 flex items-center justify-between text-xs">
        <span className="text-gray-500">
          Checked by: <span className="font-medium">{item.publisher}</span>
        </span>
        <a
          href={item.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary-600 hover:text-primary-800 font-medium flex items-center"
        >
          View Details
          <svg className="w-3 h-3 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
        </a>
      </div>
    </div>
  );
}

/**
 * PropTypes
 */
FactCheckSection.propTypes = {
  factCheck: PropTypes.shape({
    status: PropTypes.oneOf(['disputed', 'mixed', 'verified', 'found', 'not_found', 'unavailable', 'no_claims', 'not_checked']),
    message: PropTypes.string,
    fact_checks: PropTypes.arrayOf(PropTypes.shape({
      claim: PropTypes.string,
      claimant: PropTypes.string,
      rating: PropTypes.string,
      publisher: PropTypes.string,
      url: PropTypes.string,
      title: PropTypes.string
    })),
    claims_searched: PropTypes.number
  })
};

FactCheckSection.defaultProps = {
  factCheck: null
};

FactCheckItem.propTypes = {
  item: PropTypes.shape({
    claim: PropTypes.string.isRequired,
    claimant: PropTypes.string,
    rating: PropTypes.string.isRequired,
    publisher: PropTypes.string.isRequired,
    url: PropTypes.string.isRequired
  }).isRequired
};

export default FactCheckSection;
