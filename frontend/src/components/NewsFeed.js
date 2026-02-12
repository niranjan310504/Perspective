import React, { useState, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';

/**
 * Political lean colors and labels
 */
const LEAN_CONFIG = {
  'left': { 
    color: 'bg-blue-500', 
    lightColor: 'bg-blue-100', 
    textColor: 'text-blue-700',
    label: 'Left-leaning'
  },
  'center-left': { 
    color: 'bg-teal-500', 
    lightColor: 'bg-teal-100', 
    textColor: 'text-teal-700',
    label: 'Center-Left'
  },
  'center': { 
    color: 'bg-gray-500', 
    lightColor: 'bg-gray-100', 
    textColor: 'text-gray-700',
    label: 'Center'
  },
  'right': { 
    color: 'bg-red-500', 
    lightColor: 'bg-red-100', 
    textColor: 'text-red-700',
    label: 'Right-leaning'
  },
  'unknown': { 
    color: 'bg-gray-400', 
    lightColor: 'bg-gray-100', 
    textColor: 'text-gray-600',
    label: 'Unknown'
  }
};

/**
 * Single news article card
 */
function NewsCard({ article, onAnalyze }) {
  const lean = LEAN_CONFIG[article.source_lean] || LEAN_CONFIG.unknown;
  
  // Format date
  const formatDate = (dateStr) => {
    if (!dateStr) return '';
    try {
      const date = new Date(dateStr);
      const now = new Date();
      const diffMs = now - date;
      const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
      const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
      
      if (diffHours < 1) return 'Just now';
      if (diffHours < 24) return `${diffHours}h ago`;
      if (diffDays < 7) return `${diffDays}d ago`;
      return date.toLocaleDateString();
    } catch {
      return '';
    }
  };

  return (
    <article className="bg-white rounded-xl border border-gray-200 overflow-hidden hover:shadow-lg transition-shadow duration-200 flex flex-col">
      {/* Image */}
      {article.image_url && (
        <div className="h-40 overflow-hidden bg-gray-100">
          <img 
            src={article.image_url} 
            alt=""
            className="w-full h-full object-cover"
            onError={(e) => e.target.style.display = 'none'}
          />
        </div>
      )}
      
      {/* Content */}
      <div className="p-4 flex-grow flex flex-col">
        {/* Source & Lean badge */}
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-600">
            {article.source}
          </span>
          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${lean.lightColor} ${lean.textColor}`}>
            {lean.label}
          </span>
        </div>
        
        {/* Title */}
        <h3 className="font-semibold text-gray-900 mb-2 line-clamp-2 flex-grow">
          <a 
            href={article.url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="hover:text-blue-600 transition-colors"
          >
            {article.title}
          </a>
        </h3>
        
        {/* Description */}
        {article.description && (
          <p className="text-gray-600 text-sm mb-3 line-clamp-2">
            {article.description}
          </p>
        )}
        
        {/* Footer */}
        <div className="flex items-center justify-between mt-auto pt-3 border-t border-gray-100">
          <span className="text-xs text-gray-500">
            {formatDate(article.published_at)}
          </span>
          <div className="flex gap-2">
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="px-3 py-1.5 bg-gray-100 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-200 transition-colors inline-flex items-center gap-1"
            >
              Read
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
            <button
              onClick={() => onAnalyze(article)}
              className="px-3 py-1.5 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition-colors"
            >
              Analyze Bias
            </button>
          </div>
        </div>
      </div>
    </article>
  );
}

NewsCard.propTypes = {
  article: PropTypes.shape({
    id: PropTypes.string.isRequired,
    title: PropTypes.string.isRequired,
    description: PropTypes.string,
    url: PropTypes.string.isRequired,
    source: PropTypes.string.isRequired,
    source_lean: PropTypes.string.isRequired,
    published_at: PropTypes.string,
    image_url: PropTypes.string,
    category: PropTypes.string
  }).isRequired,
  onAnalyze: PropTypes.func.isRequired
};

/**
 * Filter tabs component
 */
function FilterTabs({ activeFilter, onFilterChange, type }) {
  const filters = type === 'lean' 
    ? [
        { value: null, label: 'All' },
        { value: 'left', label: '← Left' },
        { value: 'center', label: 'Center' },
        { value: 'right', label: 'Right →' }
      ]
    : [
        { value: null, label: 'All' },
        { value: 'general', label: 'General' },
        { value: 'india', label: 'India' },
        { value: 'world', label: 'World' },
        { value: 'opinion', label: 'Opinion' }
      ];

  return (
    <div className="flex gap-2 flex-wrap">
      {filters.map(filter => (
        <button
          key={filter.value || 'all'}
          onClick={() => onFilterChange(filter.value)}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            activeFilter === filter.value
              ? 'bg-indigo-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          {filter.label}
        </button>
      ))}
    </div>
  );
}

FilterTabs.propTypes = {
  activeFilter: PropTypes.string,
  onFilterChange: PropTypes.func.isRequired,
  type: PropTypes.oneOf(['lean', 'category']).isRequired
};

/**
 * News Feed Component
 */
function NewsFeed({ apiBaseUrl, onAnalyzeArticle }) {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [leanFilter, setLeanFilter] = useState(null);
  const [categoryFilter, setCategoryFilter] = useState(null);
  const [lastFetched, setLastFetched] = useState(null);

  const fetchNews = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (leanFilter) params.append('lean', leanFilter);
      if (categoryFilter) params.append('category', categoryFilter);
      params.append('limit', '50');

      const url = `${apiBaseUrl}/news/feed?${params.toString()}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error('Failed to fetch news');
      }

      const data = await response.json();
      
      if (data.success) {
        setArticles(data.data.articles);
        setLastFetched(new Date());
      } else {
        throw new Error(data.error || 'Unknown error');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, leanFilter, categoryFilter]);

  // Fetch on mount and when filters change
  useEffect(() => {
    fetchNews();
  }, [fetchNews]);

  // Auto-refresh every 5 minutes
  useEffect(() => {
    const interval = setInterval(fetchNews, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchNews]);

  const handleAnalyze = (article) => {
    onAnalyzeArticle({
      url: article.url,
      title: article.title,
      source: article.source,
      sourceLean: article.source_lean
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Live News Feed</h2>
          <p className="text-gray-600 text-sm mt-1">
            Real-time news from multiple sources across the political spectrum
          </p>
        </div>
        <button
          onClick={fetchNews}
          disabled={loading}
          aria-label="Refresh news feed"
          className="flex items-center gap-2 px-4 py-2 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50"
        >
          <svg 
            className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="bg-gray-50 rounded-xl p-4 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Filter by Political Lean:
          </label>
          <FilterTabs 
            activeFilter={leanFilter} 
            onFilterChange={setLeanFilter}
            type="lean"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Filter by Category:
          </label>
          <FilterTabs 
            activeFilter={categoryFilter} 
            onFilterChange={setCategoryFilter}
            type="category"
          />
        </div>
      </div>

      {/* Source diversity indicator */}
      <div className="flex items-center justify-between text-sm text-gray-600">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
            Left
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-gray-500"></span>
            Center
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-red-500"></span>
            Right
          </span>
        </div>
        {lastFetched && (
          <span>
            Last updated: {lastFetched.toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* Error state */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700">
          <p className="font-medium">Failed to load news</p>
          <p className="text-sm mt-1">{error}</p>
          <button 
            onClick={fetchNews}
            className="mt-2 text-sm underline hover:no-underline"
          >
            Try again
          </button>
        </div>
      )}

      {/* Loading state */}
      {loading && !articles.length && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="bg-white rounded-xl border border-gray-200 overflow-hidden animate-pulse">
              <div className="h-40 bg-gray-200"></div>
              <div className="p-4 space-y-3">
                <div className="h-4 bg-gray-200 rounded w-1/3"></div>
                <div className="h-5 bg-gray-200 rounded"></div>
                <div className="h-5 bg-gray-200 rounded w-3/4"></div>
                <div className="h-4 bg-gray-200 rounded w-1/2"></div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Articles grid */}
      {!loading && articles.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {articles.map(article => (
            <NewsCard 
              key={article.id} 
              article={article}
              onAnalyze={handleAnalyze}
            />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && articles.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">No articles found with the selected filters.</p>
          <button 
            onClick={() => { setLeanFilter(null); setCategoryFilter(null); }}
            className="mt-2 text-indigo-600 hover:underline"
          >
            Clear filters
          </button>
        </div>
      )}
    </div>
  );
}

NewsFeed.propTypes = {
  apiBaseUrl: PropTypes.string.isRequired,
  onAnalyzeArticle: PropTypes.func.isRequired
};

export default NewsFeed;
