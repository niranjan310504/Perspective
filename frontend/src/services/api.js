import axios from 'axios';

/**
 * API Configuration
 */
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // ms

/**
 * Axios instance with default config
 */
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

/**
 * Sleep utility for retry delays
 */
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Retry wrapper with exponential backoff
 * 
 * @param {Function} fn - Async function to retry
 * @param {number} retries - Number of retries
 * @param {number} delay - Initial delay in ms
 * @returns {Promise} Result of fn
 */
const withRetry = async (fn, retries = MAX_RETRIES, delay = RETRY_DELAY) => {
  try {
    return await fn();
  } catch (error) {
    // Don't retry on client errors (4xx) except 429 (rate limit)
    if (error.response && error.response.status >= 400 && error.response.status < 500 && error.response.status !== 429) {
      throw error;
    }
    
    if (retries > 0) {
      console.log(`Retrying... ${retries} attempts left`);
      await sleep(delay);
      return withRetry(fn, retries - 1, delay * 2); // Exponential backoff
    }
    throw error;
  }
};

/**
 * Analyze text for media bias
 * 
 * @param {Object} payload - { text: string } or { url: string }
 * @returns {Promise<Object>} Analysis results
 */
export const analyzeText = async (payload) => {
  try {
    const response = await withRetry(() => api.post('/analyze', payload));
    return response.data;
  } catch (error) {
    if (error.response) {
      // Server responded with error
      const status = error.response.status;
      const data = error.response.data;
      
      if (status === 429) {
        return {
          success: false,
          error: 'Too many requests. Please wait a moment and try again.'
        };
      }
      
      return {
        success: false,
        error: data.error || `Server error (${status})`
      };
    } else if (error.request) {
      // No response received
      return {
        success: false,
        error: 'Unable to connect to the server. Please ensure the backend is running.'
      };
    } else {
      // Request setup error
      return {
        success: false,
        error: error.message
      };
    }
  }
};

/**
 * Get all bias type information
 * 
 * @returns {Promise<Object>} Bias type descriptions
 */
export const getBiasTypes = async () => {
  try {
    const response = await withRetry(() => api.get('/bias-types'));
    return response.data;
  } catch (error) {
    console.error('Failed to fetch bias types:', error);
    return { success: false, error: 'Failed to fetch bias types' };
  }
};

/**
 * Health check
 * 
 * @returns {Promise<Object>} Server health status
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    return { status: 'unhealthy', model_loaded: false };
  }
};

/**
 * Check if API is available
 * 
 * @returns {Promise<boolean>}
 */
export const isApiAvailable = async () => {
  try {
    const health = await checkHealth();
    return health.status === 'healthy';
  } catch {
    return false;
  }
};

export default api;
