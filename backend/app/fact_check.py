"""
Google Fact Check API Integration
==================================

Uses Google Fact Check Tools API to find existing fact-checks
for claims in news articles.

API Documentation: https://developers.google.com/fact-check/tools/api
"""

import os
import re
import logging
import requests
from typing import List, Dict, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# Google Fact Check API endpoint
FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


class FactChecker:
    """
    Google Fact Check API client.
    
    Searches for existing fact-checks related to the input text.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the fact checker.
        
        Args:
            api_key: Google API key. If not provided, reads from GOOGLE_FACT_CHECK_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('GOOGLE_FACT_CHECK_API_KEY')
        if not self.api_key:
            logger.warning("Google Fact Check API key not configured. Fact checking disabled.")
    
    def extract_key_claims(self, text: str, max_claims: int = 3) -> List[str]:
        """
        Extract key searchable phrases from the text.
        
        Args:
            text: Input article text
            max_claims: Maximum number of claims to extract
            
        Returns:
            List of key phrases to search
        """
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Strategy: Use first few sentences and any quoted statements
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        claims = []
        
        # Add first 1-2 sentences (usually contain main claim)
        for sent in sentences[:2]:
            # Limit length for API query
            if len(sent) > 200:
                sent = sent[:200]
            claims.append(sent)
        
        # Look for quoted statements (often key claims)
        quotes = re.findall(r'"([^"]{20,150})"', text)
        claims.extend(quotes[:1])
        
        return claims[:max_claims]
    
    def search_fact_checks(self, query: str, language: str = "en") -> List[Dict]:
        """
        Search Google Fact Check API for a query.
        
        Args:
            query: Search query
            language: Language code (default: en)
            
        Returns:
            List of fact-check results
        """
        if not self.api_key:
            return []
        
        try:
            params = {
                'key': self.api_key,
                'query': query,
                'languageCode': language,
                'pageSize': 5
            }
            
            response = requests.get(
                FACT_CHECK_API_URL,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('claims', [])
            else:
                logger.error(f"Fact Check API error: {response.status_code}")
                return []
                
        except requests.exceptions.Timeout:
            logger.warning("Fact Check API timeout")
            return []
        except Exception as e:
            logger.error(f"Fact Check API error: {e}")
            return []
    
    def check_article(self, text: str) -> Dict:
        """
        Check an article for related fact-checks.
        
        Args:
            text: Article text to check
            
        Returns:
            Fact check results with status and details
        """
        result = {
            'status': 'not_checked',
            'message': 'Fact checking not available',
            'fact_checks': [],
            'claims_searched': 0
        }
        
        if not self.api_key:
            result['status'] = 'unavailable'
            result['message'] = 'Fact check API not configured'
            return result
        
        # Extract key claims from the article
        claims = self.extract_key_claims(text)
        result['claims_searched'] = len(claims)
        
        if not claims:
            result['status'] = 'no_claims'
            result['message'] = 'No verifiable claims found in text'
            return result
        
        # Search for fact-checks on each claim
        all_fact_checks = []
        seen_urls = set()
        
        for claim in claims:
            fact_checks = self.search_fact_checks(claim)
            for fc in fact_checks:
                # Deduplicate by URL
                reviews = fc.get('claimReview', [])
                for review in reviews:
                    url = review.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_fact_checks.append({
                            'claim': fc.get('text', claim)[:200],
                            'claimant': fc.get('claimant', 'Unknown'),
                            'rating': review.get('textualRating', 'Unknown'),
                            'publisher': review.get('publisher', {}).get('name', 'Unknown'),
                            'url': url,
                            'title': review.get('title', ''),
                            'review_date': review.get('reviewDate', '')
                        })
        
        # Determine overall status
        if all_fact_checks:
            # Check if any are rated as false/misleading
            ratings_lower = [fc['rating'].lower() for fc in all_fact_checks]
            
            false_indicators = ['false', 'fake', 'misleading', 'incorrect', 'wrong', 'pants on fire', 'fabricated']
            true_indicators = ['true', 'correct', 'accurate', 'verified', 'fact']
            mixed_indicators = ['partly', 'partial', 'mixed', 'half']
            
            has_false = any(any(ind in r for ind in false_indicators) for r in ratings_lower)
            has_true = any(any(ind in r for ind in true_indicators) for r in ratings_lower)
            has_mixed = any(any(ind in r for ind in mixed_indicators) for r in ratings_lower)
            
            if has_false:
                result['status'] = 'disputed'
                result['message'] = 'Related claims have been disputed by fact-checkers'
            elif has_mixed:
                result['status'] = 'mixed'
                result['message'] = 'Related claims have mixed fact-check ratings'
            elif has_true:
                result['status'] = 'verified'
                result['message'] = 'Related claims have been verified by fact-checkers'
            else:
                result['status'] = 'found'
                result['message'] = 'Related fact-checks found'
            
            result['fact_checks'] = all_fact_checks[:5]  # Limit to top 5
        else:
            result['status'] = 'not_found'
            result['message'] = 'No existing fact-checks found for this content'
        
        return result


# Singleton instance
_fact_checker = None

def get_fact_checker() -> FactChecker:
    """Get or create the fact checker instance."""
    global _fact_checker
    if _fact_checker is None:
        _fact_checker = FactChecker()
    return _fact_checker


def check_facts(text: str) -> Dict:
    """
    Convenience function to check facts in text.
    
    Args:
        text: Article text
        
    Returns:
        Fact check results
    """
    checker = get_fact_checker()
    return checker.check_article(text)
