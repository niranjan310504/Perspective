"""
Dataset Schema Definition for Perspective
==========================================

This module defines the structure and validation rules for the 
Indian News Bias Detection dataset.
"""

# Column definitions for the dataset CSV
DATASET_SCHEMA = {
    "columns": {
        # Metadata columns
        "article_id": {
            "type": "string",
            "description": "Unique identifier for each article",
            "format": "ART_YYYYMMDD_XXXXX"
        },
        "source": {
            "type": "string",
            "description": "News outlet name (e.g., TimesOfIndia, NDTV, Republic)"
        },
        "url": {
            "type": "string",
            "description": "Original article URL"
        },
        "publish_date": {
            "type": "date",
            "description": "Publication date (YYYY-MM-DD)"
        },
        "category": {
            "type": "string",
            "description": "News category (politics, sports, entertainment, etc.)"
        },
        
        # Content columns
        "headline": {
            "type": "string",
            "description": "Article headline/title"
        },
        "content": {
            "type": "string",
            "description": "Full article text (cleaned)"
        },
        "word_count": {
            "type": "integer",
            "description": "Number of words in content"
        },
        
        # Binary bias labels (0 or 1)
        "label_political": {
            "type": "integer",
            "description": "Political bias present (0/1)"
        },
        "label_gender": {
            "type": "integer",
            "description": "Gender bias present (0/1)"
        },
        "label_entity": {
            "type": "integer",
            "description": "Entity bias present (0/1)"
        },
        "label_racial": {
            "type": "integer",
            "description": "Racial bias present (0/1)"
        },
        "label_religious": {
            "type": "integer",
            "description": "Religious bias present (0/1)"
        },
        "label_regional": {
            "type": "integer",
            "description": "Regional bias present (0/1)"
        },
        "label_sensationalism": {
            "type": "integer",
            "description": "Sensationalism present (0/1)"
        },
        
        # Labeling metadata
        "labeling_method": {
            "type": "string",
            "description": "How labels were generated (llm_gpt4, manual, hybrid)"
        },
        "confidence_score": {
            "type": "float",
            "description": "LLM confidence in labels (0.0 to 1.0)"
        },
        "verified": {
            "type": "boolean",
            "description": "Whether labels were human-verified"
        }
    }
}

# Bias label names in order
BIAS_LABELS = [
    "political",
    "gender",
    "entity",
    "racial",
    "religious",
    "regional",
    "sensationalism"
]

# Label column names
LABEL_COLUMNS = [f"label_{bias}" for bias in BIAS_LABELS]

# Mapping of bias types to detailed descriptions
BIAS_DESCRIPTIONS = {
    "political": {
        "name": "Political Bias",
        "description": "Favoring or opposing political parties, ideologies, leaders, or policies",
        "examples": [
            "The government's brilliant move...",
            "Opposition's failed policies...",
            "The ruling party's visionary leadership..."
        ],
        "indicators": [
            "Loaded language about political entities",
            "One-sided coverage of political events",
            "Missing context for opposing viewpoints"
        ]
    },
    "gender": {
        "name": "Gender Bias",
        "description": "Stereotyping, unequal representation, or discrimination based on gender",
        "examples": [
            "Lady doctor saves patient...",
            "The emotional woman leader...",
            "He, as expected from a man of his stature..."
        ],
        "indicators": [
            "Unnecessary gender mentions",
            "Gender-based stereotypes",
            "Unequal coverage of achievements"
        ]
    },
    "entity": {
        "name": "Entity Bias",
        "description": "Undue favor or criticism toward specific organizations, companies, or public figures",
        "examples": [
            "The star-studded gala organized by [Company]...",
            "The controversial businessman...",
            "The renowned institution's flawless track record..."
        ],
        "indicators": [
            "Excessive praise or criticism",
            "Missing balanced perspectives",
            "Promotional or hit-piece language"
        ]
    },
    "racial": {
        "name": "Racial Bias",
        "description": "Discrimination or stereotyping based on race, caste, or ethnicity",
        "examples": [
            "The dark-skinned suspect...",
            "People from that community are known to...",
            "The fair-complexioned beauty..."
        ],
        "indicators": [
            "Unnecessary racial descriptors",
            "Ethnic stereotypes",
            "Caste-based assumptions"
        ]
    },
    "religious": {
        "name": "Religious Bias",
        "description": "Favoring or targeting specific religious groups or beliefs",
        "examples": [
            "The peace-loving majority community...",
            "Members of the minority community allegedly...",
            "Religious fanatics from..."
        ],
        "indicators": [
            "Religious identity when irrelevant",
            "Generalizations about religious groups",
            "Unequal framing of religious events"
        ]
    },
    "regional": {
        "name": "Regional Bias",
        "description": "Bias toward or against specific states, regions, or linguistic groups",
        "examples": [
            "As expected from people of that state...",
            "The progressive southern states versus...",
            "The backward regions of..."
        ],
        "indicators": [
            "Regional stereotypes",
            "State-based generalizations",
            "Urban vs rural bias"
        ]
    },
    "sensationalism": {
        "name": "Sensationalism",
        "description": "Exaggeration, clickbait, emotional manipulation, or fear-mongering",
        "examples": [
            "SHOCKING revelations emerge...",
            "You won't BELIEVE what happened next!",
            "Crisis looms as disaster strikes..."
        ],
        "indicators": [
            "All-caps or excessive punctuation",
            "Emotional manipulation",
            "Exaggerated claims without evidence",
            "Clickbait headlines"
        ]
    }
}

# Indian news sources for data collection
NEWS_SOURCES = {
    "mainstream": [
        "Times of India",
        "Hindustan Times",
        "The Hindu",
        "Indian Express",
        "NDTV",
        "India Today"
    ],
    "right_leaning": [
        "Republic TV",
        "OpIndia",
        "Swarajya"
    ],
    "left_leaning": [
        "The Wire",
        "Scroll.in",
        "The Quint"
    ],
    "regional": [
        "The New Indian Express",
        "Deccan Herald",
        "The Telegraph"
    ]
}

# Validation rules
MIN_ARTICLE_LENGTH = 50  # words
MAX_ARTICLE_LENGTH = 5000  # words
MIN_CONFIDENCE_FOR_TRAINING = 0.7  # LLM confidence threshold


def validate_article(article_dict: dict) -> tuple[bool, list]:
    """
    Validate an article against the schema rules.
    
    Args:
        article_dict: Dictionary containing article data
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required fields
    required_fields = ["article_id", "content", "headline"]
    for field in required_fields:
        if field not in article_dict or not article_dict[field]:
            errors.append(f"Missing required field: {field}")
    
    # Check word count
    if "content" in article_dict:
        word_count = len(article_dict["content"].split())
        if word_count < MIN_ARTICLE_LENGTH:
            errors.append(f"Article too short: {word_count} words (min: {MIN_ARTICLE_LENGTH})")
        if word_count > MAX_ARTICLE_LENGTH:
            errors.append(f"Article too long: {word_count} words (max: {MAX_ARTICLE_LENGTH})")
    
    # Check label values
    for label_col in LABEL_COLUMNS:
        if label_col in article_dict:
            if article_dict[label_col] not in [0, 1]:
                errors.append(f"Invalid label value for {label_col}: must be 0 or 1")
    
    return len(errors) == 0, errors


if __name__ == "__main__":
    # Print schema summary
    print("=" * 60)
    print("PERSPECTIVE DATASET SCHEMA")
    print("=" * 60)
    
    print("\nBias Labels:")
    for bias in BIAS_LABELS:
        print(f"  - {bias}: {BIAS_DESCRIPTIONS[bias]['name']}")
    
    print(f"\nTotal columns: {len(DATASET_SCHEMA['columns'])}")
    print(f"Label columns: {len(LABEL_COLUMNS)}")
