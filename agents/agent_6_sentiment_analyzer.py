import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import Counter
import xml.etree.ElementTree as ET
import numpy as np
from urllib.parse import quote
from dotenv import load_dotenv
load_dotenv()

# Keep Hugging Face fully offline in this script so cached models can be used
# without triggering background network requests or safetensors conversion jobs.
LOCAL_HF_CACHE = os.path.join(
    os.environ.get("LOCALAPPDATA", os.getcwd()),
    "hf_cache_agent6",
)
os.makedirs(LOCAL_HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", LOCAL_HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", LOCAL_HF_CACHE)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# NLP and ML libraries
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Data handling
import pandas as pd

# LangChain and OpenAI for advanced sentiment analysis
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"


# ─────────────────────────────────────────────────────────────────────────────
# LANGCHAIN TOOLS FOR SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
@tool
def analyze_sentiment_batch(reviews_text: str, merchant_name: str) -> str:
    """
    Analyze sentiment of multiple customer reviews using GPT-4o-mini.
    Input: JSON string containing list of review texts
    Returns: JSON with sentiment scores, themes, and classification
    """
    try:
        reviews = json.loads(reviews_text) if isinstance(reviews_text, str) else reviews_text
    except json.JSONDecodeError:
        reviews = [reviews_text]

    analysis_prompt = f"""
Filter out noise and non-review content, then
analyze the sentiment of these customer reviews only for the merchant '{merchant_name}'.
For each review, determine:
1. Sentiment label: POSITIVE, NEGATIVE, or NEUTRAL
2. Confidence score (0.0-1.0)
3. Key themes mentioned
4. Severity of issues (if negative)

Reviews:
{json.dumps(reviews, indent=2)}

Return ONLY valid JSON with this structure:
{{
  "overall_sentiment": "POSITIVE|NEGATIVE|NEUTRAL",
  "average_confidence": 0.85,
  "sentiment_distribution": {{"positive": 0.6, "negative": 0.2, "neutral": 0.2}},
  "reviews_analyzed": [
    {{"text": "...", "sentiment": "POSITIVE", "confidence": 0.95, "themes": ["theme1", "theme2"], "issues": []}}
  ],
  "key_positive_themes": ["delivery", "customer service"],
  "key_negative_themes": ["slow processing"],
  "recommendation": "APPROVE|REVIEW|REJECT",
  "summary": "brief assessment"
}}
"""

    try:
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)
        response = llm.invoke(analysis_prompt)
        content = response.content

        # Extract JSON from response
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return content[start:end]
        except Exception:
            pass

        return json.dumps({"error": "Failed to parse LLM response", "raw": content})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def extract_review_themes(reviews_json: str) -> str:
    """
    Extract and cluster common themes from customer reviews.
    Input: JSON array of review texts
    Returns: JSON with theme clusters and sentiment associations
    """
    try:
        reviews = json.loads(reviews_json)
    except json.JSONDecodeError:
        reviews = [reviews_json]

    theme_extraction_prompt = f"""
Analyze these customer reviews and extract the most common themes/topics discussed.
Group related themes together and score their sentiment association.

Reviews:
{json.dumps(reviews, indent=2)}

Return ONLY valid JSON:
{{
  "themes": [
    {{"name": "delivery speed", "mentions": 12, "positive_mentions": 10, "negative_mentions": 2, "neutral_mentions": 0}},
    {{"name": "customer support", "mentions": 8, "positive_mentions": 3, "negative_mentions": 5, "neutral_mentions": 0}}
  ],
  "top_positive_themes": ["delivery speed", "product quality"],
  "top_negative_themes": ["customer support", "pricing"],
  "theme_sentiment_map": {{"delivery speed": 0.83, "customer support": -0.38}}
}}
"""

    try:
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)
        response = llm.invoke(theme_extraction_prompt)
        content = response.content

        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return content[start:end]
        except Exception:
            pass

        return json.dumps({"error": "Failed to parse LLM response", "raw": content})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def compile_sentiment_assessment(
    merchant_name: str,
    reviews_count: int,
    overall_sentiment: str,
    sentiment_distribution_json: str,
    positive_themes_json: str,
    negative_themes_json: str,
    satisfaction_rating: float,
    recommendation: str
) -> str:
    """
    Compile final sentiment assessment output for the merchant.
    Returns structured JSON ready for downstream processing.
    """
    try:
        sentiment_dist = json.loads(sentiment_distribution_json) if isinstance(sentiment_distribution_json, str) else sentiment_distribution_json
    except json.JSONDecodeError:
        sentiment_dist = {}

    try:
        pos_themes = json.loads(positive_themes_json) if isinstance(positive_themes_json, str) else positive_themes_json
    except json.JSONDecodeError:
        pos_themes = []

    try:
        neg_themes = json.loads(negative_themes_json) if isinstance(negative_themes_json, str) else negative_themes_json
    except json.JSONDecodeError:
        neg_themes = []

    assessment = {
        "assessment_metadata": {
            "agent": "Agent 6 - Customer Sentiment Analyzer",
            "merchant_name": merchant_name,
            "assessment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "assessment_version": "1.0"
        },
        "sentiment_analysis": {
            "total_reviews_analyzed": reviews_count,
            "overall_sentiment": overall_sentiment,
            "sentiment_distribution": sentiment_dist,
            "customer_satisfaction_rating": satisfaction_rating
        },
        "themes_analysis": {
            "key_positive_themes": pos_themes,
            "key_negative_themes": neg_themes
        },
        "recommendation": {
            "flag": recommendation,
            "rationale": f"Based on {reviews_count} reviews with {overall_sentiment} sentiment and {satisfaction_rating:.1f}/5.0 satisfaction rating"
        }
    }

    return json.dumps(assessment, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT SETUP FOR LANGCHAIN
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Agent 6 – the Customer Sentiment Analyzer in a merchant 
onboarding pipeline. You analyze customer reviews to assess merchant reputation and 
customer satisfaction.

Your job is to:
1. Analyze sentiment distribution across collected reviews
2. Extract key themes and topics from customer feedback
3. Identify patterns in positive and negative reviews
4. Compile a final sentiment assessment with a recommendation

WORKFLOW:
Step 1 → analyze_sentiment_batch (get overall sentiment distribution)
Step 2 → extract_review_themes (identify key themes)
Step 3 → compile_sentiment_assessment (create final report)

Be analytical, objective, and data-driven. Consider both volume and intensity 
of feedback. Always provide clear recommendation: APPROVE, REVIEW, or REJECT."""

_tools = [
    analyze_sentiment_batch,
    extract_review_themes,
    compile_sentiment_assessment,
]

_manager_llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

_sentiment_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

_sentiment_agent = create_openai_functions_agent(
    llm=_manager_llm, 
    tools=_tools, 
    prompt=_sentiment_prompt
)

_sentiment_executor = AgentExecutor(
    agent=_sentiment_agent,
    tools=_tools,
    verbose=True,
    max_iterations=10,
    return_intermediate_steps=True
)




# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ReviewData:
    """Data class for individual review"""
    source: str
    text: str
    rating: Optional[float]
    timestamp: str
    author: str


@dataclass
class SentimentOutput:
    """Output structure for Agent 6"""
    merchant_name: str
    overall_sentiment_score: float  # -1.0 to 1.0
    customer_satisfaction_rating: float  # 0-5
    sentiment_distribution: Dict[str, float]  # positive, neutral, negative
    key_positive_themes: List[Tuple[str, float]]
    key_negative_themes: List[Tuple[str, float]]
    review_count: int
    sample_positive_reviews: List[str]
    sample_negative_reviews: List[str]
    recommendation_flag: str  # "APPROVE", "REVIEW", "REJECT"
    processing_timestamp: str


class Agent6CustomerSentimentAnalyzer:
    """
    Customer Sentiment & Review Analysis Agent for Merchant Onboarding
    Analyzes customer reviews and feedback to assess merchant reputation
    """

    def __init__(self, merchant_name: str, merchant_url: str, industry: str):
        """
        Initialize Agent 6
        
        Args:
            merchant_name: Name of the merchant
            merchant_url: Website URL of the merchant
            industry: Industry classification
        """
        self.merchant_name = merchant_name
        self.merchant_url = merchant_url
        self.industry = industry

        self.sentiment_analyzer = self._initialize_sentiment_analyzer()
        self.http_session = requests.Session()
        self.http_session.trust_env = False
        self.http_session.headers.update({
            "User-Agent": "Agent6CustomerSentimentAnalyzer/1.0"
        })
        
        self.reviews: List[ReviewData] = []
        self.langchain_executor = _sentiment_executor  # Reference to module-level executor

    def _initialize_sentiment_analyzer(self):
        """Prefer cached models, then try download, then fall back to TextBlob."""
        model_candidates = [
            ("ProsusAI/finbert", "FinBERT"),
            ("distilbert-base-uncased-finetuned-sst-2-english", "default sentiment model"),
        ]

        for model_name, display_name in model_candidates:
            cached_pipeline = self._build_sentiment_pipeline(
                model_name=model_name,
                local_files_only=True,
            )
            if cached_pipeline is not None:
                print(f"Loaded {display_name} from local cache.")
                return cached_pipeline

            downloadable_pipeline = self._build_sentiment_pipeline(
                model_name=model_name,
                local_files_only=False,
            )
            if downloadable_pipeline is not None:
                print(f"Downloaded and loaded {display_name}.")
                return downloadable_pipeline

        print("No transformer sentiment model available, using TextBlob fallback.")
        return None

    def _build_sentiment_pipeline(self, model_name: str, local_files_only: bool):
        """Build a sentiment pipeline from cache or by downloading if allowed."""
        source = "local cache" if local_files_only else "remote download"

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                use_safetensors=False,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                use_safetensors=False,
            )
            return pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                truncation=True,
            )
        except Exception as error:
            print(f"Could not load {model_name} from {source}: {error}")
            return None

    def _score_review_sentiment(self, review_text: str) -> Tuple[str, float]:
        """Return a normalized sentiment label and confidence-like score."""
        polarity = TextBlob(review_text).sentiment.polarity
        normalized_text = review_text.lower()
        negative_cues = {
            "bad", "problems", "problem", "issues", "issue", "slow", "fraud",
            "scam", "complaint", "complaints", "chargeback", "dispute",
            "higher-than-normal", "step ahead", "risk", "locked", "hold"
        }
        positive_cues = {
            "great", "reliable", "helpful", "recommended", "recommend",
            "good experience", "strong support", "fast", "excellent"
        }

        if self.sentiment_analyzer is not None:
            result = self.sentiment_analyzer(review_text[:512])[0]
            label = str(result["label"]).upper()
            score = float(result["score"])

            # Headline-style content often comes back as neutral from the transformer
            # even when the wording is clearly opinionated. Use TextBlob polarity as a
            # lightweight tie-breaker in that case.
            if label == "NEUTRAL":
                if polarity > 0.15:
                    return "POSITIVE", min(1.0, abs(polarity))
                if polarity < -0.15:
                    return "NEGATIVE", min(1.0, abs(polarity))
                if any(cue in normalized_text for cue in negative_cues):
                    return "NEGATIVE", 0.4
                if any(cue in normalized_text for cue in positive_cues):
                    return "POSITIVE", 0.4

            return label, score

        if polarity > 0.1:
            return "POSITIVE", min(1.0, abs(polarity))
        if polarity < -0.1:
            return "NEGATIVE", min(1.0, abs(polarity))
        return "NEUTRAL", 0.0

    def _extract_top_themes_from_texts(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Extract top TF-IDF themes from a list of review texts."""
        if not texts:
            return []

        vectorizer = TfidfVectorizer(
            max_features=20,
            stop_words="english",
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

        top_indices = np.argsort(scores)[::-1]
        filtered_themes = []
        low_value_terms = {
            "good", "great", "nice", "bad", "poor", "product", "products",
            "service", "customer", "company", "merchant", "experience", "overall",
            "high", "slow", "quality", "delivery", "prices", "support", "respond"
        }
        low_value_phrases = {
            "good experience", "experience overall", "overall customer",
            "product quality", "customer service", "strong customer"
        }

        for i in top_indices:
            score = float(scores[i])
            if score <= 0:
                continue

            theme = feature_names[i].strip()
            theme_parts = theme.split()

            if any(part.isdigit() for part in theme_parts):
                continue

            if len(theme_parts) == 1 and len(theme) < 5:
                continue

            if len(theme_parts) == 1 and theme in low_value_terms:
                continue

            if len(theme_parts) > 1 and theme in low_value_phrases:
                continue

            filtered_themes.append((theme, score))

            if len(filtered_themes) == 5:
                break

        return filtered_themes

    def _fetch_json(
        self,
        url: str,
        params: Optional[Dict[str, str]] = None,
        timeout: int = 15
    ) -> Optional[Dict]:
        """Fetch JSON from a public API endpoint and fail gracefully."""
        try:
            response = self.http_session.get(url, params=params, timeout=timeout)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                print(
                    f"Rate limited by {url}"
                    + (f"; retry after {retry_after} seconds" if retry_after else "")
                )
                return None
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API request failed for {url}: {e}")
            return None

    def _is_review_relevant(self, text: str) -> bool:
        """Filter for review-like text that is actually about the merchant."""
        normalized_text = str(text).lower()
        merchant_tokens = {token for token in self.merchant_name.lower().split() if len(token) > 2}
        sentiment_terms = {
            "review", "reviews", "scam", "fraud", "complaint", "complaints",
            "support", "customer", "service", "refund", "issue", "issues",
            "recommend", "recommended", "experience", "delivery", "shipping",
            "chargeback", "reliable", "terrible", "awful", "great", "slow",
            "helpful", "problem", "problems", "dispute", "buyer protection"
        }
        noisy_listing_terms = {
            "wts", "wtb", "spot", "paypal ff", "friends and family",
            "g&s", "shipped conus", "libertad", "geiger", "maples",
            "online casino", "casino bonus", "anbieter", "vergleich"
        }
        payment_method_phrases = {
            "have paypal", "take paypal", "accept paypal", "via paypal",
            "using paypal", "use paypal", "paid with paypal", "pay with paypal"
        }

        mentions_merchant = self.merchant_name.lower() in normalized_text
        if not mentions_merchant and len(merchant_tokens) > 1:
            mentions_merchant = any(token in normalized_text for token in merchant_tokens)

        has_review_signal = any(term in normalized_text for term in sentiment_terms)
        has_noise_signal = any(term in normalized_text for term in noisy_listing_terms)
        mentions_as_payment_method = any(term in normalized_text for term in payment_method_phrases)
        return (
            mentions_merchant
            and has_review_signal
            and not has_noise_signal
            and not mentions_as_payment_method
        )

    def _create_review(
        self,
        source: str,
        text: str,
        rating: Optional[float] = None,
        timestamp: Optional[str] = None,
        author: str = "Unknown"
    ) -> Optional[ReviewData]:
        """Create a ReviewData record if the text is meaningful."""
        cleaned_text = " ".join(str(text).split())
        if len(cleaned_text) < 2:
            return None
        if not self._is_review_relevant(cleaned_text):
            return None

        return ReviewData(
            source=source,
            text=cleaned_text,
            rating=rating,
            timestamp=timestamp or datetime.now().isoformat(),
            author=author
        )

    def _get_industry_query_terms(self) -> List[str]:
        """Return a small set of industry-specific query terms."""
        industry_map = {
            "fintech": ["payments", "merchant services", "checkout"],
            "ecommerce": ["shopping", "fulfillment", "returns"],
            "saas": ["billing", "support", "integration"],
            "healthcare": ["patients", "appointments", "insurance"],
            "education": ["students", "courses", "learning platform"],
            "travel": ["booking", "refunds", "customer support"],
        }
        return industry_map.get(self.industry.lower(), [self.industry, "customer experience"])
        
    def search_reviews(self) -> List[ReviewData]:
        """
        Search for customer reviews across multiple platforms
        
        Returns:
            List of ReviewData objects
        """
        print(f"Searching reviews for {self.merchant_name}...")
        
        # Search from multiple sources
        reviews = []
        
        # 1. E-commerce platforms (mock implementation)
        reviews.extend(self._search_ecommerce_platforms())
                 
        # 2. Social media (mock implementation)
        reviews.extend(self._search_social_media())
        
        
        # 3. Review aggregators (mock implementation)
        reviews.extend(self._search_review_aggregators())
        
        # 4. Industry-specific sites (mock implementation)
        reviews.extend(self._search_industry_sites())
        
        
        self.reviews = reviews
        print(f"Found {len(reviews)} reviews")
        return reviews
    
    def _search_ecommerce_platforms(self) -> List[ReviewData]:
        """Search Google News RSS for merchant review and support mentions."""
        query = quote(
            f'"{self.merchant_name}" review OR "{self.merchant_name}" "customer service" '
            f'OR "{self.merchant_name}" complaint OR "{self.merchant_name}" refund'
        )
        query1 = quote(
            f'"{self.merchant_name}"')
        feed_url = f"https://news.google.com/rss/search?q={query1}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            response = self.http_session.get(feed_url, timeout=15)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            #print(f"Google News API request successful for {feed_url}")
        except Exception as e:
            print(f"API request failed for {feed_url}: {e}")
            return []

        reviews = []
        for item in root.findall(".//item")[:5]:
            review = self._create_review(
                source="Google News",
                text=" ".join(
                    part for part in [
                        item.findtext("title", default=""),
                        item.findtext("description", default=""),
                    ] if part
                ),
                timestamp=item.findtext("pubDate", default=None),
                author=item.findtext("source", default="Google News")
            )
            if review:
                reviews.append(review)
        print(f"No. of ecommerce reviews {len(reviews)}") 
        return reviews
    
    def _search_social_media(self) -> List[ReviewData]:
        """Search Reddit posts mentioning the merchant via Reddit's public JSON API."""
        data = self._fetch_json(
            "https://www.reddit.com/search.json",
            params={
                "q": f'"{self.merchant_name}"',
                "restrict_sr": "false",
                "sort": "new",
                #"limit": "5",
                "t": "year",
            }
        )

        reviews = []
        posts = ((data or {}).get("data") or {}).get("children", [])
        for post in posts:
            post_data = post.get("data", {})
            content_parts = [
                post_data.get("title", ""),
                post_data.get("selftext", "")
            ]
            review = self._create_review(
                source="Reddit",
                text=" ".join(part for part in content_parts if part),
                timestamp=datetime.fromtimestamp(
                    post_data.get("created_utc", datetime.now().timestamp())
                ).isoformat(),
                author=post_data.get("author", "RedditUser")
            )
            if review:
                reviews.append(review)
        print(f"No. of Social Media reviews {len(reviews)}")        
        return reviews
    
    def _search_review_aggregators(self) -> List[ReviewData]:
        """Search public discussion results through the Hacker News Algolia API."""
        data = self._fetch_json(
            "https://hn.algolia.com/api/v1/search_by_date",
            params={
                "query": f"{self.merchant_name}",
                #"tags": "(story,comment)",
                #"hitsPerPage": "5",
            }
        )

        reviews = []
        for hit in (data or {}).get("hits", []):
            review = self._create_review(
                source="Hacker News",
                text=" ".join(
                    part for part in [
                        hit.get("title", ""),
                        hit.get("story_text", ""),
                        hit.get("comment_text", ""),
                    ] if part
                ),
                timestamp=hit.get("created_at"),
                author=hit.get("author", "HNUser")
            )
            if review:
                reviews.append(review)
        print(f"No. of Review Aggregator reviews {len(reviews)}")        
        return reviews
    
    def _search_industry_sites(self) -> List[ReviewData]:
        """Search industry/community discussions using the Stack Exchange API."""
        industry_terms = " ".join(self._get_industry_query_terms())
        data = self._fetch_json(
            "https://api.stackexchange.com/2.3/search/advanced",
            params={
                "order": "desc",
                "sort": "relevance",
                "q": f"{self.merchant_name}",
                "site": "money",
                "pagesize": "5",
                "filter": "default",
            }
        )

        reviews = []
        for item in (data or {}).get("items", []):
            tags = ", ".join(item.get("tags", []))
            review = self._create_review(
                source="Stack Exchange",
                text=f"{item.get('title', '')}. Tags: {tags}",
                timestamp=datetime.fromtimestamp(
                    item.get("creation_date", datetime.now().timestamp())
                ).isoformat(),
                author=item.get("owner", {}).get("display_name", "StackUser")
            )
            if review:
                reviews.append(review)
        print(f"No. of Industry-specific reviews {len(reviews)}")        
        return reviews
    
    def analyze_sentiment(self) -> Tuple[Dict, List[str], List[str]]:
        """
        Analyze sentiment of all collected reviews
        
        Returns:
            Tuple of (sentiment_distribution, positive_reviews, negative_reviews)
        """
        if not self.reviews:
            print("No reviews to analyze")
            return {}, [], []
        
        sentiments = []
        positive_reviews = []
        negative_reviews = []
        
        for review in self.reviews:
            # Analyze sentiment
            try:
                label, score = self._score_review_sentiment(review.text)
                
                # Normalize sentiment
                if label == "POSITIVE":
                    sentiments.append(score)
                    positive_reviews.append(review.text)
                elif label == "NEGATIVE":
                    sentiments.append(-score)
                    negative_reviews.append(review.text)
                else:
                    sentiments.append(0)
            except Exception as e:
                print(f"Error analyzing sentiment: {e}")
                continue
        
        # Calculate sentiment distribution
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        sentiment_dist = {
            "positive": positive_count / len(sentiments) if sentiments else 0,
            "negative": negative_count / len(sentiments) if sentiments else 0,
            "neutral": neutral_count / len(sentiments) if sentiments else 0
        }
        
        return sentiment_dist, positive_reviews, negative_reviews
    
    def extract_themes(self) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Extract common themes from reviews
        
        Returns:
            Tuple of (positive_themes, negative_themes)
        """
        if not self.reviews:
            return [], []

        try:
            positive_texts = []
            negative_texts = []

            for review in self.reviews:
                label, _ = self._score_review_sentiment(review.text)
                if label == "POSITIVE":
                    positive_texts.append(review.text)
                elif label == "NEGATIVE":
                    negative_texts.append(review.text)

            positive_themes = self._extract_top_themes_from_texts(positive_texts)
            negative_themes = self._extract_top_themes_from_texts(negative_texts)
        except Exception as e:
            print(f"Error extracting themes: {e}")
            positive_themes = []
            negative_themes = []
        
        return positive_themes, negative_themes
    
    def calculate_satisfaction_rating(self, sentiment_dist: Dict) -> float:
        """
        Calculate overall customer satisfaction rating (0-5)
        
        Args:
            sentiment_dist: Sentiment distribution dictionary
            
        Returns:
            Satisfaction rating 0-5
        """
        # Weight sentiment distribution to 5-star scale
        rating = (
            sentiment_dist.get("positive", 0) * 5 +
            sentiment_dist.get("neutral", 0) * 2.5 +
            sentiment_dist.get("negative", 0) * 0.5
        )
        return min(5.0, max(0.0, rating))
    
    def generate_recommendation_flag(
        self,
        sentiment_dist: Dict,
        satisfaction_rating: float
    ) -> str:
        """
        Generate recommendation flag based on sentiment analysis
        
        Args:
            sentiment_dist: Sentiment distribution
            satisfaction_rating: Overall satisfaction rating
            
        Returns:
            Recommendation flag ("APPROVE", "REVIEW", "REJECT")
        """
        negative_ratio = sentiment_dist.get("negative", 0)
        
        if negative_ratio > 0.4 or satisfaction_rating < 2.0:
            return "REJECT"
        elif negative_ratio > 0.2 or satisfaction_rating < 3.0:
            return "REVIEW"
        else:
            return "APPROVE"

    # ─────────────────────────────────────────────────────────────────────────────
    # LANGCHAIN SENTIMENT ANALYSIS METHODS
    # ─────────────────────────────────────────────────────────────────────────────
    def run_langchain_sentiment_analysis(self) -> dict:
        """
        Run LangChain-based sentiment analysis on collected customer reviews.
        Uses the instance's reviews and merchant_name.
        
        Returns:
            Dictionary with formatted sentiment analysis results matching SentimentOutput structure:
            - overall_sentiment_score
            - customer_satisfaction_rating
            - sentiment_distribution
            - key_positive_themes
            - key_negative_themes
            - sample_positive_reviews (top 3)
            - sample_negative_reviews (top 3)
            - recommendation_flag
            - summary
        """
        if not self.reviews:
            return {
                "merchant_name": self.merchant_name,
                "overall_sentiment_score": 0.0,
                "customer_satisfaction_rating": 2.5,
                "sentiment_distribution": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                "key_positive_themes": [],
                "key_negative_themes": [],
                "review_count": 0,
                "sample_positive_reviews": [],
                "sample_negative_reviews": [],
                "recommendation_flag": "REVIEW",
                "summary": "No reviews available for analysis.",
                "assessment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        # Prepare review texts for analysis
        review_texts = [review.text for review in self.reviews]

        prompt_text = f"""
Please perform a complete Sentiment Analysis for the merchant '{self.merchant_name}'.
Use ALL available tools in the correct sequence.

CUSTOMER REVIEWS ({len(review_texts)} total):
{json.dumps(review_texts, indent=2)}

Analyze this merchant's customer sentiment fully and return the complete structured assessment.
Include sentiment distribution, key themes, and a clear recommendation.
"""

        try:
            result = self.langchain_executor.invoke({"input": prompt_text})
            
            # Extract sentiment analysis batch results for sample reviews
            sentiment_analysis = self._extract_sentiment_analysis_from_steps(
                result.get("intermediate_steps", [])
            )
            
            # Extract structured assessment from intermediate steps
            structured_assessment = self._extract_sentiment_assessment_from_steps(
                result.get("intermediate_steps", [])
            )

            # Flatten for output in requested format
            return self._flatten_sentiment_for_output(
                structured_assessment,
                len(review_texts),
                result.get("output", ""),
                sentiment_analysis=sentiment_analysis
            )
        except Exception as e:
            print(f"Error in LangChain sentiment analysis: {e}")
            return {
                "merchant_name": self.merchant_name,
                "overall_sentiment_score": 0.0,
                "customer_satisfaction_rating": 2.5,
                "sentiment_distribution": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                "key_positive_themes": [],
                "key_negative_themes": [],
                "review_count": len(self.reviews),
                "sample_positive_reviews": [],
                "sample_negative_reviews": [],
                "recommendation_flag": "REVIEW",
                "summary": f"Error during analysis: {str(e)}",
                "assessment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def _extract_sentiment_analysis_from_steps(self, intermediate_steps: list) -> dict:
        """
        Extract the analyze_sentiment_batch results for sample reviews.
        """
        for action, observation in intermediate_steps:
            tool_name = getattr(action, "tool", "")
            if tool_name == "analyze_sentiment_batch":
                try:
                    return json.loads(observation)
                except (json.JSONDecodeError, TypeError):
                    continue
        return {}

    def _extract_sentiment_assessment_from_steps(self, intermediate_steps: list) -> dict:
        """
        Pull the JSON result of compile_sentiment_assessment from agent intermediate steps.
        """
        for action, observation in intermediate_steps:
            tool_name = getattr(action, "tool", "")
            if tool_name == "compile_sentiment_assessment":
                try:
                    return json.loads(observation)
                except (json.JSONDecodeError, TypeError):
                    continue
        return {}

    def _flatten_sentiment_for_output(self, assessment: dict, reviews_count: int, 
                                      raw_output: str = "", sentiment_analysis: dict = None) -> dict:
        """
        Convert nested assessment structure into SentimentOutput-compatible format.
        """
        if not assessment:
            return {
                "merchant_name": self.merchant_name,
                "overall_sentiment_score": 0.0,
                "customer_satisfaction_rating": 2.5,
                "sentiment_distribution": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                "key_positive_themes": [],
                "key_negative_themes": [],
                "review_count": reviews_count,
                "sample_positive_reviews": [],
                "sample_negative_reviews": [],
                "recommendation_flag": "REVIEW",
                "summary": "Unable to generate sentiment analysis.",
                "assessment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "raw_output": raw_output
            }

        sentiment_section = assessment.get("sentiment_analysis", {})
        themes_section = assessment.get("themes_analysis", {})
        recommendation_section = assessment.get("recommendation", {})
        
        # Calculate overall_sentiment_score from distribution
        dist = sentiment_section.get("sentiment_distribution", {})
        overall_sentiment_score = (
            dist.get("positive", 0) - dist.get("negative", 0)
        )
        
        # Extract sample reviews from sentiment analysis results
        positive_samples = []
        negative_samples = []
        if sentiment_analysis:
            reviews_analyzed = sentiment_analysis.get("reviews_analyzed", [])
            for review_item in reviews_analyzed:
                if review_item.get("sentiment") == "POSITIVE":
                    positive_samples.append(review_item.get("text", ""))
                elif review_item.get("sentiment") == "NEGATIVE":
                    negative_samples.append(review_item.get("text", ""))
            
            # Trim to top 3
            positive_samples = positive_samples[:3]
            negative_samples = negative_samples[:3]
        
        # Generate brief assessment summary
        satisfaction = sentiment_section.get("customer_satisfaction_rating", 2.5)
        sentiment = sentiment_section.get("overall_sentiment", "NEUTRAL")
        pos_count = dist.get("positive", 0)
        neg_count = dist.get("negative", 0)
        
        if recommendation_section.get("flag") == "APPROVE":
            summary = f"{sentiment} sentiment ({pos_count:.0%} positive, {neg_count:.0%} negative). Satisfaction: {satisfaction:.1f}/5. No significant issues detected."
        elif recommendation_section.get("flag") == "REJECT":
            summary = f"{sentiment} sentiment ({pos_count:.0%} positive, {neg_count:.0%} negative). Satisfaction: {satisfaction:.1f}/5. Critical issues identified - manual review required."
        else:
            summary = f"{sentiment} sentiment ({pos_count:.0%} positive, {neg_count:.0%} negative). Satisfaction: {satisfaction:.1f}/5. Moderate concerns warrant further review."

        return {
            "merchant_name": assessment.get("assessment_metadata", {}).get("merchant_name", self.merchant_name),
            "overall_sentiment_score": overall_sentiment_score,
            "customer_satisfaction_rating": sentiment_section.get("customer_satisfaction_rating", 2.5),
            "sentiment_distribution": sentiment_section.get("sentiment_distribution", {}),
            "key_positive_themes": themes_section.get("key_positive_themes", []),
            "key_negative_themes": themes_section.get("key_negative_themes", []),
            "review_count": sentiment_section.get("total_reviews_analyzed", reviews_count),
            "sample_positive_reviews": positive_samples,
            "sample_negative_reviews": negative_samples,
            "recommendation_flag": recommendation_section.get("flag", "REVIEW"),
            "summary": summary,
            "assessment_date": assessment.get("assessment_metadata", {}).get("assessment_date", ""),
            "raw_output": raw_output
        }
    
    def process(self) -> SentimentOutput:
        """
        Main processing method for Agent 6
        
        Returns:
            SentimentOutput with analysis results
        """
        print(f"Agent 6: Starting sentiment analysis for {self.merchant_name}")
        
        # Step 1: Search for reviews
        self.search_reviews()
        
        # Check if OpenAI API key is available
        if os.getenv("OPENAI_API_KEY"):
            # Step 2: Run LangChain sentiment analysis
            langchain_result = self.run_langchain_sentiment_analysis()
            
            # Step 3: Create SentimentOutput from LangChain results
            output = SentimentOutput(
                merchant_name=langchain_result.get("merchant_name", self.merchant_name),
                overall_sentiment_score=langchain_result.get("overall_sentiment_score", 0.0),
                customer_satisfaction_rating=langchain_result.get("customer_satisfaction_rating", 2.5),
                sentiment_distribution=langchain_result.get("sentiment_distribution", {}),
                key_positive_themes=[(theme, 0.5) for theme in langchain_result.get("key_positive_themes", [])],
                key_negative_themes=[(theme, 0.5) for theme in langchain_result.get("key_negative_themes", [])],
                review_count=langchain_result.get("review_count", len(self.reviews)),
                sample_positive_reviews=langchain_result.get("sample_positive_reviews", []),
                sample_negative_reviews=langchain_result.get("sample_negative_reviews", []),
                recommendation_flag=langchain_result.get("recommendation_flag", "REVIEW"),
                processing_timestamp=datetime.now().isoformat()
            )
        else:
            # Fallback to legacy sentiment analysis methods
            print("OpenAI API key not found. Using legacy sentiment analysis.")
            sentiment_dist, positive_reviews, negative_reviews = self.analyze_sentiment()
            positive_themes, negative_themes = self.extract_themes()
            satisfaction_rating = self.calculate_satisfaction_rating(sentiment_dist)
            recommendation_flag = self.generate_recommendation_flag(sentiment_dist, satisfaction_rating)
            
            # Calculate overall sentiment score
            overall_sentiment_score = sentiment_dist.get("positive", 0) - sentiment_dist.get("negative", 0)
            
            output = SentimentOutput(
                merchant_name=self.merchant_name,
                overall_sentiment_score=overall_sentiment_score,
                customer_satisfaction_rating=satisfaction_rating,
                sentiment_distribution=sentiment_dist,
                key_positive_themes=positive_themes,
                key_negative_themes=negative_themes,
                review_count=len(self.reviews),
                sample_positive_reviews=positive_reviews[:3],
                sample_negative_reviews=negative_reviews[:3],
                recommendation_flag=recommendation_flag,
                processing_timestamp=datetime.now().isoformat()
            )
        
        print(f"Agent 6: Analysis complete. Recommendation: {output.recommendation_flag}")
        return output


# Example usage
if __name__ == "__main__":
    # Initialize Agent 6
    agent = Agent6CustomerSentimentAnalyzer(
        merchant_name="Paypal",
        merchant_url="https://paypal.com",
        industry="Fintech"
    )
    
    # Process merchant
    result = agent.process()
    
    # Display results
    print("\n" + "="*80)
    print("AGENT 6 - CUSTOMER SENTIMENT & REVIEW ANALYSIS OUTPUT")
    print("="*80)
    print(f"Merchant: {result.merchant_name}")
    print(f"Overall Sentiment Score: {result.overall_sentiment_score:.2f}")
    print(f"Customer Satisfaction Rating: {result.customer_satisfaction_rating:.2f}/5.0")
    print(f"Review Count: {result.review_count}")
    print(f"Sentiment Distribution: {result.sentiment_distribution}")
    print(f"Recommendation Flag: {result.recommendation_flag}")
    print(f"Processing Time: {result.processing_timestamp}")
    print("\nKey Positive Themes:")
    for theme, score in result.key_positive_themes:
        print(f"  - {theme}: {score:.3f}")
    print("\nKey Negative Themes:")
    for theme, score in result.key_negative_themes:
        print(f"  - {theme}: {score:.3f}")
    print("\nSample Positive Reviews:")
    for i, review in enumerate(result.sample_positive_reviews, 1):
        print(f"  {i}. {review[:1000]}...")
    print("\nSample Negative Reviews:")
    for i, review in enumerate(result.sample_negative_reviews, 1):
        print(f"  {i}. {review[:1000]}...")
    print("="*80)
