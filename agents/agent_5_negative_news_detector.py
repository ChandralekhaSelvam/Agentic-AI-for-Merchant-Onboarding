import os
import json
import re
import hashlib
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from urllib.parse import quote, quote_plus
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

import requests
from openai import OpenAI

@dataclass
class MerchantProfile:
    """Merchant identity provided by Agent 2 (validated profile)."""
    name: str
    url: str = ""
    industry: str = ""

@dataclass
class NewsArticle:
    """A single article/post/complaint retrieved during screening."""
    source: str
    title: str
    url: str = ""
    snippet: str = ""
    published_date: str = ""
    author: str = ""

@dataclass
class RiskFinding:
    """A classified adverse media finding with evidence and severity."""
    title: str
    category: str          # FINANCIAL | LEGAL | REGULATORY | REPUTATIONAL | OPERATIONAL | SANCTIONS
    severity: str          # CRITICAL | HIGH | MEDIUM | LOW
    summary: str
    source_url: str = ""
    source_name: str = ""
    published_date: str = ""
    confidence: float = 0.5
    evidence_snippet: str = ""
    requires_manual_review: bool = False

@dataclass
class NegativeNewsOutput:
    """Final output structure for Agent 5 — feeds into onboarding decision."""
    merchant_name: str
    screening_id: str
    timestamp: str

    # Risk verdict
    overall_risk_severity: str           # CRITICAL | HIGH | MEDIUM | LOW | NONE
    overall_risk_score: float            # 0–100
    recommended_action: str              # APPROVE | EDD | MANUAL_REVIEW | DECLINE
    data_sufficiency: str                # SUFFICIENT | LIMITED | INSUFFICIENT

    # Findings detail
    findings: List[Dict]
    findings_by_category: Dict[str, int]
    findings_by_severity: Dict[str, int]

    # Source intelligence
    total_articles_collected: int
    articles_by_source: Dict[str, int]
    sources_consulted: List[str]

    # Cross-source conclusion
    corroboration_score: float           # 0–1, multi-source agreement
    cross_source_summary: str
    key_risk_indicators: List[str]

    # Executive output
    executive_summary: str
    recommendation_rationale: str

    # Audit trail
    search_queries_used: List[str]
    reasoning_trace: List[str]
    processing_time_seconds: float
    model_used: str

ADVERSE_SIGNALS = frozenset({
    "fraud", "scam", "lawsuit", "sued", "penalty", "fine", "violation",
    "breach", "bankruptcy", "criminal", "indictment", "investigation",
    "sanction", "money laundering", "embezzlement", "class action",
    "regulatory", "enforcement", "complaint", "settlement", "hack",
    "arrested", "charged", "convicted", "default", "liquidation",
    "scandal", "controversy", "warning", "revoked", "suspended",
    "cease and desist", "ponzi", "misleading", "defraud", "probe",
    "allegation", "allegations", "unethical", "illegal", "banned",
    "shut down", "raided", "seizure", "confiscated", "blacklisted",
})

INDUSTRY_EXTRA_QUERIES: Dict[str, List[str]] = {
    "fintech":    ['{name} compliance failure', '{name} payment fraud'],
    "payments":   ['{name} chargeback fraud', '{name} processor ban'],
    "crypto":     ['{name} rug pull exit scam', '{name} unregistered securities'],
    "gambling":   ['{name} illegal gambling', '{name} gaming license revoked',
                   '{name} betting fraud', '{name} gambling ban'],
    "pharma":     ['{name} FDA warning letter', '{name} counterfeit drug'],
    "ecommerce":  ['{name} fake products', '{name} consumer protection'],
    "banking":    ['{name} FDIC enforcement', '{name} bank fraud'],
    "insurance":  ['{name} denied claims scandal', '{name} insurance fraud'],
    "lending":    ['{name} predatory lending', '{name} usury violation'],
}

# Regex to strip legal entity suffixes from merchant names for search
_LEGAL_SUFFIXES = re.compile(
    r'\b(Inc\.?|LLC\.?|Ltd\.?|Corp\.?|Co\.?|PLC\.?|GmbH|S\.?A\.?|'
    r'N\.?V\.?|AG|SE|Pty\.?|Pvt\.?|Limited|Incorporated|Corporation|'
    r'Company|Holdings?|Group|Enterprises?|International|Online)\s*\.?\s*$',
    re.IGNORECASE,
)

class Agent5NegativeNewsDetector:
    """
    Adverse media screening agent for merchant onboarding risk assessment.

    Search strategy:
        1. Normalize name: "BetKing Online Ltd." → "BetKing"
        2. Run NARROW queries: "BetKing" fraud OR scam (exact match)
        3. If 0 results → BROAD queries: BetKing fraud scam (no quotes)
        4. Google News uses international results (no US-only restriction)
    """

    # Tier 1: Narrow queries (quoted exact match + adverse keywords)
    NARROW_QUERY_TEMPLATES = [
        '"{name}" fraud OR scam',
        '"{name}" lawsuit OR sued OR indicted',
        '"{name}" fine OR penalty OR sanction',
        '"{name}" regulatory action enforcement',
        '"{name}" scandal OR controversy',
        '"{name}" complaint OR "class action"',
        '"{name}" data breach OR hack',
        '"{name}" bankruptcy OR insolvent',
        '"{name}" money laundering',
        '"{name}" OFAC OR "sanctions list"',
    ]

    # Tier 2: Broad queries (no quotes — catches case variations, partial mentions)
    BROAD_QUERY_TEMPLATES = [
        '{name} fraud scam controversy',
        '{name} lawsuit sued legal action',
        '{name} fine penalty regulatory',
        '{name} scandal allegations unethical',
        '{name} complaint investigation',
        '{name} banned revoked suspended',
    ]

    def __init__(self, merchant_name: str, merchant_url: str, industry: str,
                 openai_api_key: str = ""):
        self.merchant = MerchantProfile(
            name=merchant_name,
            url=merchant_url,
            industry=industry,
        )

        # Core brand name stripped of legal suffixes — used for ALL searches
        self._search_name = self._normalize_search_name(merchant_name)

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        self.llm = OpenAI(api_key=api_key, timeout=45)
        self.model = "gpt-4o-mini"

        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers["User-Agent"] = "Agent5-NegativeNews/2.0"

        self.articles: List[NewsArticle] = []
        self.wiki_info: Dict[str, Any] = {}
        self._trace: List[str] = []
        self._queries: List[str] = []
        self._key_risk_indicators: List[str] = []

    @staticmethod
    def _normalize_search_name(name: str) -> str:
        """
        Strip legal suffixes and generic words to get the core brand name.
            'BetKing Online Ltd.'      → 'BetKing'
            'PayPal Holdings Inc.'     → 'PayPal'
            'Blue Bottle Coffee'       → 'Blue Bottle Coffee'
        """
        cleaned = name.strip()
        for _ in range(3):
            prev = cleaned
            cleaned = _LEGAL_SUFFIXES.sub("", cleaned).strip().rstrip(".,")
            if cleaned == prev:
                break
        return cleaned if cleaned else name.strip()

    # ─────────────────────────────────────────────────────────────────
    # NETWORK HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _get(self, url: str, params: Optional[Dict] = None,
             timeout: int = 15) -> Optional[requests.Response]:
        try:
            resp = self.session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                print(f"    rate-limited by {url.split('/')[2]}")
                return None
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            print(f"    request failed ({url.split('/')[2]}): {exc}")
            return None

    def _get_json(self, url: str, params: Optional[Dict] = None,
                  timeout: int = 15) -> Optional[Dict]:
        resp = self._get(url, params, timeout)
        if resp is None:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    def _get_rss(self, url: str, params: Optional[Dict] = None,
                 timeout: int = 15) -> Optional[ET.Element]:
        resp = self._get(url, params, timeout)
        if resp is None:
            return None
        try:
            return ET.fromstring(resp.content)
        except ET.ParseError:
            return None

    # ─────────────────────────────────────────────────────────────────
    # RELEVANCE FILTER
    # ─────────────────────────────────────────────────────────────────

    def _has_adverse_relevance(self, text: str) -> bool:
        """
        Determine if text is both about the target merchant AND contains
        adverse/risk signals. Case-insensitive matching on brand name.
        """
        if not text or len(text.strip()) < 20:
            return False

        lower = text.lower()
        brand_lower = self._search_name.lower()

        # Condition 1: mentions merchant (case-insensitive)
        if brand_lower in lower:
            mentions_merchant = True
        else:
            # For multi-word names, check significant tokens (> 2 chars)
            tokens = [t for t in brand_lower.split() if len(t) > 2]
            if not tokens:
                mentions_merchant = False
            elif len(tokens) == 1:
                # Single-word brand: must appear in text
                mentions_merchant = tokens[0] in lower
            else:
                # Multi-word: at least half of tokens must appear
                mentions_merchant = sum(t in lower for t in tokens) >= max(1, len(tokens) // 2)

        if not mentions_merchant:
            return False

        # Condition 2: has adverse signal
        return any(signal in lower for signal in ADVERSE_SIGNALS)

    def _to_article(self, source: str, title: str, url: str = "",
                    snippet: str = "", published_date: str = "",
                    author: str = "") -> Optional[NewsArticle]:
        """Build a NewsArticle only if the content passes the adverse relevance filter."""
        combined = f"{title} {snippet}".strip()
        if not self._has_adverse_relevance(combined):
            return None
        return NewsArticle(
            source=source, title=title.strip(), url=url,
            snippet=snippet[:500].strip(),
            published_date=published_date, author=author,
        )

    # ─────────────────────────────────────────────────────────────────
    # QUERY BUILDER
    # ─────────────────────────────────────────────────────────────────

    def _build_queries(self) -> List[str]:
        """
        Generate two tiers of search queries:
            Tier 1 (NARROW): Quoted exact-match — high precision, low recall
            Tier 2 (BROAD):  Unquoted — lower precision, high recall (fallback)
        """
        name = self._search_name
        suffix = f" {self.merchant.industry}" if len(name.split()) <= 1 else ""

        queries: List[str] = []

        # Tier 1: Narrow quoted queries
        for tpl in self.NARROW_QUERY_TEMPLATES:
            queries.append(tpl.format(name=name) + suffix)

        # Industry-specific (unquoted — these are already specific enough)
        ind = self.merchant.industry.lower() if self.merchant.industry else ""
        for key, templates in INDUSTRY_EXTRA_QUERIES.items():
            if key in ind:
                queries.extend(t.format(name=name) for t in templates)

        # Tier 2: Broad unquoted queries (used as fallback)
        for tpl in self.BROAD_QUERY_TEMPLATES:
            queries.append(tpl.format(name=name) + suffix)

        # Deduplicate preserving order
        seen: set = set()
        unique = []
        for q in queries:
            k = q.lower()
            if k not in seen:
                seen.add(k)
                unique.append(q)

        self._queries = unique
        return self._queries

    def _narrow_queries(self) -> List[str]:
        """Return only the narrow (quoted) queries — first 10-14 entries."""
        narrow_count = len(self.NARROW_QUERY_TEMPLATES)
        # Add industry extras count
        ind = self.merchant.industry.lower() if self.merchant.industry else ""
        for key, templates in INDUSTRY_EXTRA_QUERIES.items():
            if key in ind:
                narrow_count += len(templates)
        return self._queries[:narrow_count]

    def _broad_queries(self) -> List[str]:
        """Return only the broad (unquoted) fallback queries."""
        narrow_count = len(self._narrow_queries())
        return self._queries[narrow_count:]

    # ─────────────────────────────────────────────────────────────────
    # GOOGLE NEWS RSS HELPER
    # ─────────────────────────────────────────────────────────────────

    def _fetch_google_rss(self, query: str, max_items: int = 5) -> List[NewsArticle]:
        """Fetch Google News RSS for a single query. No locale restriction."""
        results: List[NewsArticle] = []
        root = self._get_rss(
            f"https://news.google.com/rss/search?q={quote(query)}&hl=en"
        )
        if root is None:
            return results

        for item in root.findall(".//item")[:max_items]:
            src_el = item.find("source")
            desc = re.sub(r"<[^>]+>", "", item.findtext("description", ""))
            art = self._to_article(
                source="Google News",
                title=item.findtext("title", ""),
                url=item.findtext("link", ""),
                snippet=desc,
                published_date=item.findtext("pubDate", ""),
                author=src_el.text if src_el is not None else "",
            )
            if art:
                results.append(art)
        return results

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 1: Google News RSS
    # ─────────────────────────────────────────────────────────────────

    def _search_google_news(self) -> List[NewsArticle]:
        """
        Google News RSS — free, no key, international results.

        Strategy:
            1. Try narrow (quoted) queries first
            2. If 0 results → try broad (unquoted) queries
            3. If still 0 → try just the brand name + industry
        """
        print("  [1/8] Google News RSS")
        results: List[NewsArticle] = []

        # Pass 1: Narrow queries (top 4)
        for query in self._narrow_queries()[:4]:
            results.extend(self._fetch_google_rss(query))
            time.sleep(0.5)

        # Pass 2: If 0, try broad queries
        if not results:
            print("       → 0 from narrow queries, trying broad search...")
            for query in self._broad_queries()[:3]:
                results.extend(self._fetch_google_rss(query, max_items=8))
                time.sleep(0.5)

        # Pass 3: If still 0, try just the brand name
        if not results:
            last_resort = f"{self._search_name} news problems issues"
            if self.merchant.industry:
                last_resort += f" {self.merchant.industry}"
            results.extend(self._fetch_google_rss(last_resort, max_items=10))

        print(f"       → {len(results)} relevant articles")
        return results

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 2: Reddit JSON API
    # ─────────────────────────────────────────────────────────────────

    def _search_reddit(self) -> List[NewsArticle]:
        """Reddit public search — tries quoted then unquoted."""
        print("  [2/8] Reddit")
        results: List[NewsArticle] = []
        name = self._search_name

        # Pass 1: Quoted queries
        queries = [
            f'"{name}" fraud OR scam OR lawsuit',
            f'"{name}" investigation OR breach OR controversy',
        ]
        # Pass 2: Unquoted broader queries (used if pass 1 returns 0)
        broad_queries = [
            f'{name} scam fraud complaint',
            f'{name} problems issues controversy',
        ]

        for q_list in [queries, broad_queries]:
            for q in q_list:
                data = self._get_json(
                    "https://www.reddit.com/search.json",
                    params={"q": q, "sort": "relevance", "limit": "10", "t": "all"},
                )
                for child in ((data or {}).get("data") or {}).get("children", []):
                    p = child.get("data", {})
                    if p.get("score", 0) < 2:
                        continue
                    ts = p.get("created_utc", 0)
                    art = self._to_article(
                        source=f"Reddit r/{p.get('subreddit', '?')}",
                        title=p.get("title", ""),
                        url=f"https://reddit.com{p.get('permalink', '')}",
                        snippet=(p.get("selftext", "") or p.get("title", ""))[:500],
                        published_date=datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else "",
                        author=p.get("author", ""),
                    )
                    if art:
                        results.append(art)
                time.sleep(2)

            if results:
                break  # Got results from narrow, skip broad

        print(f"       → {len(results)} relevant posts")
        return results

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 3: Hacker News Algolia API
    # ─────────────────────────────────────────────────────────────────

    def _search_hacker_news(self) -> List[NewsArticle]:
        """HN Algolia — free, fast, tech/fintech community discussions."""
        print("  [3/8] Hacker News")
        results: List[NewsArticle] = []
        name = self._search_name

        for q in [f"{name} fraud scam", f"{name} lawsuit regulatory fine",
                  f"{name} controversy scandal"]:
            data = self._get_json(
                "https://hn.algolia.com/api/v1/search_by_date",
                params={"query": q, "tags": "(story,comment)", "hitsPerPage": "10"},
            )
            for hit in (data or {}).get("hits", []):
                title = hit.get("title", "") or ""
                body = hit.get("story_text", "") or hit.get("comment_text", "") or ""
                oid = hit.get("objectID", "")
                art = self._to_article(
                    source="Hacker News",
                    title=title or body[:100],
                    url=hit.get("url") or f"https://news.ycombinator.com/item?id={oid}",
                    snippet=f"{title} {body}"[:500],
                    published_date=hit.get("created_at", ""),
                    author=hit.get("author", ""),
                )
                if art:
                    results.append(art)
            time.sleep(0.3)

        print(f"       → {len(results)} relevant items")
        return results

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 4: CFPB Consumer Complaints
    # ─────────────────────────────────────────────────────────────────

    def _search_cfpb(self) -> List[NewsArticle]:
        """CFPB Consumer Complaint Database — free gov API, daily updates."""
        print("  [4/8] CFPB Complaints")
        results: List[NewsArticle] = []

        # Try normalized name first, then full legal name
        for search_term in [self._search_name, self.merchant.name]:
            data = self._get_json(
                "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/",
                params={"search_term": search_term, "size": "10",
                        "sort": "created_date_desc", "no_aggs": "true"},
            )
            for hit in ((data or {}).get("hits") or {}).get("hits", []):
                s = hit.get("_source", {})
                text = (
                    f"Product: {s.get('product', '')}. "
                    f"Issue: {s.get('issue', '')}. Sub-issue: {s.get('sub_issue', '')}. "
                    f"Response: {s.get('company_response', '')}. "
                    f"{(s.get('complaint_what_happened') or '')[:300]}"
                )
                art = self._to_article(
                    source="CFPB",
                    title=f"Consumer complaint: {s.get('issue', 'N/A')}",
                    url="https://www.consumerfinance.gov/data-research/consumer-complaints/",
                    snippet=text,
                    published_date=s.get("date_received", ""),
                    author="Consumer (via CFPB)",
                )
                if art:
                    results.append(art)
            if results:
                break

        print(f"       → {len(results)} relevant complaints")
        return results

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 5: SEC EDGAR Full-Text Search
    # ─────────────────────────────────────────────────────────────────

    def _search_sec_edgar(self) -> List[NewsArticle]:
        """SEC EDGAR EFTS — free, covers enforcement and regulatory filings."""
        print("  [5/8] SEC EDGAR")
        results: List[NewsArticle] = []
        name = self._search_name

        for params in [
            {"q": f'"{name}" AND (fraud OR penalty OR violation)', "forms": "8-K,10-K"},
            {"q": f'"{name}"', "forms": "LIT-REL,ADMIN"},
        ]:
            params.update({"dateRange": "custom", "startdt": "2020-01-01",
                           "enddt": datetime.now().strftime("%Y-%m-%d")})
            data = self._get_json("https://efts.sec.gov/LATEST/search-index", params=params)
            for hit in ((data or {}).get("hits") or {}).get("hits", [])[:5]:
                s = hit.get("_source", {})
                ftype = s.get("file_type", s.get("form_type", ""))
                entity = (s.get("display_names") or [""])[0]
                art = self._to_article(
                    source="SEC EDGAR",
                    title=f"SEC {ftype}: {entity}",
                    url=f"https://www.sec.gov/cgi-bin/browse-edgar?company={quote_plus(name)}&action=getcompany",
                    snippet=f"Filing: {ftype}. Entity: {entity}. Date: {s.get('file_date', '')}.",
                    published_date=s.get("file_date", ""),
                    author="SEC",
                )
                if art:
                    results.append(art)

        print(f"       → {len(results)} relevant filings")
        return results

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 6: Wikipedia API
    # ─────────────────────────────────────────────────────────────────

    def _check_wikipedia(self) -> Dict[str, Any]:
        """Wikipedia REST — entity verification, notability, controversy flags."""
        print("  [6/8] Wikipedia")
        info: Dict[str, Any] = {"exists": False, "extract": "", "controversy_hits": 0}

        # Try multiple name variants
        name_variants = list(dict.fromkeys([
            self._search_name,
            self.merchant.name,
            self._search_name.replace(" ", "_"),
        ]))

        for name_variant in name_variants:
            slug = name_variant.replace(" ", "_")
            data = self._get_json(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(slug)}"
            )
            if data and "extract" in data:
                info["exists"] = True
                extract = data["extract"]
                info["extract"] = extract[:500]
                controversy_terms = {
                    "controversy", "scandal", "lawsuit", "fraud",
                    "investigation", "fine", "penalty", "criticized", "accused",
                }
                info["controversy_hits"] = sum(t in extract.lower() for t in controversy_terms)
                break

        print(f"       → page {'found' if info['exists'] else 'not found'}")
        self.wiki_info = info
        return info

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 7: DuckDuckGo Lite
    # ─────────────────────────────────────────────────────────────────

    def _search_duckduckgo(self) -> List[NewsArticle]:
        """DDG Lite HTML — tries multiple queries, avoids rate-limited JSON endpoint."""
        print("  [7/8] DuckDuckGo Lite")
        results: List[NewsArticle] = []

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            print("       → beautifulsoup4 not installed, skipping")
            return results

        # Multiple queries with decreasing specificity
        queries = [
            f"{self._search_name} fraud OR scandal OR lawsuit OR investigation",
            f"{self._search_name} controversy allegations complaints",
            f"{self._search_name} problems issues negative news",
        ]

        for query in queries:
            try:
                resp = self.session.post("https://lite.duckduckgo.com/lite/",
                                         data={"q": query}, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                for link in (soup.select("a.result-link") or soup.select("td a[href^='http']"))[:10]:
                    href = link.get("href", "")
                    title = link.get_text(strip=True)
                    if not href or not title or "duckduckgo" in href:
                        continue
                    snippet = ""
                    parent_row = link.find_parent("tr")
                    if parent_row:
                        next_row = parent_row.find_next_sibling("tr")
                        if next_row:
                            snippet = next_row.get_text(strip=True)[:300]

                    art = self._to_article(source="DuckDuckGo", title=title,
                                            url=href, snippet=snippet)
                    if art:
                        results.append(art)
            except Exception as exc:
                print(f"       → query failed: {exc}")

            if results:
                break  # Got results, skip less specific queries
            time.sleep(1)

        print(f"       → {len(results)} relevant results")
        return results

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 8: Stack Exchange
    # ─────────────────────────────────────────────────────────────────

    def _search_stackexchange(self) -> List[NewsArticle]:
        """Stack Exchange Money SE — free API, finance community Q&A."""
        print("  [8/8] Stack Exchange")
        results: List[NewsArticle] = []

        data = self._get_json(
            "https://api.stackexchange.com/2.3/search/advanced",
            params={"order": "desc", "sort": "relevance",
                    "q": f"{self._search_name} fraud complaint scam",
                    "site": "money", "pagesize": "5", "filter": "default"},
        )
        for item in (data or {}).get("items", []):
            ts = item.get("creation_date", 0)
            art = self._to_article(
                source="Stack Exchange",
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=f"Tags: {', '.join(item.get('tags', []))}. Score: {item.get('score', 0)}",
                published_date=datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else "",
                author=item.get("owner", {}).get("display_name", ""),
            )
            if art:
                results.append(art)

        print(f"       → {len(results)} relevant discussions")
        return results

    # ─────────────────────────────────────────────────────────────────
    # COLLECT ALL SOURCES
    # ─────────────────────────────────────────────────────────────────

    def _collect_all_sources(self) -> List[NewsArticle]:
        """Run all 8 source searches, deduplicate by URL."""
        extra = ""
        if self._search_name != self.merchant.name:
            extra = f" (legal: {self.merchant.name})"
        print(f"\n  Searching 8 sources for '{self._search_name}'{extra}...\n")

        seen_urls: set = set()
        combined: List[NewsArticle] = []

        for method in [
            self._search_google_news, self._search_reddit,
            self._search_hacker_news, self._search_cfpb,
            self._search_sec_edgar, self._search_duckduckgo,
            self._search_stackexchange,
        ]:
            try:
                for art in method():
                    if art.url and art.url in seen_urls:
                        continue
                    if art.url:
                        seen_urls.add(art.url)
                    combined.append(art)
            except Exception as exc:
                print(f"    ✗ {method.__name__} failed: {exc}")

        self.articles = combined
        return combined

    # ─────────────────────────────────────────────────────────────────
    # LLM ANALYSIS
    # ─────────────────────────────────────────────────────────────────

    def _call_llm(self, system: str, user: str, temperature: float = 0.1,
                  max_tokens: int = 3000) -> str:
        """OpenAI call with 3 retries."""
        for attempt in range(3):
            try:
                resp = self.llm.chat.completions.create(
                    model=self.model, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                print(f"    ⚠ LLM attempt {attempt+1}/3 failed: {exc}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return ""

    def _analyze_articles(self) -> Tuple[List[RiskFinding], str]:
        """Send collected articles to GPT-4o-mini for risk classification."""
        if not self.articles:
            return [], "No articles collected from any source."

        blocks = []
        for i, a in enumerate(self.articles[:20]):
            blocks.append(
                f"[{i+1}] Source={a.source} | Date={a.published_date}\n"
                f"Title: {a.title}\nSnippet: {a.snippet[:300]}\nURL: {a.url}"
            )

        system = """You are an expert adverse media screening analyst. Analyze articles from MULTIPLE sources to identify merchant risks.

For EACH relevant article determine:
- is_about_target: bool (disambiguation — is this actually about the target merchant?)
- is_negative: bool
- category: FINANCIAL | LEGAL | REGULATORY | REPUTATIONAL | OPERATIONAL | SANCTIONS
- severity: CRITICAL | HIGH | MEDIUM | LOW
- confidence: 0.0–1.0

SEVERITY GUIDE:
  CRITICAL = active sanctions, terrorism financing, confirmed fraud convictions
  HIGH     = indictments, major fines (>$1M), fraud charges, corruption
  MEDIUM   = civil litigation, data breaches, regulatory warnings, consumer issues
  LOW      = minor infractions, resolved issues, isolated complaints

CROSS-SOURCE RULES:
- Same issue in 2+ sources → higher confidence and severity
- Government sources (CFPB, SEC) outweigh social media
- Single-source findings → note lower confidence

Return ONLY valid JSON:
{
  "findings": [
    {"article_index":1, "is_about_target":true, "is_negative":true,
     "title":"...", "category":"FINANCIAL", "severity":"HIGH",
     "summary":"2-3 sentences", "confidence":0.85, "evidence":"key fact",
     "requires_manual_review":false, "corroborated_by":["3","7"]}
  ],
  "cross_source_summary": "2-3 sentences on patterns across sources",
  "key_risk_indicators": ["indicator 1", "indicator 2"],
  "overall_assessment": "1-2 sentence verdict"
}"""

        user = (
            f"TARGET: {self.merchant.name} (search name: {self._search_name}) | "
            f"Industry: {self.merchant.industry or 'N/A'} | "
            f"URL: {self.merchant.url or 'N/A'} | "
            f"ARTICLES ({len(self.articles)} from "
            f"{len(set(a.source.split()[0] for a in self.articles))} source types):\n\n"
            + "\n---\n".join(blocks)
            + "\n\nAnalyze and return JSON."
        )

        print("\n  Running GPT-4o-mini analysis...")
        raw = self._call_llm(system, user)
        return self._parse_llm_output(raw)

    def _parse_llm_output(self, raw: str) -> Tuple[List[RiskFinding], str]:
        """Parse LLM JSON → findings + cross-source summary."""
        if not raw:
            return [], "LLM analysis unavailable."

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    return [], "Failed to parse LLM JSON."
            else:
                return [], "No JSON in LLM response."

        valid_sev = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        valid_cat = {"FINANCIAL", "LEGAL", "REGULATORY", "REPUTATIONAL", "OPERATIONAL", "SANCTIONS"}

        findings = []
        for r in data.get("findings", []):
            if not r.get("is_about_target", True) or not r.get("is_negative", True):
                continue
            sev = r.get("severity", "LOW").upper()
            cat = r.get("category", "REPUTATIONAL").upper()
            findings.append(RiskFinding(
                title=r.get("title", "Unnamed"),
                category=cat if cat in valid_cat else "REPUTATIONAL",
                severity=sev if sev in valid_sev else "LOW",
                summary=r.get("summary", ""),
                confidence=max(0.0, min(1.0, float(r.get("confidence", 0.5)))),
                evidence_snippet=r.get("evidence", ""),
                requires_manual_review=r.get("requires_manual_review", False),
            ))

        self._key_risk_indicators = data.get("key_risk_indicators", [])
        return findings, data.get("cross_source_summary", "")

    # ─────────────────────────────────────────────────────────────────
    # SUMMARY GENERATION
    # ─────────────────────────────────────────────────────────────────

    def _generate_summary(self, findings: List[RiskFinding],
                          data_suff: str, corr: float) -> str:
        """Concise executive summary via GPT-4o-mini."""
        f_text = "\n".join(
            f"- [{f.severity}] {f.category}: {f.title} ({f.confidence:.0%})"
            for f in findings
        ) or "No adverse findings."

        sources = ", ".join(set(a.source.split()[0] for a in self.articles)) or "None"

        prompt = (
            f"Write a 4-6 sentence executive summary for this merchant risk screening.\n\n"
            f"Merchant: {self.merchant.name} | Industry: {self.merchant.industry or 'N/A'}\n"
            f"Data sufficiency: {data_suff} | Sources: {sources}\n"
            f"Articles: {len(self.articles)} | Corroboration: {corr:.0%}\n\n"
            f"Findings:\n{f_text}\n\n"
            f"Be professional and objective. Note corroborated vs single-source findings. "
            f"Flag data sufficiency concerns. Recommend next steps."
        )
        return self._call_llm("You are a senior compliance analyst.", prompt,
                               temperature=0.2, max_tokens=400) or \
               f"Screening for {self.merchant.name}: {len(findings)} finding(s). Sufficiency: {data_suff}."

    # ─────────────────────────────────────────────────────────────────
    # RISK SCORING & DECISION
    # ─────────────────────────────────────────────────────────────────

    def _score_data_sufficiency(self) -> str:
        """Assess data availability — flags thin-file entities."""
        score = 0
        if len(self.articles) >= 5:   score += 3
        elif self.articles:           score += 1
        if self.merchant.url:         score += 1
        if self.wiki_info.get("exists"): score += 2
        if len(set(a.source.split()[0] for a in self.articles)) >= 3: score += 1
        if   score >= 5: return "SUFFICIENT"
        elif score >= 2: return "LIMITED"
        return "INSUFFICIENT"

    def _score_corroboration(self, findings: List[RiskFinding]) -> float:
        """How well are findings corroborated across independent sources?"""
        if not findings or len(self.articles) < 2:
            return 0.0
        diversity = min(1.0, len(set(a.source.split()[0] for a in self.articles)) / 5)
        cats = Counter(f.category for f in findings)
        repeats = sum(1 for v in cats.values() if v > 1)
        return round(diversity * 0.5 + min(1.0, repeats / 3) * 0.5, 2)

    def _score_risk(self, findings: List[RiskFinding], suff: str, corr: float) -> float:
        """Compute 0–100 risk score. Corroborated findings score higher."""
        if not findings:
            return 25.0 if suff == "INSUFFICIENT" else (10.0 if suff == "LIMITED" else 0.0)
        w = {"CRITICAL": 40, "HIGH": 25, "MEDIUM": 12, "LOW": 5}
        base = sum(w.get(f.severity, 5) * f.confidence for f in findings)
        return min(100.0, round(base * (1 + corr * 0.3), 1))

    def _pick_severity(self, findings: List[RiskFinding]) -> str:
        """Pick severity based on the aggregate risk score, not max finding."""
        if not findings:
            return "NONE"
        score = self._score_risk(findings, "SUFFICIENT", 0.0)
        if score >= 76: return "CRITICAL"
        if score >= 51: return "HIGH"
        if score >= 26: return "MEDIUM"
        return "LOW"

    def _decide_action(self, sev: str, score: float, suff: str) -> Tuple[str, str]:
        """Map severity/score → onboarding action + rationale."""
        if sev == "CRITICAL" or score >= 76:
            return "DECLINE", "Critical adverse media identified — reject onboarding."
        if sev == "HIGH" or score >= 51:
            return "MANUAL_REVIEW", "High-severity findings — human analyst review required."
        if sev == "MEDIUM" or score >= 26:
            return "EDD", "Medium-severity findings — enhanced due diligence warranted."
        if suff == "INSUFFICIENT":
            return "MANUAL_REVIEW", "Insufficient public data — enhanced due diligence for thin-file entity."
        return "APPROVE", "No significant adverse media — standard onboarding may proceed."

    # ─────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────

    def process(self) -> NegativeNewsOutput:
        """Run the full Agent 5 screening pipeline."""
        t0 = time.time()
        sid = (f"SCR-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-"
               f"{hashlib.md5(self.merchant.name.encode()).hexdigest()[:8]}")

        print(f"\n{'='*70}")
        print(f"  AGENT 5 - NEGATIVE NEWS DETECTION")
        print(f"  Merchant: {self.merchant.name}")
        if self._search_name != self.merchant.name:
            print(f"  Search Name: {self._search_name} (legal suffixes stripped)")
        print(f"  Industry: {self.merchant.industry or 'N/A'} | URL: {self.merchant.url or 'N/A'}")
        print(f"{'='*70}")

        self._trace.append("Starting adverse media screening.")

        # 1 — queries
        self._build_queries()
        self._trace.append(f"Generated {len(self._queries)} search queries "
                           f"({len(self._narrow_queries())} narrow + {len(self._broad_queries())} broad).")

        # 2 — multi-source search
        self._collect_all_sources()
        by_source = Counter(a.source.split()[0] for a in self.articles)
        self._trace.append(f"Collected {len(self.articles)} articles from {len(by_source)} source types.")

        # 3 — wikipedia check
        self._check_wikipedia()
        self._trace.append(f"Wikipedia: {'found' if self.wiki_info.get('exists') else 'not found'}.")

        # 4 — LLM analysis
        self._trace.append("Running GPT-4o-mini analysis...")
        findings, cross_summary = self._analyze_articles()
        self._trace.append(f"Identified {len(findings)} adverse finding(s).")

        # 5 — scoring
        suff = self._score_data_sufficiency()
        corr = self._score_corroboration(findings)
        sev  = self._pick_severity(findings)
        score = self._score_risk(findings, suff, corr)
        action, rationale = self._decide_action(sev, score, suff)
        self._trace.append(f"Risk: severity={sev}, score={score}, action={action}")

        # 6 — summary
        self._trace.append("Generating executive summary...")
        summary = self._generate_summary(findings, suff, corr)

        elapsed = round(time.time() - t0, 2)

        output = NegativeNewsOutput(
            merchant_name=self.merchant.name,
            screening_id=sid,
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_risk_severity=sev,
            overall_risk_score=score,
            recommended_action=action,
            data_sufficiency=suff,
            findings=[asdict(f) for f in findings],
            findings_by_category=dict(Counter(f.category for f in findings)),
            findings_by_severity=dict(Counter(f.severity for f in findings)),
            total_articles_collected=len(self.articles),
            articles_by_source=dict(by_source),
            sources_consulted=list(by_source.keys()),
            corroboration_score=corr,
            cross_source_summary=cross_summary,
            key_risk_indicators=self._key_risk_indicators,
            executive_summary=summary,
            recommendation_rationale=rationale,
            search_queries_used=self._queries,
            reasoning_trace=self._trace,
            processing_time_seconds=elapsed,
            model_used=self.model,
        )

        self._print_report(output)
        return output

    def _print_report(self, r: NegativeNewsOutput):
        S = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "NONE": "✅"}
        A = {"DECLINE": "🚫", "MANUAL_REVIEW": "⚠️", "EDD": "🔍", "APPROVE": "✅"}

        print(f"\n{'='*70}")
        print(f"  📋 SCREENING REPORT: {r.merchant_name}")
        print(f"{'='*70}")
        print(f"  Screening ID:       {r.screening_id}")
        print(f"  Severity:           {S.get(r.overall_risk_severity, '?')} {r.overall_risk_severity}")
        print(f"  Risk Score:         {r.overall_risk_score}/100")
        print(f"  Action:             {A.get(r.recommended_action, '?')} {r.recommended_action}")
        print(f"  Rationale:          {r.recommendation_rationale}")
        print(f"  Data Sufficiency:   {r.data_sufficiency}")
        print(f"  Articles:           {r.total_articles_collected}")
        print(f"  Sources:            {', '.join(r.sources_consulted) or 'None'}")
        print(f"  Corroboration:      {r.corroboration_score:.0%}")
        print(f"  Time:               {r.processing_time_seconds}s")

        if r.findings:
            print(f"\n  {'─'*66}")
            print(f"  FINDINGS ({len(r.findings)}):")
            for i, f in enumerate(r.findings, 1):
                rev = " ⚠️ REVIEW" if f.get("requires_manual_review") else ""
                print(f"  {i}. {S.get(f['severity'],'?')} [{f['severity']}] {f['category']}: {f['title']}")
                print(f"     Confidence: {f['confidence']:.0%}{rev}")
                if f.get("summary"):
                    print(f"     {f['summary'][:120]}")
                print()
        else:
            print(f"\n No adverse media findings.\n")

        if r.key_risk_indicators:
            print(f"  KEY RISK INDICATORS:")
            for ind in r.key_risk_indicators:
                print(f"    • {ind}")

        if r.cross_source_summary:
            print(f"\n  CROSS-SOURCE ANALYSIS:")
            print(f"  {r.cross_source_summary}")

        print(f"\n  EXECUTIVE SUMMARY:")
        print(f"  {r.executive_summary.replace('**', '')}")

        print(f"\n  REASONING TRACE:")
        for step in r.reasoning_trace:
            print(f"    {step}")
        print(f"{'='*70}\n")

# Example
if __name__ == "__main__":

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("Set OPENAI_API_KEY: export OPENAI_API_KEY='sk-...'")
        exit(1)

    # Test: BetKing Online Ltd. (gambling, expects name normalization)
    print("=" * 70)
    print("TEST: BetKing Online Ltd. → search name: BetKing")
    print("=" * 70)

    agent5 = Agent5NegativeNewsDetector(
        merchant_name="BetKing Online Ltd.",
        merchant_url="https://m.betking.com/",
        industry="Gambling",
        openai_api_key=OPENAI_API_KEY,
    )
    print(f"  Legal name:  {agent5.merchant.name}")
    print(f"  Search name: {agent5._search_name}")
    print(f"  Queries preview:")
    agent5._build_queries()
    for i, q in enumerate(agent5._queries[:5], 1):
        print(f"    {i}. {q}")
    print(f"    ... ({len(agent5._queries)} total)")

    result = agent5.process()

    with open("agent5_report.json", "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print("Saved: agent5_report.json")