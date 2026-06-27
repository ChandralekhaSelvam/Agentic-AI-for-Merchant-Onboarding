from typing import Any, Dict, List, Optional, Tuple
import re
import pathlib
import pandas as pd

URL_PATTERN = re.compile(
    r"^https?://[\w\-]+(\.[\w\-]+)+([/?#].*)?$",
    re.IGNORECASE,
)


def load_policy_data(xlsx_path: str) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load policy tables from the provided Excel file.

    Returns:
      - industry_map: { canonical_industry: [alias1, alias2, ...] }
      - mcc_map: { "1234": {"description": <industry>, "risk_level": <>, "risk_score": <> } }
      - high_risk_reasons: list of {"flag": bool, "reason": str}
      - tpb_rules: list of dicts representing TPB/threshold rows (if present)
    Heuristics are tolerant to sheets with columns similar to the prompt excerpt.
    """
    industry_map: Dict[str, List[str]] = {}
    mcc_map: Dict[str, Dict[str, Any]] = {}
    high_risk_reasons: List[Dict[str, Any]] = []
    tpb_rules: List[Dict[str, Any]] = []

    p = pathlib.Path(xlsx_path)
    if not p.exists():
        return industry_map, mcc_map, high_risk_reasons, tpb_rules

    try:
        sheets = pd.read_excel(p, sheet_name=None, engine="openpyxl")
    except Exception:
        return industry_map, mcc_map, high_risk_reasons, tpb_rules

    for sheet_name, df in sheets.items():
        cols = [str(c).lower() for c in df.columns.astype(str)]

        # Parse Industry <-> MCC rows (sheet containing Industry and MCC columns)
        if any("industry" in c for c in cols) and any("mcc" in c or "mcc_code" in c or "code" in c for c in cols):
            industry_col = next((c for c in df.columns if "industry" in str(c).lower()), None)
            mcc_col = next((c for c in df.columns if "mcc" in str(c).lower() or "mcc_code" in str(c).lower() or (str(c).lower()=="code")), None)
            risk_col = next((c for c in df.columns if "risk_level" in str(c).lower() or (("risk" in str(c).lower()) and "score" not in str(c).lower())), None)
            score_col = next((c for c in df.columns if "risk_score" in str(c).lower() or "score" in str(c).lower()), None)

            for _, row in df.iterrows():
                raw_ind = row.get(industry_col)
                raw_code = row.get(mcc_col)
                if pd.isna(raw_ind):
                    continue
                # split aliases (slashes, commas, pipes)
                tokens = [t.strip() for t in re.split(r"[\/,;|]", str(raw_ind)) if t and not pd.isna(t)]
                if not tokens:
                    continue
                canonical = tokens[0]
                aliases = []
                for t in tokens:
                    if t and t not in aliases:
                        aliases.append(t)
                industry_map.setdefault(canonical, [])
                for a in aliases:
                    if a not in industry_map[canonical]:
                        industry_map[canonical].append(a)

                if pd.isna(raw_code):
                    continue
                # normalize numeric codes and zero-pad when sensible
                if isinstance(raw_code, (int, float)) and raw_code == int(raw_code):
                    code = str(int(raw_code)).zfill(4)
                else:
                    code = str(raw_code).strip()
                    if code.isdigit() and len(code) <= 4:
                        code = code.zfill(4)

                if not code:
                    continue

                risk_level = str(row.get(risk_col)).strip() if risk_col and not pd.isna(row.get(risk_col)) else None
                risk_score = None
                if score_col and not pd.isna(row.get(score_col)):
                    try:
                        risk_score = float(row.get(score_col))
                    except Exception:
                        risk_score = None

                mcc_map[code] = {"description": canonical, "risk_level": risk_level, "risk_score": risk_score}

        # Parse High_Risk_Flag / Risk_Reason style table
        elif any("risk_reason" in c or "high_risk_flag" in c or ("risk" in c and "reason" in c) for c in cols):
            flag_col = next((c for c in df.columns if "high_risk_flag" in str(c).lower() or "high_risk" in str(c).lower()), None)
            reason_col = next((c for c in df.columns if "risk_reason" in str(c).lower() or "reason" in str(c).lower()), None)
            if reason_col is None:
                continue
            for _, row in df.iterrows():
                flag = True
                if flag_col and not pd.isna(row.get(flag_col)):
                    v = row.get(flag_col)
                    if isinstance(v, str):
                        flag = v.strip().lower() in ("true", "1", "yes")
                    else:
                        flag = bool(v)
                reason = str(row.get(reason_col)).strip() if not pd.isna(row.get(reason_col)) else ""
                high_risk_reasons.append({"flag": flag, "reason": reason})

        # Parse TPB / thresholds table heuristics
        elif any("tpb" in c or ("percentage" in c and "tpb" in "tpb") for c in cols) or ("tpb_percentage_min" in "".join(cols) or "tpb" in "".join(cols)):
            # generic row copy
            for _, row in df.iterrows():
                tpb_rules.append(row.to_dict())

        # Generic fallback: if sheet name or columns suggest TPB / Questionnaires / Playbooks,
        # we ignore for validation agent but could be returned in future.
        else:
            # attempt to pick up TPB-like tables by presence of 'TPB' or 'Action_Required' columns
            if any("tpb" in c or "action_required" in c for c in cols):
                for _, row in df.iterrows():
                    tpb_rules.append(row.to_dict())

    return industry_map, mcc_map, high_risk_reasons, tpb_rules


class ValidationAgent:
    """Validates merchant profile completeness, format, taxonomy correctness and flags risk metadata."""

    REQUIRED_FIELDS = ["merchant_name", "address", "website", "industry", "mcc"]

    def __init__(self, policy_xlsx: Optional[str] = None):
        self.INDUSTRY_TAXONOMY, self.MCC_CODE_MAPPING, self.HIGH_RISK_REASONS, self.TPB_RULES = load_policy_data(policy_xlsx) if policy_xlsx else ({}, {}, [], [])
        # fallback defaults if Excel not provided or empty
        if not self.INDUSTRY_TAXONOMY:
            self.INDUSTRY_TAXONOMY = {
                "Retail": ["Retail", "E-commerce", "Shopping"],
                "Gambling": ["Gambling", "Gaming", "Casino"],
                "Crypto": ["Crypto", "Cryptocurrency", "Digital Assets"],
                "Cannabis": ["Cannabis", "CBD", "Marijuana"],
                "Travel": ["Travel", "Hospitality", "Tourism"],
                "Healthcare": ["Healthcare", "Medical", "Pharma"],
                "Food Services": ["Food Services", "Restaurants", "Catering"],
                "FinTech": ["Finance"]
            }
        if not self.MCC_CODE_MAPPING:
            self.MCC_CODE_MAPPING = {
                "6012": {"description": "finance"},
                "5411": {"description": "Grocery Stores, Supermarkets", "risk_level": "Low", "risk_score": 15},
                "5812": {"description": "Eating Places, Restaurants", "risk_level": "Low", "risk_score": 20},
                "5311": {"description": "Department Stores", "risk_level": "Low", "risk_score": 10},
                "5999": {"description": "Miscellaneous and Specialty Retail Stores", "risk_level": "Medium", "risk_score": 40},
                "4829": {"description": "Money Transfer, Financial Institutions", "risk_level": "High", "risk_score": 85},
                "7995": {"description": "Betting, including Lottery Tickets, Casinos, and Gambling", "risk_level": "High", "risk_score": 95},
            }

    def _normalize_industry(self, industry: Optional[str]) -> str:
        if not industry:
            return ""
        value = industry.strip()
        # handle multi-token industry field
        if any(sep in value for sep in ["/", ",", "|", ";"]):
            tokens = [t.strip() for t in re.split(r"[\/,;|]", value) if t.strip()]
            if tokens:
                value = tokens[0]
        lowered = value.lower()
        for canonical, aliases in self.INDUSTRY_TAXONOMY.items():
            if lowered == canonical.lower():
                return canonical
            for alias in aliases:
                if lowered == alias.lower():
                    return canonical
        # substring fallback
        for canonical, aliases in self.INDUSTRY_TAXONOMY.items():
            if canonical.lower() in lowered:
                return canonical
            for alias in aliases:
                if alias.lower() in lowered:
                    return canonical
        return value

    def _validate_website(self, website: str) -> bool:
        return bool(website and URL_PATTERN.match(website.strip()))

    def _validate_mcc(self, mcc: str) -> Tuple[bool, Optional[str]]:
        if not mcc:
            return False, None
        code = str(mcc).strip()
        if not re.fullmatch(r"\d{4}", code):
            if code.isdigit() and len(code) < 4:
                code = code.zfill(4)
            else:
                return False, None
        if code not in self.MCC_CODE_MAPPING:
            return False, None
        return True, code

    def _assess_risk_from_mcc(self, mcc_code: str) -> Optional[Dict[str, Any]]:
        """Return risk dict from MCC mapping if present."""
        if not mcc_code:
            return None
        info = self.MCC_CODE_MAPPING.get(mcc_code)
        if not info:
            return None
        return {"mcc": mcc_code, "risk_level": info.get("risk_level"), "risk_score": info.get("risk_score"), "description": info.get("description")}

    def _assess_risk_from_industry(self, industry: str) -> Optional[Dict[str, Any]]:
        """If industry canonical maps to known HIGH/Medium risk via MCC map or taxonomy heuristics, return risk metadata."""
        if not industry:
            return None
        # direct industry -> search mcc_map for matching description
        for code, meta in self.MCC_CODE_MAPPING.items():
            desc = str(meta.get("description", "")).lower()
            if industry.lower() in desc:
                return {"industry": industry, "risk_level": meta.get("risk_level"), "risk_score": meta.get("risk_score"), "mcc": code}
        # fallback: if industry header itself exists in taxonomy and known risky names
        for canonical, aliases in self.INDUSTRY_TAXONOMY.items():
            if industry == canonical:
                # try find any MCC marked high for this canonical
                for code, meta in self.MCC_CODE_MAPPING.items():
                    if str(meta.get("description", "")).lower().startswith(canonical.lower()):
                        return {"industry": industry, "risk_level": meta.get("risk_level"), "risk_score": meta.get("risk_score"), "mcc": code}
        return None

    def validate(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        validated = {**profile}
        issues: List[Dict[str, Any]] = []
        flagged: List[Dict[str, Any]] = []

        missing_fields = [field for field in self.REQUIRED_FIELDS if not validated.get(field)]

        for field in missing_fields:
            issues.append({
                "field": field,
                "issue": "Missing required field",
                "suggestion": "Please provide this value.",
                "code": "MISSING_FIELD",
            })

        raw_ind = validated.get("industry", "")
        industry = self._normalize_industry(raw_ind)
        validated["industry"] = industry
        if raw_ind and industry not in self.INDUSTRY_TAXONOMY:
            issues.append({
                "field": "industry",
                "issue": "Unknown industry classification",
                "suggestion": "Use a valid industry from the policy taxonomy.",
                "code": "UNKNOWN_INDUSTRY",
            })

        raw_mcc = validated.get("mcc", "")
        if raw_mcc:
            is_mcc_valid, normalized_mcc = self._validate_mcc(raw_mcc)
            if not is_mcc_valid:
                issues.append({
                    "field": "mcc",
                    "issue": "Invalid or unknown MCC code",
                    "suggestion": "Provide a 4-digit MCC from the policy reference.",
                    "code": "INVALID_MCC",
                })
            else:
                validated["mcc"] = normalized_mcc
                validated["mcc_description"] = self.MCC_CODE_MAPPING[normalized_mcc].get("description")
                # assess risk from MCC
                risk = self._assess_risk_from_mcc(normalized_mcc)
                if risk and (str(risk.get("risk_level", "")).lower() == "high" or (risk.get("risk_score") is not None and float(risk.get("risk_score", 0)) >= 80)):
                    reason = f"Mapped MCC {normalized_mcc} flagged as {risk.get('risk_level')} (score={risk.get('risk_score')})"
                    flagged.append({"field": "mcc", "mcc": normalized_mcc, "risk_level": risk.get("risk_level"), "risk_score": risk.get("risk_score"), "reason": reason, "code": "HIGH_RISK_MCC"})
        else:
            validated["mcc"] = None

        website = validated.get("website", "")
        if website:
            if not self._validate_website(website):
                issues.append({
                    "field": "website",
                    "issue": "Invalid website format",
                    "suggestion": "Use a valid URL starting with http:// or https://.",
                    "code": "INVALID_WEBSITE",
                })

        # industry-based risk assessment
        ind_risk = self._assess_risk_from_industry(industry)
        if ind_risk and (str(ind_risk.get("risk_level", "")).lower() == "high" or (ind_risk.get("risk_score") is not None and float(ind_risk.get("risk_score", 0)) >= 80)):
            reason = f"Industry '{industry}' maps to risk_level {ind_risk.get('risk_level')} (score={ind_risk.get('risk_score')})"
            # avoid duplicate if same MCC already flagged
            already = any(f.get("mcc") == ind_risk.get("mcc") for f in flagged)
            if not already:
                flagged.append({"field": "industry", "industry": industry, "risk_level": ind_risk.get("risk_level"), "risk_score": ind_risk.get("risk_score"), "reason": reason, "code": "HIGH_RISK_INDUSTRY"})

        # include high risk reasons table entries as informational flags
        for hr in self.HIGH_RISK_REASONS:
            if hr.get("flag") and hr.get("reason"):
                flagged.append({"field": "policy", "reason": hr.get("reason"), "code": "HIGH_RISK_REASON"})

        # Score calculation
        base_score = 100
        missing_penalty = len(missing_fields) * 15
        issue_penalty = len(issues) * 10
        risk_penalty = 0
        # penalize for flagged high/medium risk presence
        for f in flagged:
            rl = str(f.get("risk_level") or "").lower()
            if rl == "high":
                risk_penalty += 25
            elif rl == "medium":
                risk_penalty += 10
        score = max(0, min(100, base_score - missing_penalty - issue_penalty - risk_penalty))

        status = "valid" if score >= 85 else "partial" if score >= 60 else "invalid"

        validated_profile = {
            **validated,
            "data_quality_score": score,
            "status": status,
        }

        data_quality_report = {
            "score": score,
            "status": status,
            "missing_fields": missing_fields,
            "issues": issues,
            "flagged_items": flagged,
        }

        result = {
            "validated_profile": validated_profile,
            "data_quality_report": data_quality_report,
            "flagged_items": flagged,
        }

        return result

if __name__ == "__main__":
    import json

    agent = ValidationAgent()

    test_cases = [
        {
            "name": "Scenario A: Perfect Low-Risk Merchant",
            "data": {
                "merchant_name": "Global Retail Corp",
                "address": "123 Business Way, NY",
                "website": "https://globalretail.com",
                "industry": "E-commerce",
                "mcc": "5311"
            }
        },
        {
            "name": "Scenario B: Broken Data (Missing fields & Bad URL)",
            "data": {
                "merchant_name": "Incomplete Shop",
                "website": "not-a-url",
                "industry": "Unknown Category",
                "mcc": "99" 
            }
        },
        {
            "name": "Scenario C: High-Risk Merchant (Gambling)",
            "data": {
                "merchant_name": "Ace Casino",
                "address": "777 Vegas Blvd",
                "website": "https://ace-casino.com",
                "industry": "Gambling",
                "mcc": "7995"
            }
        }
    ]

    print("="*80)
    print("VALIDATION AGENT INDEPENDENT TEST REPORT")
    print("="*80)

    for case in test_cases:
        print(f"\nRUNNING: {case['name']}")
        result = agent.validate(case['data'])
        
        report = result["data_quality_report"]
        print(f"  Score:  {report['score']}/100")
        print(f"  Status: {report['status'].upper()}")
        
        if report['issues']:
            print("  Issues Found:")
            for issue in report['issues']:
                print(f"    - [{issue['code']}] {issue['field']}: {issue['issue']}")
        
        if report['flagged_items']:
            print("  Risk Flags:")
            for flag in report['flagged_items']:
                print(f"    - [{flag['code']}] {flag['reason']}")
        
        print("-" * 40)
