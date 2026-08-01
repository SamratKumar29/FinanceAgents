"""
portfolio_allocator.py -- Budget-aware portfolio suggestions.

Takes a UserProfile and generates personalized stock/ETF recommendations
with specific allocation amounts based on the user's budget, risk tolerance,
and investment preferences.
"""
import json
from openai import OpenAI
import config
from state import UserProfile


PORTFOLIO_PROMPT = """You are the Chief Investment Officer creating a personalized investment portfolio.

User Profile:
{profile}

Generate a portfolio of exactly {num_positions} positions.

For EACH position provide (use this exact JSON format):
{{"ticker": "...", "company": "...", "allocation_pct": X, "amount": Y, "rationale": "...", "risk_level": "low/medium/high", "category": "ETF/Blue-Chip/Growth/Speculative"}}

Also provide:
- A "summary" section with total allocation, diversification notes
- A "disclaimer" field

Rules:
1. Total allocation_pct must equal 100%
2. All amounts must be in {currency}
3. Allocate based on risk tolerance:
   - Conservative: 60% ETFs/bonds, 40% blue-chip
   - Moderate: 40% ETFs, 40% growth, 20% speculative
   - Aggressive: 20% ETFs, 50% growth, 30% speculative
4. Only suggest stocks available on the user's preferred markets
5. Include the disclaimer: "This is general informational analysis, not personalized financial advice."

Respond with ONLY a JSON object in this structure:
{{
  "positions": [...],
  "summary": "...",
  "disclaimer": "..."
}}"""


def _format_profile(profile: UserProfile) -> str:
    """Format user profile for the LLM prompt."""
    lines = []
    lines.append(f"Location: {profile.get('location', 'Not specified')}")
    lines.append(f"Currency: {profile.get('base_currency', 'Not specified')}")
    lines.append(f"Budget: {profile.get('budget', 0):,.2f} {profile.get('base_currency', '')} ({profile.get('budget_type', 'Not specified')})")
    lines.append(f"Risk Tolerance: {profile.get('risk_tolerance', 'Not specified')}")
    lines.append(f"Investment Horizon: {profile.get('investment_horizon', 'Not specified')}")
    lines.append(f"Goals: {', '.join(profile.get('goals', [])) or 'Not specified'}")
    lines.append(f"Preferred Markets: {', '.join(profile.get('preferred_markets', [])) or 'All'}")
    lines.append(f"Preferred Sectors: {', '.join(profile.get('preferred_sectors', [])) or 'Any'}")
    lines.append(f"Number of Positions: {profile.get('num_positions', 5)}")
    existing = profile.get("existing_investments", [])
    if existing:
        lines.append(f"Existing Investments: {json.dumps(existing)}")
    return "\n".join(lines)


def suggest_portfolio(profile: UserProfile) -> dict:
    """
    Generate portfolio suggestions based on user profile.

    Returns a dict with:
    - positions: list of {ticker, company, allocation_pct, amount, rationale, risk_level, category}
    - summary: diversification notes
    - disclaimer: standard disclaimer
    - total_amount: sum of all amounts
    - currency: the currency used
    """
    num_positions = profile.get("num_positions", 5)
    currency = profile.get("base_currency", "USD")
    budget = profile.get("budget", 0)

    # Suggest num positions based on budget (guideline)
    if budget > 0:
        if budget < 10000:
            suggested = max(3, min(5, num_positions))
        elif budget < 100000:
            suggested = max(5, min(8, num_positions))
        else:
            suggested = max(8, min(12, num_positions))
    else:
        suggested = num_positions

    profile_str = _format_profile(profile)

    client = OpenAI(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": PORTFOLIO_PROMPT.format(
                    profile=profile_str,
                    num_positions=suggested,
                    currency=currency,
                )},
                {"role": "user", "content": f"Generate my portfolio with {suggested} positions for a budget of {budget:,.2f} {currency}."},
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        text = response.choices[0].message.content or "{}"

        # Parse JSON from response (handle markdown fences)
        import re
        text = text.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            result = json.loads(text)

        # Ensure required fields
        result.setdefault("positions", [])
        result.setdefault("summary", "Portfolio generated successfully.")
        result.setdefault("disclaimer", "This is general informational analysis, not personalized financial advice.")
        result["total_amount"] = budget
        result["currency"] = currency
        result["suggested_positions"] = suggested

        return result

    except Exception as e:
        return {
            "positions": [],
            "summary": f"Error generating portfolio: {str(e)}",
            "disclaimer": "This is general informational analysis, not personalized financial advice.",
            "total_amount": budget,
            "currency": currency,
            "suggested_positions": suggested,
        }


def modify_portfolio(current_portfolio: dict, remove_ticker: str, profile: UserProfile) -> dict:
    """Remove a stock from the portfolio and reallocate its amount proportionally."""
    positions = current_portfolio.get("positions", [])
    remaining = [p for p in positions if p.get("ticker", "").upper() != remove_ticker.upper()]

    if not remaining:
        return suggest_portfolio(profile)

    # Get the amount to reallocate
    removed = [p for p in positions if p.get("ticker", "").upper() == remove_ticker.upper()]
    if removed:
        freed_amount = removed[0].get("amount", 0)
    else:
        freed_amount = 0

    # Redistribute proportionally
    total_remaining_pct = sum(p.get("allocation_pct", 0) for p in remaining)
    if total_remaining_pct > 0:
        for p in remaining:
            p["allocation_pct"] = round(p.get("allocation_pct", 0) * 100 / total_remaining_pct, 1)
            p["amount"] = round(p.get("amount", 0) + freed_amount * (p["allocation_pct"] / 100), 2)

    current_portfolio["positions"] = remaining
    return current_portfolio