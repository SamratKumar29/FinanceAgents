# Vision: a full "family office" of AI experts (future ideas, not implemented)

> **Status: out of scope.** This was the original long-term idea. The current project implements
> only the investment-research slice listed below. Several roles here (tax, legal, estate, trust,
> compliance, insurance) are regulated professional advice in most jurisdictions. An LLM giving
> that advice carries real liability, so they are deliberately not built.

## Current coverage (2026-10)

| Role | Status | Where (in `src/financeagents/`) |
|---|---|---|
| Investment Analyst | Mostly: company brief from market data, financials, analyst view, news. No sector or fund research. | `pipeline.py`, `data/tools.py` |
| Risk Management Specialist | Partial: market-risk metrics only (volatility, Sharpe, drawdown, beta). No insurance or hedging. | `analytics/risk.py` |
| Chief Investment Officer | Partial: brief synthesis and risk-bucket strategy rules. | `pipeline.py`, `portfolio/allocator.py` |
| Portfolio Manager | Partial: one-off educational model allocation. No trading or rebalancing. | `portfolio/` |
| Financial Planner | Minimal: goals and horizon are collected; no financial plan is produced. | `onboarding.py` |
| All other roles below | Not implemented | — |

## Original role list

| Expert                                          | Primary Role                                                                                                       
| **Chief Investment Officer (CIO)**              | Designs the overall investment strategy and decides where capital should be invested. ([Forbes][1])                        |
| **Portfolio Manager**                           | Selects, buys, sells, and rebalances investments to match the strategy. ([Corporate Finance Institute][2])                 |
| **Financial Planner (CFP)**                     | Creates long-term financial plans covering retirement, education, insurance, and major life goals. ([Schwab Brokerage][3]) |
| **Tax Advisor / Chartered Accountant (CA/CPA)** | Minimizes taxes legally through smart structuring and tax planning. ([Corporate Finance Institute][2])                     |
| **Estate Planning Attorney**                    | Ensures wealth is transferred efficiently through wills, trusts, and succession planning. ([Schwab Brokerage][3])          |
| **Corporate Lawyer**                            | Handles contracts, business structures, acquisitions, and legal protection of assets. ([Family Office Advisory][4])        |
| **Risk Management Specialist**                  | Identifies financial risks and recommends insurance and hedging strategies. ([The Wall Street Journal][5])                 |
| **Insurance Advisor**                           | Chooses appropriate life, health, property, liability, and business insurance. ([Schwab Brokerage][3])                     |
| **Accountant / Controller**                     | Tracks cash flow, bookkeeping, financial statements, and reporting. ([Forbes][1])                                          |
| **Chief Financial Officer (CFO)**               | Oversees the family's overall finances, budgeting, liquidity, and financial controls. ([Forbes][1])                        |
| **Investment Analyst**                          | Researches companies, sectors, funds, and investment opportunities. ([Corporate Finance Institute][2])                     |
| **Private Equity Specialist**                   | Evaluates and manages investments in private companies. ([Forbes][1])                                                      |
| **Venture Capital Specialist**                  | Finds and evaluates high-growth startup investment opportunities. ([Forbes][1])                                            |
| **Real Estate Advisor**                         | Manages property acquisitions, rentals, financing, and portfolio optimization. ([Corporate Finance Institute][2])          |
| **Trust Officer / Fiduciary Advisor**           | Administers trusts and ensures assets are managed according to legal wishes. ([The Wall Street Journal][5])                |
| **Philanthropy Advisor**                        | Structures charitable giving to maximize both impact and tax efficiency. ([The Wall Street Journal][5])                    |
| **Banking & Lending Specialist**                | Arranges financing, credit lines, and banking solutions for large assets. ([The Wall Street Journal][6])                   |
| **Compliance Officer**                          | Ensures investments and operations comply with laws and regulations. ([Family Office Advisory][4])                         |
