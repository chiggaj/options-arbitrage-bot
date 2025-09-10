# Options Arbitrage Bot

A Python-based trading bot that scans for **options parity dislocations** (mispricings between stock and option legs).  
It identifies **reversals** and other arbitrage setups where margin can be leveraged safely, while enforcing guardrails for risk management.  

Designed for: automated scanning, profitability assessment, audit logging, and later full execution integration (via Tradier or IBKR).  

---

## ⚡ Features
- Detects **options reversals** and parity gaps across top underlyings (SPY, QQQ, AAPL, MSFT, etc.).
- Filters based on liquidity (min OI, volume, staleness).
- Enforces configurable spreads, probability thresholds, and capital usage limits.
- Dual capital model (cash + margin) with safety multipliers.
- Logs all opportunities and skips to structured CSVs for review.
- Configurable tier system for symbol prioritization.
- Prepared for IBKR integration (NBBO, shortable checks, combo order support).

---

## 🛠️ Installation
Clone this repository and install dependencies:

```bash
git clone git@github.com:chiggaj/options-arbitrage-bot.git
cd options-arbitrage-bot
pip install -r requirements.txt
