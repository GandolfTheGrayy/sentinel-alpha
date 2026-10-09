## Lab memory (as of 2026-10-09)
- Baseline: 45d, 205 trades, -0.29R (CI -0.43..-0.14). Random-entry control -0.13R (n=31, CI -0.51..+0.20). Nearly no family is separable from the control yet.
- Exit anatomy: stops are 55% of trades at -1.06R; targets +1.62R, signals +0.47R, time +0.34R. Entry quality is poor (33% win rate).
- Shorts +0.15R (n=37) vs longs -0.38R (n=168) in a weak tape (SPY -6% vs SMA50, BTC -12.5%). HYPOTHESIS, untested: long-biased entries are hurt in downtrends. A regime/side filter is worth testing when the sample grows.
- vwap_revert#1: n=35, -0.54R, CI fully negative. Proposed retire 10-09 (status: pending).
- squeeze#2: +0.87R but n=6, CI spans zero (crypto long wins at about +2.8R each). squeeze#1 is -0.42R (n=18). Watch; do not trust yet.
- meanrev_bb#2: wins often (52%) but PF 0.62, so stop/target asymmetry is poor; last 30 trades ~0R. A wider-target or tighter-stop param test is a candidate.
- Allocator gives 0.20 to trend_ema#2 (n=1) and 0.15 to breakout_orb#2 (n=3, -1.16R): capital on unproven variants. Watch.
- Feature bins (rsi_2, atr_pct, dow, hour) are noisy at n≈41 per bin; no action. Mon/Tue weakness (-0.5R, n=122) is only a sample-adequate pattern, likely regime.
- No backtests were run in the 10-09 session (budget $1.5). Next session: backtest meanrev_bb#2 variants and a side/regime filter, and compare against random_entry.
