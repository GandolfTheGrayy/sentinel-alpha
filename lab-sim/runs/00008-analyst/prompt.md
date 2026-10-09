# Sentinel lab digest - 2026-10-07 03:15 UTC (analyst session)
Mode **sim** | equity $99,398 | today +0 (+0.00%) | week +0 | since start -600 (-0.60%) | open lots 18
Population: 14 active, 0 incubating, 2 probation, 0 retired (cap 24). Claude budget this week: $0.00 of $62.50.

## Since the last analyst session (2026-10-06T03:15 UTC)
- 65 trades closed
- Lifecycle: vwap_revert#1: active -> probation (last 35 trades -0.54R (CI upper -0.14))

## Tournament (all-time on closed trades; last30 = last 30 trades)
| variant | status | alloc | n | expR (95% CI) | win | PF | last30 expR | maxDD% | P&L | origin |
|---|---|---|---|---|---|---|---|---|---|---|
| squeeze#2 | active | 0.20 | 6 | +0.87 (-0.43..+2.17) | 50% | 1.859 | +0.87 (n=6) | 0.4 | +149 | seed |
| overnight#1 | active | 0.20 | 4 | -0.00 (-0.00..-0.00) | 50% | 1.034 | -0.00 (n=4) | 0.0 | +0 | seed |
| random_entry#1 | active | 0.01 | 31 | -0.13 (-0.51..+0.20) | 35% | 0.5 | -0.09 (n=30) | 4.7 | -75 | seed |
| meanrev_bb#1 | active | 0.02 | 28 | -0.23 (-0.57..+0.07) | 39% | 1.066 | -0.23 (n=28) | 6.6 | +19 | seed |
| squeeze#1 | active | 0.02 | 18 | -0.42 (-0.89..+0.18) | 28% | 0.241 | -0.42 (n=18) | 17.8 | -402 | seed |
| crypto_mtf#1 | active | 0.02 | 11 | -0.47 (-0.95..+0.27) | 9% | 0.115 | -0.47 (n=11) | 13.7 | -317 | seed |
| trend_ema#1 | active | 0.02 | 6 | -0.47 (-0.83..-0.18) | 0% | 0.0 | -0.47 (n=6) | 6.4 | -147 | seed |
| xs_momentum#1 | active | 0.02 | 2 | -0.63 (-0.63..-0.63) | 0% | 0.0 | -0.63 (n=2) | 3.7 | -85 | seed |
| breakout_orb#1 | active | 0.02 | 4 | -0.78 (-0.78..-0.78) | 25% | 0.013 | -0.78 (n=4) | 4.9 | -112 | seed |
| breakout_donchian#2 | active | 0.02 | 1 | -0.97 (-0.97..-0.97) | 0% | 0.0 | -0.97 (n=1) | 1.9 | -45 | seed |
| trend_ema#2 | active | 0.20 | 1 | -1.04 (-1.04..-1.04) | 0% | 0.0 | -1.04 (n=1) | 0.2 | -41 | seed |
| breakout_orb#2 | active | 0.15 | 3 | -1.16 (-1.16..-1.16) | 0% | 0.0 | -1.16 (n=3) | 0.4 | -63 | seed |
| breakout_donchian#1 | active | 0.02 | 0 | n/a (+0.00..+0.00) | n/a | n/a | n/a (n=0) | 0.0 | +0 | seed |
| buy_hold#1 | active | 0.01 | 0 | n/a (+0.00..+0.00) | n/a | n/a | n/a (n=0) | 0.0 | +0 | seed |
| meanrev_bb#2 | probation | 0.02 | 60 | -0.12 (-0.35..+0.11) | 52% | 0.621 | +0.02 (n=30) | 13.9 | -191 | seed |
| vwap_revert#1 | probation | 0.02 | 35 | -0.54 (-0.87..-0.18) | 17% | 0.265 | -0.63 (n=30) | 46.1 | -891 | seed |

Control (random entries, same exits): n=31, expectancy -0.13R, win 35%. A family is only interesting if it beats this with a non-overlapping confidence interval.

## Factor attribution
**Attribution window:** last 45 days, 205 trades, expectancy -0.29R (95% CI -0.43..-0.14), win rate 33%, P&L -2174.

**By family:** meanrev_bb n=88 -0.15R wr=48%; vwap_revert n=35 -0.54R wr=17%; random_entry n=27 -0.27R wr=26%; squeeze n=24 -0.10R wr=33%; crypto_mtf n=11 -0.47R wr=9%; breakout_orb n=7 -0.94R wr=14%; trend_ema n=7 -0.55R wr=0%; overnight n=4 -0.00R wr=50%

**By market:** equities n=153 -0.32R wr=35%; crypto n=52 -0.20R wr=25%

**By regime:** range/high_vol n=52 -0.06R wr=40%; range/mid_vol n=44 -0.50R wr=18%; range/low_vol n=39 -0.11R wr=41%; trend/low_vol n=31 -0.49R wr=26%; trend/mid_vol n=24 -0.23R wr=38%; trend/high_vol n=15 -0.55R wr=33%

**By exit reason:** stop n=112 -1.06R wr=0%; target n=26 +1.62R wr=100%; time n=26 +0.34R wr=54%; session_end n=20 +0.02R wr=45%; signal n=20 +0.47R wr=90%; rebalance n=1 -0.24R wr=0%

**By side:** long n=168 -0.38R wr=27%; short n=37 +0.15R wr=57%

**Top features** (gradient boosting permutation importance (R-multiple)):
- `rsi_2` imp=0.193 -> 0.00745..2.04: -0.35R (n=41), 2.04..7.71: -0.34R (n=41), 7.71..60.4: -0.38R (n=41), 60.4..94: -0.47R (n=41), 94..99.8: +0.12R (n=41)
- `atr_pct` imp=0.130 -> 0.193..0.365: -0.30R (n=41), 0.365..0.458: -0.31R (n=41), 0.458..0.58: -0.56R (n=41), 0.58..0.704: -0.23R (n=41), 0.704..4.46: -0.03R (n=41)
- `dow` imp=0.107 -> Mon: -0.51R (n=69), Tue: -0.42R (n=53), Fri: -0.15R (n=62), Sat: +0.22R (n=13), Sun: +0.58R (n=8)
- `bb_pos` imp=0.074 -> -0.369..-0.144: -0.18R (n=41), -0.144..0.0429: -0.44R (n=41), 0.0429..0.389: -0.32R (n=41), 0.389..1.01: -0.46R (n=41), 1.01..1.44: -0.03R (n=41)
- `vol_ratio` imp=0.070 -> 0.489..0.952: -0.38R (n=41), 0.952..1.06: -0.08R (n=41), 1.06..1.32: -0.03R (n=41), 1.32..1.54: -0.53R (n=41), 1.54..2.35: -0.40R (n=41)
- `signal_strength` imp=0.068 -> 0.179..0.5: -0.35R (n=51), 0.5..0.729: -0.60R (n=31), 0.729..1: -0.18R (n=123)
- `ret_5` imp=0.059 -> -0.0985..-0.00941: -0.37R (n=41), -0.00941..-0.00635: -0.29R (n=41), -0.00635..0.000315: -0.39R (n=41), 0.000315..0.00721: -0.24R (n=41), 0.00721..0.054: -0.14R (n=41)
- `mkt_dist_sma50_pct` imp=0.059 -> -15.8..-10.5: -0.09R (n=41), -10.5..-6.63: -0.37R (n=41), -6.63..-5.93: -0.19R (n=41), -5.93..-3.99: -0.65R (n=41), -3.99..-2.62: -0.13R (n=41)
- `hour_et` imp=0.054 -> 02h: -0.53R (n=2), 03h: -0.89R (n=5), 05h: -0.04R (n=1), 06h: -1.01R (n=2), 07h: -0.51R (n=5), 08h: -1.09R (n=1), 09h: -0.30R (n=22), 10h: -0.49R (n=13), 11h: -0.30R (n=11), 12h: -0.45R (n=24), 13h: -0.10R (n=28), 14h: -0.55R (n=31), 15h: -0.05R (n=35), 16h: -0.32R (n=10), 17h: -1.07R (n=1), 18h: -1.11R (n=2), 19h: +1.25R (n=2), 20h: +0.97R (n=3), 21h: +0.58R (n=4), 22h: -1.04R (n=1), 23h: +0.58R (n=2)
- `sleeve_dd_pct` imp=0.054 -> -0.001..0.00145: -0.30R (n=41), 0.00145..1.49: -0.25R (n=41), 1.49..3.93: -0.23R (n=41), 3.93..6.86: -0.52R (n=42), 6.86..14.5: -0.12R (n=40)

**Logistic (P(win), standardised coefs):** hour_et +0.59, dow +0.56, sleeve_dd_pct +0.50, rsi_14 +0.49, mkt_ret_1d -0.42, ret_1 +0.40, bb_pos -0.35, mkt_rvol_20d -0.31

**Time-of-week:** best Fri 15h n=9 +0.23R; Fri 13h n=9 +0.01R | worst Mon 14h n=10 -0.86R; Mon 12h n=10 -0.76R; Tue 09h n=9 -0.44R

## Notable trades (last 7 days)
- WIN squeeze#2 long LTC/USD +2.83R (target, 871 min) - squeeze release after 4 bars, momentum long
- WIN squeeze#2 long ETH/USD +2.80R (target, 3295 min) - squeeze release after 4 bars, momentum long
- WIN squeeze#2 long BTC/USD +2.79R (target, 2874 min) - squeeze release after 4 bars, momentum long
- LOSS breakout_orb#2 long TSLA -1.31R (stop, 25 min) - ORB 15m long break, range 0.86%, vol x1.5
- LOSS meanrev_bb#2 long NFLX -1.27R (stop, 21 min) - close below BB lower, RSI2 2
- LOSS vwap_revert#1 long QQQ -1.25R (stop, 23 min) - -4.4 ATR below VWAP on vol x1.5

## Market regime
- SPY: 5d -4.5%, 20d -2.7%, realised vol 34% ann., ADX14 16, vs SMA50 -6.2%
- BTC/USD: 5d -5.1%, 20d -10.7%, realised vol 56% ann., ADX14 21, vs SMA50 -12.5%

## Families available
- `breakout_donchian` (equities/crypto, 1h): Donchian channel breakout with a chandelier-style ATR trailing stop; works on equities and crypto. Params: tf:1h/4h/15m, n:10-60, stop_atr:1.0-4.0, trail_atr:1.5-6.0, max_hold:24-400, min_atr_pct:0.0-1.5
- `breakout_orb` (equities, 5m): Opening-range breakout: trade the first break of the opening N-minute range with volume confirmation; flat by the close. Params: or_minutes:15/30/60, latest_entry_min:60-300, vol_mult:0.8-2.5, r_mult:1.0-4.0, stop_atr:0.8-3.0, max_hold:12-78
- `buy_hold` (equities/crypto, 1d, CONTROL): CONTROL: buy-and-hold the benchmark of each market with a very wide stop. The passive baseline. Params: none
- `crypto_mtf` (crypto, 15m): Long-only crypto: 4h EMA trend up, wait for a 15m RSI pullback, enter on the first up-close; ATR stop, R target and trailing. Params: trend_ema:20-120, fast_ema:8-40, rsi_pullback:20.0-50.0, stop_atr:1.0-4.0, r_mult:1.5-6.0, trail_atr:1.5-6.0, max_hold:24-600
- `meanrev_bb` (equities, 15m): Fade closes outside the Bollinger bands when RSI(2) is extreme; target the middle band; optional daily trend filter. Params: tf:5m/15m, bb_n:10-40, bb_k:1.5-3.0, rsi_lo:3.0-25.0, rsi_hi:75.0-97.0, stop_atr:0.8-3.0, max_hold:4-60, trend_filter:0/1
- `overnight` (equities, 1d): Buy strong names shortly before the close and sell shortly after the next open to harvest the overnight premium. Params: n_symbols:1-8, trend_sma:20-200, max_day_move_pct:1.0-6.0, stop_atr:1.0-5.0, exit_after_open_min:1-60, rank_lookback:5-60
- `random_entry` (equities/crypto, 15m, CONTROL): CONTROL: random entries with the standard ATR stop / R target / time exits. Any real family must beat this. Params: none
- `squeeze` (equities/crypto, 15m): After at least k bars of Bollinger bands inside the Keltner channel, enter when the bands release, in the direction of momentum. Params: tf:15m/1h, min_squeeze:3-20, kc_mult:1.0-2.5, bb_k:1.5-2.5, stop_atr:0.8-3.0, r_mult:1.0-4.0, max_hold:10-200
- `trend_ema` (equities/crypto, 15m): Enter on an EMA fast/slow crossover when ADX confirms a trend; ride it with an ATR trailing stop. Params: tf:15m/1h, fast:5-30, slow:20-120, adx_min:12.0-40.0, stop_atr:1.0-4.0, trail_atr:1.5-5.0, max_hold:20-300
- `vwap_revert` (equities, 5m): When price is stretched more than z ATRs from the session VWAP on a volume climax, fade it with VWAP as the target. Params: tf:5m/15m, z:1.0-4.0, vol_mult:0.8-3.0, min_after_open:5-120, stop_atr:0.8-3.0, max_hold:6-60, latest_entry_min:120-360
- `xs_momentum` (equities, 1d): Each morning rank the equity universe by medium-term return (skipping the last few days) and hold the top names; rotate out when they drop from the top set. Params: lookback:20-160, skip:0-10, n_long:2-8, stop_atr:1.5-6.0, abs_filter:0/1

## Lab memory (your notes from earlier sessions)
_(empty - this is the first session)_

---

Today's task (daily analyst session): review what happened since the last session, update the running
picture of which families/conditions are working, and propose at most 3 small, testable changes (param tweaks or
entry filters) with the strongest evidence. Test the most promising one or two with the backtester before proposing.

Answer with the JSON object described by the schema.