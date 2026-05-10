import type { VercelRequest, VercelResponse } from '@vercel/node';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';

const FINNHUB = 'https://finnhub.io/api/v1';
const CLAUDE_MODEL = 'claude-sonnet-4-6';
const GEMINI_RESEARCH_MODEL = 'gemini-2.5-flash';

const SYNTH_PROMPT = (ctx: string) => `You are an investment forecaster. Read the data below and output ONLY a JSON object — no prose outside it — matching this exact schema:

{
  "summary": "<2-3 sentence overall thesis>",
  "predictions": {
    "one_day":   {"direction":"up|down|neutral","magnitude_pct":<float>,"confidence":<int 0-100>,"rationale":"<one sentence>"},
    "one_week":  {"direction":"up|down|neutral","magnitude_pct":<float>,"confidence":<int 0-100>,"rationale":"<one sentence>"},
    "one_month": {"direction":"up|down|neutral","magnitude_pct":<float>,"confidence":<int 0-100>,"rationale":"<one sentence>"},
    "one_year":  {"direction":"up|down|neutral","magnitude_pct":<float>,"confidence":<int 0-100>,"rationale":"<one sentence>"}
  },
  "key_catalysts": [
    {"date":"YYYY-MM-DD or 'unknown'","event":"<short>","direction":"bullish|bearish|mixed","impact":"low|medium|high"}
  ],
  "risks": ["<bullet>", "..."],
  "evidence_cited": ["<short citation references actually used>"]
}

Reason carefully. Don't be falsely confident — if a horizon is genuinely uncertain, set direction "neutral" and confidence below 50.

DATA:
${ctx}`;

async function fetchJson(url: string) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const ticker = String(req.query.ticker || '').toUpperCase().trim();
  if (!/^[A-Z][A-Z.\-]{0,7}$/.test(ticker)) {
    return res.status(400).json({ error: 'invalid ticker format' });
  }
  const finnhubKey = process.env.FINNHUB_API_KEY;
  const anthropicKey = process.env.ANTHROPIC_API_KEY;
  const geminiKey = process.env.GEMINI_API_KEY;
  if (!finnhubKey || !anthropicKey || !geminiKey) {
    return res.status(500).json({ error: 'missing API keys (FINNHUB / ANTHROPIC / GEMINI)' });
  }

  try {
    const today = new Date().toISOString().slice(0, 10);
    const weekAgo = new Date(Date.now() - 7 * 86400000).toISOString().slice(0, 10);
    const yearOut = new Date(Date.now() + 365 * 86400000).toISOString().slice(0, 10);

    // Parallel Finnhub fetch: quote, profile, news, earnings
    const [quote, profile, newsRaw, earningsRaw]: any[] = await Promise.all([
      fetchJson(`${FINNHUB}/quote?symbol=${ticker}&token=${finnhubKey}`),
      fetchJson(`${FINNHUB}/stock/profile2?symbol=${ticker}&token=${finnhubKey}`),
      fetchJson(`${FINNHUB}/company-news?symbol=${ticker}&from=${weekAgo}&to=${today}&token=${finnhubKey}`).catch(() => []),
      fetchJson(`${FINNHUB}/calendar/earnings?symbol=${ticker}&from=${today}&to=${yearOut}&token=${finnhubKey}`).catch(() => ({})),
    ]);

    if (!quote.c) return res.status(404).json({ error: 'ticker not found on Finnhub' });

    const newsTop = (Array.isArray(newsRaw) ? newsRaw : []).slice(0, 8).map((n: any) => ({
      headline: n.headline,
      source: n.source,
      datetime: n.datetime ? new Date(n.datetime * 1000).toISOString().slice(0, 10) : null,
      summary: (n.summary || '').slice(0, 220),
      url: n.url,
    }));
    const earnings = (earningsRaw?.earningsCalendar || []).slice(0, 4);

    // Gemini deep research with Google Search grounding
    const genai = new GoogleGenerativeAI(geminiKey);
    let researchText = '';
    let researchSources: any[] = [];
    try {
      const model = genai.getGenerativeModel({
        model: GEMINI_RESEARCH_MODEL,
        tools: [{ googleSearch: {} } as any],
      });
      const prompt = `Conduct deep equity research on ${ticker} (${profile.name || ticker}). Use Google Search aggressively across multiple queries.

Cover, with concrete dates, figures, and source attributions:
1. Recent material events (last 90 days): earnings beats/misses with magnitude, executive changes, lawsuits, product launches, regulatory actions, M&A activity, FDA decisions.
2. Upcoming catalysts (next 90 days): earnings dates, product launches, conference appearances, regulatory deadlines, ex-dividend dates.
3. Sector and macro context: industry trends, top competitors' recent moves, macro themes (rates, geopolitics) directly affecting this name.
4. Recent analyst views: notable upgrades/downgrades with new price targets and reasoning.
5. Notable insider transactions or institutional positioning shifts (13F changes, Form 4 activity).
6. Any contrarian or under-discussed angle worth flagging.

Format: structured markdown. Be specific. No fluff. Cite sources inline.`;
      const r = await model.generateContent(prompt);
      researchText = r.response.text();
      const candidate: any = r.response.candidates?.[0];
      const grounding = candidate?.groundingMetadata || candidate?.grounding_metadata;
      researchSources = (grounding?.groundingChunks || grounding?.grounding_chunks || []).map((c: any) => ({
        title: c.web?.title,
        uri: c.web?.uri,
      })).filter((s: any) => s.uri);
    } catch (e: any) {
      researchText = `(deep research unavailable: ${e?.message || 'gemini error'})`;
    }

    // Claude synthesis
    const anthropic = new Anthropic({ apiKey: anthropicKey });
    const ctx = `Ticker: ${ticker} (${profile.name || ticker})
Industry: ${profile.finnhubIndustry || 'unknown'} · Exchange: ${profile.exchange || ''}
Current price: $${quote.c} (today ${quote.dp >= 0 ? '+' : ''}${(quote.dp || 0).toFixed(2)}%)
Day range: $${quote.l} – $${quote.h} · Prev close: $${quote.pc}

Recent news (last 7 days, Finnhub):
${newsTop.map((n: any) => `- [${n.datetime}] ${n.headline} — ${n.source}`).join('\n') || '(none)'}

Upcoming earnings (Finnhub calendar):
${earnings.map((e: any) => `- ${e.date}: EPS est ${e.epsEstimate ?? 'n/a'}, rev est ${e.revenueEstimate ?? 'n/a'}`).join('\n') || '(none scheduled)'}

DEEP RESEARCH (Gemini + Google Search):
${researchText}`;

    const synth = await anthropic.messages.create({
      model: CLAUDE_MODEL,
      max_tokens: 2200,
      messages: [{ role: 'user', content: SYNTH_PROMPT(ctx) }],
    });
    const txt = synth.content
      .filter((b: any) => b.type === 'text')
      .map((b: any) => b.text)
      .join('');
    const m = txt.match(/\{[\s\S]*\}/);
    let report: any;
    if (m) {
      try { report = JSON.parse(m[0]); }
      catch { report = { error: 'json parse failed', raw: txt.slice(0, 600) }; }
    } else {
      report = { error: 'no json in response', raw: txt.slice(0, 600) };
    }

    res.setHeader('Cache-Control', 's-maxage=300, stale-while-revalidate=900');
    return res.status(200).json({
      ticker,
      name: profile.name || ticker,
      logo: profile.logo || null,
      industry: profile.finnhubIndustry || null,
      generated_at: new Date().toISOString(),
      quote: {
        price: quote.c,
        change_pct: quote.dp,
        high: quote.h,
        low: quote.l,
        prev_close: quote.pc,
      },
      news: newsTop,
      earnings,
      research: { text: researchText, sources: researchSources },
      report,
    });
  } catch (e: any) {
    console.error(e);
    return res.status(500).json({ error: e?.message || 'report failed' });
  }
}
