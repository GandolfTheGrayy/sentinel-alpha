import type { VercelRequest, VercelResponse } from '@vercel/node';

const FINNHUB = 'https://finnhub.io/api/v1';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const ticker = String(req.query.ticker || '').toUpperCase().trim();
  if (!/^[A-Z][A-Z.\-]{0,7}$/.test(ticker)) {
    return res.status(400).json({ error: 'invalid ticker format' });
  }
  const key = process.env.FINNHUB_API_KEY;
  if (!key) return res.status(500).json({ error: 'FINNHUB_API_KEY missing' });

  try {
    const [qRes, pRes] = await Promise.all([
      fetch(`${FINNHUB}/quote?symbol=${ticker}&token=${key}`),
      fetch(`${FINNHUB}/stock/profile2?symbol=${ticker}&token=${key}`),
    ]);
    const quote: any = await qRes.json();
    const profile: any = await pRes.json();
    if (!quote.c) return res.status(404).json({ error: 'ticker not found' });

    res.setHeader('Cache-Control', 's-maxage=10, stale-while-revalidate=30');
    return res.status(200).json({
      ticker,
      name: profile.name || ticker,
      logo: profile.logo || null,
      industry: profile.finnhubIndustry || null,
      exchange: profile.exchange || null,
      currency: profile.currency || 'USD',
      price: quote.c,
      change: quote.d,
      change_pct: quote.dp,
      high: quote.h,
      low: quote.l,
      open: quote.o,
      prev_close: quote.pc,
      timestamp: quote.t,
    });
  } catch (e: any) {
    return res.status(500).json({ error: e?.message || 'quote failed' });
  }
}
