import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const FINNHUB = 'https://finnhub.io/api/v1';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_KEY;
  const finnhubKey = process.env.FINNHUB_API_KEY;
  if (!url || !key || !finnhubKey) return res.status(500).json({ error: 'missing keys' });
  const supa = createClient(url, key, { auth: { persistSession: false } });

  try {
    const { data: positions } = await supa.from('positions').select('*').eq('closed', false);
    const open = positions || [];
    const tickers = [...new Set(open.map((p: any) => p.ticker))];

    const meta: Record<string, { industry: string; logo: string; name: string; price: number | null }> = {};
    await Promise.all(tickers.map(async (t) => {
      try {
        const [pRes, qRes] = await Promise.all([
          fetch(`${FINNHUB}/stock/profile2?symbol=${t}&token=${finnhubKey}`),
          fetch(`${FINNHUB}/quote?symbol=${t}&token=${finnhubKey}`),
        ]);
        const profile: any = await pRes.json();
        const quote: any = await qRes.json();
        meta[t] = {
          industry: profile.finnhubIndustry || 'Uncategorized',
          logo: profile.logo || '',
          name: profile.name || t,
          price: quote.c || null,
        };
      } catch {
        meta[t] = { industry: 'Uncategorized', logo: '', name: t, price: null };
      }
    }));

    const byIndustry: Record<string, { count: number; value: number; portfolios: Set<string>; tickers: Set<string> }> = {};
    for (const p of open) {
      const m = meta[p.ticker] || { industry: 'Uncategorized', price: null };
      const ind = m.industry;
      const cur = m.price ?? Number(p.entry_price);
      const value = Number(p.shares) * cur;
      if (!byIndustry[ind]) byIndustry[ind] = { count: 0, value: 0, portfolios: new Set(), tickers: new Set() };
      byIndustry[ind].count++;
      byIndustry[ind].value += value;
      byIndustry[ind].portfolios.add(p.portfolio_id);
      byIndustry[ind].tickers.add(p.ticker);
    }

    const sectors = Object.entries(byIndustry)
      .map(([industry, v]) => ({
        industry,
        positions: v.count,
        value: Math.round(v.value * 100) / 100,
        portfolios: [...v.portfolios],
        tickers: [...v.tickers],
      }))
      .sort((a, b) => b.value - a.value);

    res.setHeader('Cache-Control', 's-maxage=120, stale-while-revalidate=300');
    return res.status(200).json({ sectors, ticker_meta: meta });
  } catch (e: any) {
    return res.status(500).json({ error: e?.message || 'sectors failed' });
  }
}
