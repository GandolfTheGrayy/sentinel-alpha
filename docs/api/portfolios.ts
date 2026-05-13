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
    const { data: portfolios, error } = await supa.from('portfolios').select('*').order('created_at', { ascending: true });
    if (error) throw error;
    if (!portfolios?.length) return res.status(200).json({ portfolios: [] });

    const { data: positions } = await supa.from('positions').select('*').eq('closed', false);
    const allTickers = [...new Set((positions || []).map((p: any) => p.ticker))];
    const priceMap: Record<string, number> = {};
    await Promise.all(allTickers.map(async (t) => {
      try {
        const r = await fetch(`${FINNHUB}/quote?symbol=${t}&token=${finnhubKey}`);
        const q: any = await r.json();
        if (q?.c) priceMap[t] = q.c;
      } catch {}
    }));

    const summaries = portfolios.map((p: any) => {
      const open = (positions || []).filter((x: any) => x.portfolio_id === p.id);
      const positions_value = open.reduce((s: number, x: any) => {
        const cur = priceMap[x.ticker];
        return s + (cur != null ? Number(x.shares) * cur : Number(x.shares) * Number(x.entry_price));
      }, 0);
      const total_value = Number(p.cash) + positions_value;
      const total_return_pct = ((total_value - Number(p.starting_cash)) / Number(p.starting_cash)) * 100;
      return { ...p, open_count: open.length, positions_value, total_value, total_return_pct };
    });

    res.setHeader('Cache-Control', 'no-store');
    return res.status(200).json({ portfolios: summaries });
  } catch (e: any) {
    return res.status(500).json({ error: e?.message || 'list failed' });
  }
}
