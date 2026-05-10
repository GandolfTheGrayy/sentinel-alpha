import type { VercelRequest, VercelResponse } from '@vercel/node';

const FINNHUB = 'https://finnhub.io/api/v1';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const q = String(req.query.q || '').trim();
  if (q.length < 1) return res.status(400).json({ error: 'q required' });
  const key = process.env.FINNHUB_API_KEY;
  if (!key) return res.status(500).json({ error: 'FINNHUB_API_KEY missing' });

  try {
    const r = await fetch(`${FINNHUB}/search?q=${encodeURIComponent(q)}&exchange=US&token=${key}`);
    const data: any = await r.json();
    const results = (data.result || [])
      .filter((x: any) => x.type === 'Common Stock' && !x.symbol.includes('.'))
      .slice(0, 10)
      .map((x: any) => ({ ticker: x.symbol, name: x.description, type: x.type }));
    res.setHeader('Cache-Control', 's-maxage=300, stale-while-revalidate=600');
    return res.status(200).json({ results });
  } catch (e: any) {
    return res.status(500).json({ error: e?.message || 'search failed' });
  }
}
