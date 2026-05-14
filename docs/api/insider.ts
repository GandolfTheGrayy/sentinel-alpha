import type { VercelRequest, VercelResponse } from '@vercel/node';

const SUBMISSIONS = 'https://data.sec.gov/submissions/CIK';
const TICKER_MAP = 'https://www.sec.gov/files/company_tickers.json';
const ARCHIVE = 'https://www.sec.gov/Archives/edgar/data';

let tickerMap: Record<string, number> | null = null;

async function getCik(ticker: string, ua: string): Promise<number | null> {
  if (!tickerMap) {
    try {
      const r = await fetch(TICKER_MAP, { headers: { 'User-Agent': ua } });
      const data: any = await r.json();
      tickerMap = {};
      for (const row of Object.values(data) as any[]) {
        tickerMap[row.ticker.toUpperCase()] = row.cik_str;
      }
    } catch { return null; }
  }
  return tickerMap[ticker.toUpperCase()] ?? null;
}

function parseForm4(xml: string): { acquired: number; disposed: number; value: number; insider: string; title: string } | null {
  const stripped = xml.replace(/<\?xml[^>]*\?>/, '');
  const insider = (stripped.match(/<rptOwnerName>\s*([^<]+)/)?.[1] || '').trim();
  const title = (stripped.match(/<officerTitle>\s*([^<]+)/)?.[1] || '').trim();
  let acquired = 0, disposed = 0, value = 0;
  const txRegex = /<nonDerivativeTransaction>([\s\S]*?)<\/nonDerivativeTransaction>/g;
  let m: RegExpExecArray | null;
  while ((m = txRegex.exec(stripped))) {
    const block = m[1];
    const sh = parseFloat(block.match(/<transactionShares>\s*<value>([\d.]+)/)?.[1] || '0');
    const price = parseFloat(block.match(/<transactionPricePerShare>\s*<value>([\d.]+)/)?.[1] || '0');
    const code = (block.match(/<transactionAcquiredDisposedCode>\s*<value>(\w)/)?.[1] || '').toUpperCase();
    if (code === 'A') { acquired += sh; value += sh * price; }
    else if (code === 'D') { disposed += sh; value -= sh * price; }
  }
  return { acquired, disposed, value, insider, title };
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const ticker = String(req.query.ticker || '').toUpperCase().trim();
  if (!/^[A-Z][A-Z.\-]{0,7}$/.test(ticker)) return res.status(400).json({ error: 'invalid ticker' });
  const ua = process.env.SEC_USER_AGENT;
  if (!ua) return res.status(200).json({ ticker, error: 'SEC_USER_AGENT not configured' });
  const lookback = Math.min(parseInt(String(req.query.days || '60'), 10) || 60, 180);
  const maxFilings = 15;

  try {
    const cik = await getCik(ticker, ua);
    if (!cik) return res.status(200).json({ ticker, filings_count: 0, error: 'ticker not found in SEC ticker map' });
    const cikPadded = String(cik).padStart(10, '0');
    const subRes = await fetch(`${SUBMISSIONS}${cikPadded}.json`, { headers: { 'User-Agent': ua } });
    if (!subRes.ok) throw new Error(`SEC submissions ${subRes.status}`);
    const sub: any = await subRes.json();
    const recent = sub.filings?.recent || {};
    const forms: string[] = recent.form || [];
    const cutoff = new Date(Date.now() - lookback * 86400000).toISOString().slice(0, 10);
    const candidates: { filed: string; accession: string; doc: string }[] = [];
    for (let i = 0; i < forms.length && candidates.length < maxFilings; i++) {
      if (forms[i] !== '4') continue;
      const filed = recent.filingDate[i];
      if (filed < cutoff) continue;
      candidates.push({ filed, accession: recent.accessionNumber[i], doc: recent.primaryDocument[i] });
    }

    const txs: any[] = [];
    for (const c of candidates) {
      try {
        const url = `${ARCHIVE}/${cik}/${c.accession.replace(/-/g, '')}/${c.doc}`;
        const xRes = await fetch(url, { headers: { 'User-Agent': ua } });
        if (!xRes.ok) continue;
        const text = await xRes.text();
        const parsed = parseForm4(text);
        if (parsed && (parsed.acquired > 0 || parsed.disposed > 0)) {
          txs.push({ ...parsed, filed: c.filed });
        }
      } catch {}
    }

    const buys = txs.filter(t => t.acquired > t.disposed).length;
    const sells = txs.filter(t => t.disposed > t.acquired).length;
    const net_value = txs.reduce((s, t) => s + t.value, 0);
    const sentiment = net_value > 100_000 ? 'bullish' : net_value < -100_000 ? 'bearish' : 'neutral';

    res.setHeader('Cache-Control', 's-maxage=900, stale-while-revalidate=3600');
    return res.status(200).json({
      ticker,
      lookback_days: lookback,
      filings_count: txs.length,
      buys,
      sells,
      net_value: Math.round(net_value),
      sentiment,
      transactions: txs.slice(0, 5).map(t => ({
        filed: t.filed,
        insider: t.insider,
        title: t.title,
        acquired: t.acquired,
        disposed: t.disposed,
        value: Math.round(t.value),
      })),
    });
  } catch (e: any) {
    return res.status(500).json({ error: e?.message || 'insider lookup failed' });
  }
}
