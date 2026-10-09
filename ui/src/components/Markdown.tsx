/**
 * A small, safe Markdown renderer. Produces React elements only (no innerHTML), so all
 * text is escaped by React. Supports: headings, paragraphs, bullet/numbered lists,
 * blockquotes, horizontal rules, fenced code, pipe tables, bold/italic, inline code
 * and links (http/https/mailto only).
 */
import clsx from 'clsx'
import { Fragment, useMemo, type ReactNode } from 'react'

type Block =
  | { type: 'h'; level: number; text: string }
  | { type: 'p'; text: string }
  | { type: 'ul'; items: string[] }
  | { type: 'ol'; items: string[]; start: number }
  | { type: 'code'; lang: string; code: string }
  | { type: 'quote'; text: string }
  | { type: 'hr' }
  | { type: 'table'; header: string[]; rows: string[][] }

const FENCE_OPEN = /^\s*```\s*([\w-]*)\s*$/
const FENCE_CLOSE = /^\s*```\s*$/
const HEADING = /^(#{1,6})\s+(.+?)\s*#*\s*$/
const HR = /^\s*([-*_])(\s*\1){2,}\s*$/
const QUOTE = /^\s*>\s?(.*)$/
const UL = /^\s*[-*+]\s+(.*)$/
const OL = /^\s*(\d+)[.)]\s+(.*)$/
const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/

function splitRow(line: string): string[] {
  let s = line.trim()
  if (s.startsWith('|')) s = s.slice(1)
  if (s.endsWith('|')) s = s.slice(0, -1)
  return s.split('|').map((c) => c.trim())
}

export function parseMarkdown(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, '\n').split('\n')
  const blocks: Block[] = []
  let para: string[] = []
  const flush = () => {
    if (para.length) {
      blocks.push({ type: 'p', text: para.join(' ') })
      para = []
    }
  }
  let i = 0
  while (i < lines.length) {
    const line = lines[i] ?? ''
    const fence = FENCE_OPEN.exec(line)
    if (fence) {
      flush()
      const buf: string[] = []
      i++
      while (i < lines.length && !FENCE_CLOSE.test(lines[i] ?? '')) {
        buf.push(lines[i] ?? '')
        i++
      }
      i++
      blocks.push({ type: 'code', lang: fence[1] ?? '', code: buf.join('\n') })
      continue
    }
    if (/^\s*$/.test(line)) {
      flush()
      i++
      continue
    }
    const h = HEADING.exec(line)
    if (h) {
      flush()
      blocks.push({ type: 'h', level: h[1]?.length ?? 1, text: h[2] ?? '' })
      i++
      continue
    }
    if (HR.test(line)) {
      flush()
      blocks.push({ type: 'hr' })
      i++
      continue
    }
    if (QUOTE.test(line)) {
      flush()
      const buf: string[] = []
      while (i < lines.length) {
        const m = QUOTE.exec(lines[i] ?? '')
        if (!m) break
        buf.push(m[1] ?? '')
        i++
      }
      blocks.push({ type: 'quote', text: buf.join(' ') })
      continue
    }
    if (UL.test(line)) {
      flush()
      const items: string[] = []
      while (i < lines.length) {
        const cur = lines[i] ?? ''
        const m = UL.exec(cur)
        if (m) {
          items.push(m[1] ?? '')
          i++
        } else if (/^\s{2,}\S/.test(cur) && items.length) {
          items[items.length - 1] += ' ' + cur.trim()
          i++
        } else break
      }
      blocks.push({ type: 'ul', items })
      continue
    }
    const ol = OL.exec(line)
    if (ol) {
      flush()
      const items: string[] = []
      const start = Number(ol[1] ?? '1') || 1
      while (i < lines.length) {
        const cur = lines[i] ?? ''
        const m = OL.exec(cur)
        if (m) {
          items.push(m[2] ?? '')
          i++
        } else if (/^\s{2,}\S/.test(cur) && items.length) {
          items[items.length - 1] += ' ' + cur.trim()
          i++
        } else break
      }
      blocks.push({ type: 'ol', items, start })
      continue
    }
    if (/^\s*\|/.test(line) && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1] ?? '')) {
      flush()
      const header = splitRow(line)
      i += 2
      const rows: string[][] = []
      while (i < lines.length && /^\s*\|/.test(lines[i] ?? '')) {
        rows.push(splitRow(lines[i] ?? ''))
        i++
      }
      blocks.push({ type: 'table', header, rows })
      continue
    }
    para.push(line.trim())
    i++
  }
  flush()
  return blocks
}

// ---------------------------------------------------------------------------
// Inline
// ---------------------------------------------------------------------------

const INLINE =
  /(`[^`\n]+`)|(\*\*[^*\n]+?\*\*)|(?<![\w])(__[^_\n]+?__)(?![\w])|(\*[^*\s\n](?:[^*\n]*?[^*\s\n])?\*)|(?<![\w])(_[^_\s\n](?:[^_\n]*?[^_\s\n])?_)(?![\w])|(\[[^\]\n]+?\]\([^)\s]+?\))/g

const SAFE_HREF = /^(https?:\/\/|mailto:)/i

export function renderInline(text: string, keyPrefix = 'i'): ReactNode[] {
  const out: ReactNode[] = []
  let last = 0
  let k = 0
  for (const m of text.matchAll(INLINE)) {
    const idx = m.index ?? 0
    if (idx > last) out.push(text.slice(last, idx))
    const [full, code, bold, bold2, ital, ital2, link] = m
    const key = `${keyPrefix}-${k++}`
    if (code) {
      out.push(<code key={key}>{code.slice(1, -1)}</code>)
    } else if (bold || bold2) {
      const inner = (bold ?? bold2 ?? '').slice(2, -2)
      out.push(<strong key={key}>{renderInline(inner, key)}</strong>)
    } else if (ital || ital2) {
      const inner = (ital ?? ital2 ?? '').slice(1, -1)
      out.push(<em key={key}>{renderInline(inner, key)}</em>)
    } else if (link) {
      const lm = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(link)
      const label = lm?.[1] ?? link
      const href = lm?.[2] ?? ''
      if (SAFE_HREF.test(href)) {
        out.push(
          <a key={key} href={href} target="_blank" rel="noopener noreferrer">
            {renderInline(label, key)}
          </a>,
        )
      } else {
        out.push(<span key={key}>{label}</span>)
      }
    } else {
      out.push(full)
    }
    last = idx + full.length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function Markdown({ source, className }: { source: string; className?: string }) {
  const blocks = useMemo(() => parseMarkdown(source ?? ''), [source])
  if (!source?.trim()) return <p className="text-[12.5px] text-muted italic">Nothing written.</p>
  return (
    <div className={clsx('md text-text', className)}>
      {blocks.map((b, i) => {
        switch (b.type) {
          case 'h': {
            const level = Math.min(4, b.level)
            const Tag = (`h${level}`) as 'h1' | 'h2' | 'h3' | 'h4'
            return <Tag key={i}>{renderInline(b.text, `h${i}`)}</Tag>
          }
          case 'p':
            return <p key={i}>{renderInline(b.text, `p${i}`)}</p>
          case 'ul':
            return (
              <ul key={i}>
                {b.items.map((it, j) => (
                  <li key={j}>{renderInline(it, `u${i}-${j}`)}</li>
                ))}
              </ul>
            )
          case 'ol':
            return (
              <ol key={i} start={b.start}>
                {b.items.map((it, j) => (
                  <li key={j}>{renderInline(it, `o${i}-${j}`)}</li>
                ))}
              </ol>
            )
          case 'code':
            return (
              <pre key={i} data-lang={b.lang || undefined}>
                <code>{b.code}</code>
              </pre>
            )
          case 'quote':
            return <blockquote key={i}>{renderInline(b.text, `q${i}`)}</blockquote>
          case 'hr':
            return <hr key={i} />
          case 'table':
            return (
              <div key={i} className="overflow-x-auto">
                <table className="table-base">
                  <thead>
                    <tr>
                      {b.header.map((h, j) => (
                        <th key={j}>{renderInline(h, `th${i}-${j}`)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {b.rows.map((r, j) => (
                      <tr key={j}>
                        {r.map((c, k) => (
                          <td key={k}>{renderInline(c, `td${i}-${j}-${k}`)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          default:
            return <Fragment key={i} />
        }
      })}
    </div>
  )
}
