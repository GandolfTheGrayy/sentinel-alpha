import clsx from 'clsx'
import { BookOpen, Bot, ChevronDown, ChevronRight, FlaskConical, Sparkles } from 'lucide-react'
import { useMemo, useState } from 'react'
import { api } from '../api/client'
import { useAction, usePoll } from '../api/hooks'
import { PROPOSAL_STATUSES, type Decision, type ProposalStatus, type ResearchRun, type ResearchRunDetail, type RunKind } from '../api/types'
import { Markdown } from '../components/Markdown'
import { ProposalCard } from '../components/ProposalCard'
import { Badge, runKindTone, runStatusTone } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { Card, SectionTitle } from '../components/ui/Card'
import { Chips } from '../components/ui/Chips'
import { useConfirm } from '../components/ui/Confirm'
import { EmptyState, ErrorState } from '../components/ui/EmptyState'
import { Skeleton, SkeletonText } from '../components/ui/Skeleton'
import { useToast } from '../components/ui/Toast'
import { fmtDuration, fmtNum, fmtTokens, fmtUsd } from '../lib/format'
import { fmtDateTime, fmtRelative } from '../lib/time'
import { useApp } from '../state/AppContext'

export function ResearchPage() {
  const { changeTick, status } = useApp()
  const toast = useToast()
  const confirm = useConfirm()
  const runs = usePoll(() => api.getResearch(20), 30_000, [changeTick])
  const [pickedId, setSelectedId] = useState<number | null>(null)
  // Default to the most recent run until the user picks one.
  const selectedId = pickedId ?? runs.data?.[0]?.id ?? null

  const selectedRun = runs.data?.find((r) => r.id === selectedId) ?? null
  const detail = usePoll<ResearchRunDetail | null>(
    () => (selectedId === null ? Promise.resolve(null) : api.getResearchRun(selectedId)),
    selectedRun?.status === 'running' ? 5000 : null,
    [selectedId, selectedRun?.status, changeTick],
  )

  const proposals = usePoll(() => api.getProposals({ limit: 100 }), 30_000, [changeTick])
  const memory = usePoll(() => api.getMemory(), 60_000, [changeTick])
  const [propStatus, setPropStatus] = useState<ProposalStatus | null>(null)

  const runResearch = useAction((kind: RunKind) => api.runResearch(kind))
  const decide = useAction((id: number, decision: Decision) => api.decideProposal(id, decision))

  const onRun = async (kind: RunKind) => {
    const res = await runResearch.run(kind)
    if (!res) {
      toast({ tone: 'error', title: `Could not queue ${kind} run`, message: runResearch.error?.message })
      return
    }
    if (res.queued) {
      toast({ tone: 'success', title: `${kind === 'analyst' ? 'Analyst' : 'Strategist'} run queued`, message: 'It will appear in the list once it starts.' })
      void runs.refresh()
    } else {
      toast({ tone: 'warn', title: `${kind === 'analyst' ? 'Analyst' : 'Strategist'} run declined`, message: res.reason })
    }
  }

  const onDecide = async (id: number, decision: Decision) => {
    const ok = await confirm({
      title: `${decision === 'accept' ? 'Accept' : 'Reject'} proposal #${id}?`,
      message:
        decision === 'accept'
          ? 'Accepted proposals are applied by the engine: parameter changes spawn an incubating variant, filters are attached, retirements take effect immediately.'
          : 'The proposal is marked rejected and the reason is fed back into the next research digest.',
      confirmLabel: decision === 'accept' ? 'Accept' : 'Reject',
      tone: decision === 'accept' ? 'primary' : 'danger',
    })
    if (!ok) return
    const updated = await decide.run(id, decision)
    if (updated !== undefined) {
      // The contract does not pin the response body; merge it when it looks like a proposal, and refetch regardless.
      const looksLikeProposal = typeof updated === 'object' && updated !== null && 'status' in updated && 'id' in updated
      toast({ tone: 'success', title: `Proposal #${id} ${looksLikeProposal ? updated.status : decision === 'accept' ? 'accepted' : 'rejected'}` })
      if (looksLikeProposal) {
        proposals.setData((prev) => (prev ? prev.map((p) => (p.id === id ? updated : p)) : prev))
        detail.setData((prev) => (prev ? { ...prev, proposals: prev.proposals.map((p) => (p.id === id ? updated : p)) } : prev))
      }
      void proposals.refresh()
      void detail.refresh()
      void runs.refresh()
    } else {
      toast({ tone: 'error', title: 'Decision failed', message: decide.error?.message ?? 'The API did not accept the decision.' })
    }
  }

  const propCounts = useMemo(() => {
    const c: Record<ProposalStatus, number> = { pending: 0, testing: 0, accepted: 0, rejected: 0 }
    for (const p of proposals.data ?? []) c[p.status] += 1
    return c
  }, [proposals.data])

  const filteredProposals = useMemo(() => {
    const ps = proposals.data ?? []
    return propStatus ? ps.filter((p) => p.status === propStatus) : ps
  }, [proposals.data, propStatus])

  const budget = status?.budget
  const anyRunning = runs.data?.some((r) => r.status === 'running') ?? false

  return (
    <div className="space-y-4">
      <Card>
        <div className="flex flex-col md:flex-row md:items-center gap-3">
          <div className="flex items-start gap-3 min-w-0 flex-1">
            <div className="h-9 w-9 rounded-lg bg-claude/15 text-claude flex items-center justify-center shrink-0">
              <Bot className="h-4.5 w-4.5" aria-hidden />
            </div>
            <div className="min-w-0">
              <div className="text-[13px] font-semibold text-text">Claude research lab</div>
              <div className="text-[12px] text-muted">
                The analyst (Sonnet) runs daily after the close; the strategist (Opus) runs weekly. Claude only writes proposals; every proposal passes a backtest gate before it gets capital.
                {budget && (
                  <>
                    {' '}
                    Budget remaining this week: <span className={clsx('font-medium', budget.remaining_usd <= 0 ? 'text-loss' : 'text-text')}>{fmtUsd(budget.remaining_usd, { decimals: 2 })}</span>.
                  </>
                )}
              </div>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <Button variant="claude" icon={<Sparkles className="h-3.5 w-3.5" />} loading={runResearch.pending} disabled={anyRunning} onClick={() => void onRun('analyst')}>
              Run analyst now
            </Button>
            <Button variant="claude" icon={<FlaskConical className="h-3.5 w-3.5" />} loading={runResearch.pending} disabled={anyRunning} onClick={() => void onRun('strategist')}>
              Run strategist now
            </Button>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-[320px_minmax(0,1fr)] xl:grid-cols-[360px_minmax(0,1fr)] gap-4 items-start">
        <Card title="Runs" subtitle={runs.data ? `${runs.data.length} most recent` : undefined} flush>
          {runs.error && !runs.data ? (
            <ErrorState error={runs.error} onRetry={() => void runs.refresh()} />
          ) : runs.loading && !runs.data ? (
            <div className="px-4 pb-4 space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-20 w-full" />
              ))}
            </div>
          ) : runs.data && runs.data.length === 0 ? (
            <EmptyState icon={Bot} title="No research runs yet" description="The first analyst session runs after today's close, budget permitting." compact />
          ) : (
            <ol className="divide-y divide-border/60 max-h-[720px] overflow-y-auto">
              {(runs.data ?? []).map((r) => (
                <li key={r.id}>
                  <RunRow run={r} active={r.id === selectedId} onClick={() => setSelectedId(r.id)} />
                </li>
              ))}
            </ol>
          )}
        </Card>

        <Card
          title={detail.data ? `Run #${detail.data.id} · ${detail.data.kind}` : selectedRun ? `Run #${selectedRun.id}` : 'Run detail'}
          subtitle={detail.data ? `${detail.data.model} · started ${fmtDateTime(detail.data.started_at)}${detail.data.finished_at ? ` · finished ${fmtDateTime(detail.data.finished_at)}` : ''}` : undefined}
          actions={detail.data && <Badge tone={runStatusTone(detail.data.status)} dot pulse={detail.data.status === 'running'}>{detail.data.status}</Badge>}
        >
          {detail.error && !detail.data ? (
            <ErrorState error={detail.error} onRetry={() => void detail.refresh()} />
          ) : !detail.data ? (
            selectedId === null ? (
              <EmptyState icon={Bot} title="Select a run" compact />
            ) : (
              <SkeletonText lines={8} />
            )
          ) : (
            <RunDetail run={detail.data} onDecide={onDecide} deciding={decide.pending} />
          )}
        </Card>
      </div>

      <Card
        title="Proposals"
        subtitle={proposals.data ? `${filteredProposals.length} of ${proposals.data.length}` : undefined}
        actions={<Chips options={PROPOSAL_STATUSES.map((s) => ({ value: s, label: s, count: propCounts[s] }))} value={propStatus} onChange={setPropStatus} allLabel="All" size="xs" />}
      >
        {proposals.error && !proposals.data ? (
          <ErrorState error={proposals.error} onRetry={() => void proposals.refresh()} />
        ) : proposals.loading && !proposals.data ? (
          <div className="space-y-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-24 w-full" />
            ))}
          </div>
        ) : filteredProposals.length === 0 ? (
          <EmptyState title="No proposals" description={propStatus ? `Nothing with status "${propStatus}".` : 'Claude has not proposed anything yet.'} compact />
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
            {filteredProposals.map((p) => (
              <ProposalCard key={p.id} proposal={p} onDecide={onDecide} pending={decide.pending} />
            ))}
          </div>
        )}
      </Card>

      <Card
        title={
          <span className="inline-flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-claude" aria-hidden />
            Lab memory
          </span>
        }
        subtitle={memory.data ? `Claude's running notebook · updated ${fmtRelative(memory.data.updated_at)}` : undefined}
      >
        {memory.error && !memory.data ? (
          <ErrorState error={memory.error} onRetry={() => void memory.refresh()} />
        ) : !memory.data ? (
          <SkeletonText lines={6} />
        ) : (
          <Markdown source={memory.data.memory_md} />
        )}
      </Card>
    </div>
  )
}

function RunRow({ run: r, active, onClick }: { run: ResearchRun; active: boolean; onClick: () => void }) {
  const duration = r.finished_at ? (new Date(r.finished_at).getTime() - new Date(r.started_at).getTime()) / 60_000 : null
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx('w-full text-left px-4 py-3 transition-colors border-l-2', active ? 'bg-accent/8 border-l-accent' : 'border-l-transparent hover:bg-muted/8')}
      aria-current={active ? 'true' : undefined}
    >
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className="text-[11.5px] text-faint font-mono">#{r.id}</span>
        <Badge tone={runKindTone(r.kind)} size="xs">
          {r.kind}
        </Badge>
        <Badge tone="dim" size="xs">
          {r.model}
        </Badge>
        <Badge tone={runStatusTone(r.status)} size="xs" dot pulse={r.status === 'running'}>
          {r.status}
        </Badge>
        <span className="ml-auto text-[11px] text-faint whitespace-nowrap">{fmtRelative(r.started_at)}</span>
      </div>
      <p className="text-[12px] text-text leading-[17px] mt-1.5 line-clamp-2">{r.summary || '—'}</p>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1.5 text-[11px] text-muted">
        <span>
          <span className="text-text font-medium">{fmtUsd(r.cost_usd, { decimals: 2 })}</span>
        </span>
        <span>{fmtTokens(r.input_tokens)} in</span>
        <span>{fmtTokens(r.output_tokens)} out</span>
        <span>{fmtNum(r.num_turns)} turns</span>
        {duration !== null && <span>{fmtDuration(duration)}</span>}
        <span className={clsx(r.n_proposals > 0 ? 'text-text' : '')}>
          {r.n_accepted}/{r.n_proposals} accepted
        </span>
      </div>
    </button>
  )
}

function RunDetail({ run, onDecide, deciding }: { run: ResearchRunDetail; onDecide: (id: number, d: Decision) => Promise<void>; deciding: boolean }) {
  const [showDigest, setShowDigest] = useState(false)
  const [showMemory, setShowMemory] = useState(true)
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 sm:grid-cols-3 2xl:grid-cols-5 gap-2">
        <Mini label="Cost" value={fmtUsd(run.cost_usd, { decimals: 2 })} />
        <Mini label="Input tokens" value={fmtTokens(run.input_tokens)} />
        <Mini label="Output tokens" value={fmtTokens(run.output_tokens)} />
        <Mini label="Turns" value={fmtNum(run.num_turns)} />
        <Mini label="Proposals" value={`${run.n_accepted}/${run.n_proposals}`} sub="accepted" />
      </div>

      {run.error && (
        <div className="rounded-lg border border-loss/30 bg-loss/10 px-3 py-2 text-[12px]">
          <div className="font-medium text-loss">Run error</div>
          <pre className="font-mono text-[11px] text-text whitespace-pre-wrap mt-1">{run.error}</pre>
        </div>
      )}

      {run.status === 'running' ? (
        <EmptyState icon={Sparkles} title="Session in progress" description="The analysis appears here when Claude finishes. This view refreshes every 5 seconds." compact />
      ) : run.status === 'skipped' ? (
        <EmptyState icon={Bot} title="Session skipped" description={run.summary} compact />
      ) : run.analysis_md ? (
        <section>
          <SectionTitle className="mb-2">Analysis</SectionTitle>
          <div className="rounded-lg border border-border bg-panel-2 px-4 py-3">
            <Markdown source={run.analysis_md} />
          </div>
        </section>
      ) : (
        <EmptyState title="No analysis recorded" compact />
      )}

      <section>
        <SectionTitle className="mb-2">Proposals from this run ({run.proposals.length})</SectionTitle>
        {run.proposals.length === 0 ? (
          <p className="text-[12px] text-muted">No proposals.</p>
        ) : (
          <div className="space-y-2">
            {run.proposals.map((p) => (
              <ProposalCard key={p.id} proposal={p} onDecide={onDecide} pending={deciding} compact />
            ))}
          </div>
        )}
      </section>

      {run.memory_update_md && (
        <section>
          <button type="button" className="flex items-center gap-1 text-[11px] font-medium uppercase tracking-wide text-muted hover:text-text" onClick={() => setShowMemory((s) => !s)}>
            {showMemory ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            Memory update
          </button>
          {showMemory && (
            <div className="mt-2 rounded-lg border border-border bg-panel-2 px-4 py-3">
              <Markdown source={run.memory_update_md} />
            </div>
          )}
        </section>
      )}

      {run.digest_md && (
        <section>
          <button type="button" className="flex items-center gap-1 text-[11px] font-medium uppercase tracking-wide text-muted hover:text-text" onClick={() => setShowDigest((s) => !s)}>
            {showDigest ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            Input digest
          </button>
          {showDigest && (
            <div className="mt-2 rounded-lg border border-border bg-panel-2 px-4 py-3">
              <Markdown source={run.digest_md} />
            </div>
          )}
        </section>
      )}
    </div>
  )
}

function Mini({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-border bg-panel-2 px-3 py-2 min-w-0">
      <div className="text-[10.5px] uppercase tracking-wide text-muted truncate">{label}</div>
      <div className="text-[15px] font-semibold text-text mt-0.5 truncate">{value}</div>
      {sub && <div className="text-[10.5px] text-faint">{sub}</div>}
    </div>
  )
}
