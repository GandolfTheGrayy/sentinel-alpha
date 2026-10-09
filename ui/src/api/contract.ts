import type {
  AttributionReport,
  BacktestRequest,
  BacktestResult,
  Bar,
  Budget,
  ConfigTree,
  Decision,
  EngineAction,
  EngineActionResponse,
  EquityPoint,
  EquityRange,
  EventQuery,
  EventRow,
  LabMemory,
  Position,
  Proposal,
  ProposalQuery,
  ResearchRun,
  ResearchRunDetail,
  ResearchRunResponse,
  RunKind,
  StatusResponse,
  StreamEvents,
  Trade,
  TradeQuery,
  Variant,
  VariantDetail,
  VariantStatusChange,
} from './types'

export interface StreamErrorInfo {
  /** Consecutive failed connection attempts. */
  attempt: number
  /** Whether the stream ever connected successfully in this subscription. */
  everConnected: boolean
  /** Milliseconds until the next reconnect attempt. */
  retryInMs: number
}

export type StreamHandlers = {
  [K in keyof StreamEvents]?: (payload: StreamEvents[K]) => void
} & {
  onOpen?: () => void
  onError?: (info: StreamErrorInfo) => void
}

export type Unsubscribe = () => void

/** Every endpoint in docs/API.md, implemented by both the HTTP client and the mock. */
export interface ApiClient {
  getStatus(): Promise<StatusResponse>
  getEquity(range: EquityRange, variant?: string): Promise<EquityPoint[]>
  getPositions(): Promise<Position[]>
  getTrades(q?: TradeQuery): Promise<Trade[]>
  getVariants(): Promise<Variant[]>
  getVariant(id: string): Promise<VariantDetail>
  setVariantStatus(id: string, status: VariantStatusChange): Promise<Variant>
  getAttribution(): Promise<AttributionReport>
  getResearch(limit?: number): Promise<ResearchRun[]>
  getResearchRun(id: number): Promise<ResearchRunDetail>
  getMemory(): Promise<LabMemory>
  getProposals(q?: ProposalQuery): Promise<Proposal[]>
  decideProposal(id: number, decision: Decision): Promise<Proposal>
  getBudget(): Promise<Budget>
  calibrateBudget(observedWeeklyPct: number): Promise<Budget>
  getEvents(q?: EventQuery): Promise<EventRow[]>
  engineAction(action: EngineAction): Promise<EngineActionResponse>
  runResearch(kind: RunKind): Promise<ResearchRunResponse>
  getConfig(): Promise<ConfigTree>
  getBars(symbol: string, timeframe: string, limit?: number): Promise<Bar[]>
  getBacktests(limit?: number): Promise<BacktestResult[]>
  runBacktest(req: BacktestRequest): Promise<BacktestResult>
  /** Subscribe to the live stream. Returns an unsubscribe function. */
  stream(handlers: StreamHandlers): Unsubscribe
}
