import clsx from 'clsx'
import { ArrowDown, ArrowUp, ArrowUpDown } from 'lucide-react'
import { useMemo, useState, type ReactNode } from 'react'
import { EmptyState } from './EmptyState'
import { SkeletonRows } from './Skeleton'

export interface Column<T> {
  key: string
  header: ReactNode
  render: (row: T) => ReactNode
  /** Value used for sorting. If omitted, the column is not sortable. */
  sortValue?: (row: T) => number | string | null | undefined
  align?: 'left' | 'right' | 'center'
  className?: string
  headerClassName?: string
  width?: number | string
  /** Hide below this breakpoint (adds `hidden md:table-cell` etc.). */
  hideBelow?: 'sm' | 'md' | 'lg' | 'xl' | '2xl'
}

export interface SortState {
  key: string
  dir: 'asc' | 'desc'
}

interface DataTableProps<T> {
  columns: Column<T>[]
  rows: T[] | null | undefined
  rowKey: (row: T) => string | number
  loading?: boolean
  onRowClick?: (row: T) => void
  activeKey?: string | number | null
  defaultSort?: SortState
  sort?: SortState | null
  onSortChange?: (s: SortState | null) => void
  empty?: ReactNode
  emptyDescription?: ReactNode
  dense?: boolean
  maxHeight?: number | string
  className?: string
  skeletonRows?: number
  rowClassName?: (row: T) => string | undefined
}

const HIDE: Record<NonNullable<Column<unknown>['hideBelow']>, string> = {
  sm: 'hidden sm:table-cell',
  md: 'hidden md:table-cell',
  lg: 'hidden lg:table-cell',
  xl: 'hidden xl:table-cell',
  '2xl': 'hidden 2xl:table-cell',
}

export function DataTable<T>({
  columns,
  rows,
  rowKey,
  loading,
  onRowClick,
  activeKey,
  defaultSort,
  sort: sortProp,
  onSortChange,
  empty = 'Nothing here yet',
  emptyDescription,
  dense,
  maxHeight,
  className,
  skeletonRows = 6,
  rowClassName,
}: DataTableProps<T>) {
  const [sortState, setSortState] = useState<SortState | null>(defaultSort ?? null)
  const sort = sortProp !== undefined ? sortProp : sortState
  const setSort = (s: SortState | null) => {
    setSortState(s)
    onSortChange?.(s)
  }

  const sorted = useMemo(() => {
    if (!rows) return []
    if (!sort) return rows
    const col = columns.find((c) => c.key === sort.key)
    if (!col?.sortValue) return rows
    const sv = col.sortValue
    const dir = sort.dir === 'asc' ? 1 : -1
    return [...rows].sort((a, b) => {
      const va = sv(a)
      const vb = sv(b)
      if (va === vb) return 0
      if (va === null || va === undefined) return 1
      if (vb === null || vb === undefined) return -1
      if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * dir
      return String(va).localeCompare(String(vb)) * dir
    })
  }, [rows, sort, columns])

  const toggleSort = (col: Column<T>) => {
    if (!col.sortValue) return
    if (sort?.key === col.key) {
      setSort(sort.dir === 'desc' ? { key: col.key, dir: 'asc' } : null)
    } else {
      setSort({ key: col.key, dir: 'desc' })
    }
  }

  const showSkeleton = loading && (!rows || rows.length === 0)

  return (
    <div className={clsx('overflow-auto', className)} style={{ maxHeight }}>
      <table className={clsx('table-base', dense && '[&_tbody_td]:py-1.5')}>
        <thead>
          <tr>
            {columns.map((c) => {
              const sortable = !!c.sortValue
              const active = sort?.key === c.key
              return (
                <th
                  key={c.key}
                  style={{ width: c.width }}
                  className={clsx(
                    c.align === 'right' && 'num',
                    c.align === 'center' && 'text-center',
                    c.hideBelow && HIDE[c.hideBelow],
                    sortable && 'cursor-pointer select-none hover:text-text',
                    c.headerClassName,
                  )}
                  onClick={sortable ? () => toggleSort(c) : undefined}
                  aria-sort={active ? (sort?.dir === 'asc' ? 'ascending' : 'descending') : undefined}
                >
                  <span className={clsx('inline-flex items-center gap-1', c.align === 'right' && 'flex-row-reverse')}>
                    {c.header}
                    {sortable &&
                      (active ? (
                        sort?.dir === 'asc' ? (
                          <ArrowUp className="h-3 w-3 text-accent" />
                        ) : (
                          <ArrowDown className="h-3 w-3 text-accent" />
                        )
                      ) : (
                        <ArrowUpDown className="h-3 w-3 opacity-40" />
                      ))}
                  </span>
                </th>
              )
            })}
          </tr>
        </thead>
        <tbody>
          {!showSkeleton &&
            sorted.map((row) => {
              const k = rowKey(row)
              return (
                <tr
                  key={k}
                  className={clsx(onRowClick && 'row-hover cursor-pointer', activeKey !== undefined && activeKey === k && 'row-active', rowClassName?.(row))}
                  onClick={onRowClick ? () => onRowClick(row) : undefined}
                  tabIndex={onRowClick ? 0 : undefined}
                  onKeyDown={
                    onRowClick
                      ? (e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault()
                            onRowClick(row)
                          }
                        }
                      : undefined
                  }
                >
                  {columns.map((c) => (
                    <td key={c.key} className={clsx(c.align === 'right' && 'num', c.align === 'center' && 'text-center', c.hideBelow && HIDE[c.hideBelow], c.className)}>
                      {c.render(row)}
                    </td>
                  ))}
                </tr>
              )
            })}
        </tbody>
      </table>
      {showSkeleton && <SkeletonRows rows={skeletonRows} cols={Math.min(columns.length, 6)} />}
      {!loading && rows && rows.length === 0 && <EmptyState title={empty} description={emptyDescription} compact />}
    </div>
  )
}
