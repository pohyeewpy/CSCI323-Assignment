// frontend/src/App.jsx
import React, { useMemo, useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend
} from 'recharts'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function classNames(...s) { return s.filter(Boolean).join(' ') }

const Pill = ({ children }) => (
  <span className="inline-flex items-center rounded-full px-3 py-1 text-sm border">{children}</span>
)

const Card = ({ title, right, children }) => (
  <div className="rounded-2xl shadow-sm border bg-white/60 backdrop-blur p-5 min-w-0">
    <div className="flex items-start justify-between mb-3">
      <h3 className="text-base font-semibold tracking-tight">{title}</h3>
      {right}
    </div>
    {children}
  </div>
)

const RatingBars = ({ histogram, total }) => {
  const values = Object.values(histogram || {})
  const max = Math.max(1, ...values)
  return (
    <div className="grid grid-cols-5 gap-3">
      {[1,2,3,4,5].map((star) => {
        const v = histogram?.[String(star)] ?? 0
        const h = Math.round((v / max) * 100)
        return (
          <div key={star} className="flex flex-col items-center">
            <div className="h-28 w-full bg-gray-100 rounded">
              <div className="bg-gray-800 rounded w-full" style={{ height: `${h}%` }} />
            </div>
            <div className="mt-1 text-xs text-gray-600">{star}★</div>
            <div className="text-xs tabular-nums">{v}</div>
          </div>
        )
      })}
      <div className="col-span-5 text-right text-xs text-gray-500">Total: {total}</div>
    </div>
  )
}

// util: strip ```json ... ``` fences if any leak through (defensive)
const stripFences = (s) => {
  if (typeof s !== 'string') return s
  return s.replace(/^```[a-zA-Z]*\s*/,'').replace(/\s*```$/,'').trim()
}

// count total mentions; if an item has a numeric .count use it, otherwise 1
const mentionCount = (arr = []) =>
  arr.reduce((sum, it) => {
    if (it && typeof it === 'object' && typeof it.count === 'number') return sum + it.count
    return sum + 1
  }, 0)

// small UI filter for duplicates and generic tokens if backend ever sends them
const dropDupAndGeneric = (arr=[]) => {
  const generic = new Set(["Food","Great","Good","Pizza","Stars"]);
  const seen = new Set();
  const out = [];
  for (const t of arr) {
    const label = typeof t==='string' ? t : (t.label ?? "");
    if (!label) continue;
    if (generic.has(label)) continue;
    const k = label.toLowerCase();
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(typeof t==='string' ? label : {...t, label});
  }
  return out;
};

// colors for sentiment pie
const PIE_COLORS = ["#10b981","#94a3b8","#ef4444"]; // emerald / slate / red

const pretty = (s) => s.replace(/^([a-z])/, (m)=>m.toUpperCase())

export default function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [summary, setSummary] = useState(null)
  const [ratings, setRatings] = useState(null)
  const [report, setReport] = useState(null)
  const [error, setError] = useState(null)
  const [reportError, setReportError] = useState(null)

  const [places, setPlaces] = useState([])
  const [showSuggest, setShowSuggest] = useState(false)
  const [selectedPlace, setSelectedPlace] = useState(null)

  // fetch places once
  React.useEffect(() => {
    fetch(`${API_BASE}/api/places`)
      .then(r => r.json())
      .then(d => setPlaces(d?.places ?? []))
      .catch(()=>{});
  }, []);

  // filtered suggestions
  const suggestions = React.useMemo(() => {
    const q = (query || "").toLowerCase().trim();
    if (!q) return places.slice(0,8);
    return places.filter(p => p.toLowerCase().includes(q)).slice(0,8);
  }, [query, places]);

  // only allow submit if the query equals a known place (clicked)
  const canSubmit = Boolean(selectedPlace && query === selectedPlace);

  const request = async (place) => {
    if (!place) return
    setLoading(true)
    setError(null)
    setReportError(null)
    setSummary(null)
    setRatings(null)
    setReport(null)

    const urlInsights = `${API_BASE}/api/insights?place=${encodeURIComponent(place)}`
    const urlRatings  = `${API_BASE}/api/ratings?place=${encodeURIComponent(place)}`
    const urlReport   = `${API_BASE}/api/report?place=${encodeURIComponent(place)}`
    console.log('API:', API_BASE, '\ninsights:', urlInsights, '\nratings:', urlRatings, '\nreport:', urlReport)

    try {
      const [insRes, ratRes, repRes] = await Promise.allSettled([
        fetch(urlInsights),
        fetch(urlRatings),
        fetch(urlReport),
      ])

      // ---- insights
      if (insRes.status === 'fulfilled' && insRes.value.ok) {
        const ins = await insRes.value.json()

        const m = ins?.metrics || {}
        const pos = typeof m.pos_pct === 'number' ? m.pos_pct : 0
        const neg = typeof m.neg_pct === 'number' ? m.neg_pct : 0
        const label = ins?.narrative?.overall?.label
          ?? (pos >= 55 ? 'Positive' : (neg > 45 ? 'Negative' : 'Mixed'))

        // prefer rationale from narrative.overall; else exec summary; strip any fences that slipped
        let rationale = ins?.narrative?.overall?.rationale
          ?? ins?.narrative?.executive_summary
          ?? `Sentiment mix — Positive ${Math.round(pos)}% • Neutral ${Math.round(m.neu_pct||0)}% • Negative ${Math.round(neg)}%.`
        rationale = stripFences(rationale)

        setSummary({
          overall: { label, score: Math.round(ins?.narrative?.overall?.score ?? pos), rationale },
          praises: ins?.narrative?.praises ?? [],
          issues: ins?.narrative?.issues ?? [],
          suggestions: ins?.narrative?.suggestions ?? [],
          sampleQuotes: ins?.narrative?.sampleQuotes ?? { positive: [], negative: [] },
          mustTry: ins?.narrative?.mustTry ?? [],
          engine: ins?.engine,
          metrics: ins?.metrics
        })
      } else {
        setError(`Insights error ${insRes.status === 'fulfilled' ? insRes.value.status : 'network'}`)
      }

      // ---- ratings
      if (ratRes.status === 'fulfilled' && ratRes.value.ok) {
        const rat = await ratRes.value.json()
        setRatings(rat)
      } else {
        setError((e) => e ?? `Ratings error ${ratRes.status === 'fulfilled' ? ratRes.value.status : 'network'}`)
      }

      // ---- report (long form) — optional
      if (repRes.status === 'fulfilled' && repRes.value.ok) {
        const rep = await repRes.value.json()
        setReport(rep)
      } else {
        setReportError(`Report error ${repRes.status === 'fulfilled' ? repRes.value.status : 'network'}`)
      }
    } catch (e) {
      setError(e?.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  const scoreColor = useMemo(() => {
    const score = summary?.overall?.score ?? 50
    return score >= 67 ? 'bg-emerald-500' : score >= 40 ? 'bg-amber-500' : 'bg-rose-500'
  }, [summary])

  // ====== chart data (from /api/report) ======
  const starData = useMemo(() => {
    const d = report?.quantitative_metrics?.score_distribution_percent
    if (!d) return []
    return [
      { star: '1★', pct: d['1-star'] ?? d['1'] ?? 0 },
      { star: '2★', pct: d['2-star'] ?? d['2'] ?? 0 },
      { star: '3★', pct: d['3-star'] ?? d['3'] ?? 0 },
      { star: '4★', pct: d['4-star'] ?? d['4'] ?? 0 },
      { star: '5★', pct: d['5-star'] ?? d['5'] ?? 0 },
    ]
  }, [report])

  const sentData = useMemo(() => {
    const s = report?.quantitative_metrics?.sentiment_breakdown_percent
    if (!s) return []
    return [
      { name: 'Positive', value: s.positive ?? 0 },
      { name: 'Neutral',  value: s.neutral ?? 0 },
      { name: 'Negative', value: s.negative ?? 0 },
    ]
  }, [report])

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      <header className="sticky top-0 z-10 backdrop-blur bg-white/70 border-b">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-slate-900 text-white grid place-items-center font-bold">RL</div>
            <div>
              <h1 className="text-lg font-semibold leading-tight">ReviewLens</h1>
              <p className="text-xs text-gray-500 -mt-0.5">Google Maps review insights at a glance</p>
            </div>
          </div>
          <a href="#" className="text-xs text-gray-500 hover:text-gray-700">v1.0</a>
        </div>
      </header>

      <section className="max-w-6xl mx-auto px-4 py-8">
        <div className="rounded-3xl border bg-white/70 shadow-sm p-6 relative">
          <div className="flex flex-col md:flex-row gap-3 items-stretch">
            <div className="relative flex-1">
              <input
                value={query}
                onChange={(e) => { setQuery(e.target.value); setShowSuggest(true); setSelectedPlace(null); }}
                onFocus={() => setShowSuggest(true)}
                placeholder="Start typing a restaurant name (pick from list)…"
                className="w-full rounded-xl border px-4 py-3 focus:outline-none focus:ring-2 focus:ring-slate-500"
              />
              {showSuggest && suggestions.length > 0 && (
                <div className="absolute z-20 mt-1 w-full bg-white border rounded-xl shadow-lg max-h-72 overflow-auto">
                  {suggestions.map((p) => (
                    <button
                      key={p}
                      type="button"
                      className="w-full text-left px-4 py-2 hover:bg-slate-50"
                      onClick={() => {
                        setQuery(p);
                        setSelectedPlace(p);
                        setShowSuggest(false);
                      }}
                    >{p}</button>
                  ))}
                </div>
              )}
            </div>

            <button
              onClick={() => request(selectedPlace)}
              disabled={!canSubmit || loading}
              className={classNames(
                'px-5 py-3 rounded-xl font-medium',
                (!canSubmit || loading) ? 'bg-slate-300 text-slate-600' : 'bg-slate-900 text-white hover:bg-slate-800'
              )}
            >{loading ? 'Analyzing…' : 'Analyze'}</button>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Choose a name from the list (click to fill) — prevents typos from returning no data.
          </p>
        </div>
      </section>

      {error && (
        <section className="max-w-6xl mx-auto px-4">
          <div className="rounded-xl border border-rose-200 bg-rose-50 text-rose-700 p-4">{error}</div>
        </section>
      )}

      {summary && (
        <>
          {/* === GRID A–C === */}
          <main className="max-w-6xl mx-auto px-4 pb-8 grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* a. Overall sentiment */}
            <Card title="Overall sentiment" right={<div className="flex items-center gap-2">
              <Pill>{summary.overall.label} • {summary.overall.score}</Pill>
              {summary.engine && <Pill>{summary.engine}</Pill>}
            </div>}>
              <div className="flex items-center gap-4">
                <div className="h-3 w-full bg-gray-200 rounded-full overflow-hidden">
                  <div className={classNames('h-full', scoreColor)} style={{ width: `${summary.overall.score}%` }} />
                </div>
              </div>
              <p className="text-sm text-gray-600 mt-3">{summary.overall.rationale}</p>
            </Card>

            {/* b. Top praises */}
            <Card title="Top praises" right={<Pill>{mentionCount(summary.praises)} mentions</Pill>}>
              <ul className="space-y-2">
                {dropDupAndGeneric(summary.praises).slice(0,8).map((t, i) => {
                  const label = typeof t==='string'? t : t.label;
                  const count = typeof t==='object'? t.count : undefined;
                  return (
                    <li key={i} className="flex items-start gap-2">
                      <span className="mt-1 h-2 w-2 rounded-full bg-emerald-500" />
                      <div className="text-sm font-medium">
                        {label}{count!=null && <span className="text-xs text-gray-500"> ({count})</span>}
                      </div>
                    </li>
                  )
                })}
              </ul>
            </Card>

            {/* c. Top issues */}
            <Card title="Top issues" right={<Pill>{mentionCount(summary.issues)} mentions</Pill>}>
              <ul className="space-y-2">
                {dropDupAndGeneric(summary.issues).slice(0,8).map((t, i) => {
                  const label = typeof t==='string'? t : t.label;
                  const count = typeof t==='object'? t.count : undefined;
                  return (
                    <li key={i} className="flex items-start gap-2">
                      <span className="mt-1 h-2 w-2 rounded-full bg-rose-500" />
                      <div className="text-sm font-medium">
                        {label}{count!=null && <span className="text-xs text-gray-500"> ({count})</span>}
                      </div>
                    </li>
                  )
                })}
              </ul>
            </Card>
          </main>

          {/* === GRID D–G === */}
          <section className="max-w-6xl mx-auto px-4 pb-8 grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* d. Ratings & volume */}
            <Card title="Ratings & volume">
              <div className="grid grid-cols-3 gap-3 text-sm">
                <div className="rounded-xl border p-3">
                  <div className="text-xs text-gray-500">Avg rating</div>
                  <div className="text-lg font-semibold">
                    {ratings?.avgRating?.toFixed ? ratings.avgRating.toFixed(2) : ratings?.avgRating}★
                  </div>
                </div>
                <div className="rounded-xl border p-3">
                  <div className="text-xs text-gray-500">Reviews</div>
                  <div className="text-lg font-semibold">{ratings?.numReviews ?? report?.quantitative_metrics?.data_sample_size ?? '—'}</div>
                </div>
                <div className="rounded-xl border p-3">
                  <div className="text-xs text-gray-500">% positive</div>
                  <div className="text-lg font-semibold">
                    {report?.quantitative_metrics?.sentiment_breakdown_percent?.positive ?? '—'}%
                  </div>
                </div>
              </div>
            </Card>

            {/* f. Star rating distribution */}
            <Card title="Star rating distribution (percent)">
              <div className="h-56 min-w-0">
                <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                  <BarChart data={starData} barSize={28}>
                    <XAxis dataKey="star" />
                    <YAxis unit="%" />
                    <Tooltip formatter={(v)=>`${v}%`} />
                    <Bar dataKey="pct" radius={[8,8,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* g. Sentiment breakdown */}
            <Card title="Sentiment breakdown">
              <div className="h-72 min-w-0">
                <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                  <PieChart margin={{ top: 8, right: 16, bottom: 8, left: 16 }}>
                    <Pie
                      data={sentData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={105}
                      label
                    >
                      {sentData.map((_, idx) => (
                        <Cell key={idx} fill={PIE_COLORS[idx % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Legend />
                    <Tooltip formatter={(v)=>`${v}%`} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </section>

          {/* === GRID H (Insight Report + Actionable Suggestions) === */}
          {report && (
            <section className="max-w-6xl mx-auto px-4 pb-12">
              <div className="rounded-2xl border bg-white/70 shadow-sm p-6">
                <h2 className="text-lg font-semibold mb-2">Insight Report</h2>
                <p className="text-sm text-gray-700 mb-4">{report?.summary_of_key_insights}</p>

                {/* Owner-style view */}
                <div className="grid md:grid-cols-3 gap-6 mb-8">
                  <div className="rounded-xl border p-4">
                    <div className="text-xs text-gray-500 mb-1">Owner’s perspective</div>
                    <div className="text-sm leading-6">{report?.owner_perspective?.summary}</div>
                  </div>
                  <div className="rounded-xl border p-4">
                    <div className="text-xs text-gray-500 mb-2">Key KPIs</div>
                    <ul className="text-sm space-y-1">
                      <li>Avg rating: <strong>{report?.owner_perspective?.kpis?.avg_rating ?? report?.quantitative_metrics?.overall_rating_observed}★</strong></li>
                      <li>Positive: <strong>{Math.round(report?.owner_perspective?.kpis?.positive_pct ?? report?.quantitative_metrics?.sentiment_breakdown_percent?.positive ?? 0)}%</strong></li>
                      <li>Neutral: <strong>{Math.round(report?.owner_perspective?.kpis?.neutral_pct ?? report?.quantitative_metrics?.sentiment_breakdown_percent?.neutral ?? 0)}%</strong></li>
                      <li>Negative: <strong>{Math.round(report?.owner_perspective?.kpis?.negative_pct ?? report?.quantitative_metrics?.sentiment_breakdown_percent?.negative ?? 0)}%</strong></li>
                      <li>Sample size: <strong>{report?.owner_perspective?.kpis?.sample_size ?? report?.quantitative_metrics?.data_sample_size}</strong></li>
                    </ul>
                  </div>
                  <div className="rounded-xl border p-4">
                    <div className="text-xs text-gray-500 mb-2">Next 14 days — checklist</div>
                    <ul className="list-disc ml-5 text-sm space-y-1">
                      {(report?.owner_perspective?.next_14_days_checklist ?? []).map((x,i)=>(
                        <li key={i}>{x}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Actionable suggestions moved inside */}
                <div className="mt-2 mb-8">
                  <h3 className="font-medium mb-2">Actionable suggestions</h3>
                  <ul className="list-disc ml-5 text-sm space-y-1">
                    {(summary?.suggestions ?? []).map((s,i)=>(
                      <li key={i}>{typeof s==='string'? s : (s.action || s.rationale || JSON.stringify(s))}</li>
                    ))}
                  </ul>
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-medium mb-2">Key diner pain points</h3>
                    <ul className="list-disc ml-5 text-sm space-y-1">
                      {(report?.key_diner_pain_points ?? []).map((x,i)=>(
                        <li key={i}>{x}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h3 className="font-medium mb-2">Strengths & positives</h3>
                    <ul className="list-disc ml-5 text-sm space-y-1">
                      {(report?.strengths_and_positive_aspects ?? []).map((x,i)=>(
                        <li key={i}>{x}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="mt-6">
                  <h3 className="font-medium mb-2">Trends & observations</h3>
                  <ul className="list-disc ml-5 text-sm space-y-1">
                    {(report?.trends_and_observations ?? []).map((x,i)=>(
                      <li key={i}>{x}</li>
                    ))}
                  </ul>
                </div>

                <div className="mt-6">
                  <h3 className="font-medium mb-2">Conclusion</h3>
                  <p className="text-sm text-gray-700">{report?.conclusion}</p>
                </div>
              </div>
            </section>
          )}

          {/* === GRID E (Representative quotes) === */}
          <section className="max-w-6xl mx-auto px-4 pb-12 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-3">
              <Card title="Representative quotes">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <div className="text-xs uppercase tracking-wide text-emerald-600 mb-1">Positive</div>
                    <ul className="space-y-2 text-sm">
                      {summary.sampleQuotes.positive.slice(0,3).map((q,i) => (
                        <li key={i} className="p-3 bg-emerald-50 border border-emerald-100 rounded-xl italic">“{q}”</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-wide text-rose-600 mb-1">Negative</div>
                    <ul className="space-y-2 text-sm">
                      {summary.sampleQuotes.negative.slice(0,3).map((q,i) => (
                        <li key={i} className="p-3 bg-rose-50 border border-rose-100 rounded-xl italic">“{q}”</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </Card>
            </div>
          </section>
        </>
      )}

      <footer className="max-w-6xl mx-auto px-4 pb-10 text-xs text-gray-500">
        Demo encompases only a few singapore restaurants reviews.
      </footer>
    </div>
  )
}
