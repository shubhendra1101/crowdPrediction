"use client";

import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API || "http://localhost:8000";
const WS = API.replace(/^http/, "ws");
const LEVEL_COLOR = { Low: "var(--low)", Medium: "var(--medium)", High: "var(--high)", Critical: "var(--critical)" };
const SPEEDS = [1, 5, 10, 30, 60];

function fmt(t) {
  const s = Math.max(0, Math.round(t));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function ZoneView({ layout, msg, selected, onSelect }) {
  if (!layout) return <p className="muted">Loading layout…</p>;
  const zmap = Object.fromEntries((msg?.zones || []).map((z) => [z.zone_id, z]));
  const fill = (id) => LEVEL_COLOR[zmap[id]?.level] || "var(--low)";
  const label = (id) => (zmap[id]?.density == null ? "–" : zmap[id].density.toFixed(1));
  if (layout.kind === "cctv") {
    return (
      <svg viewBox={`0 0 ${layout.width} ${layout.height}`} style={{ width: "100%", height: "auto", display: "block", borderRadius: 8 }}>
        <image href={`${API}${layout.image}`} x="0" y="0" width={layout.width} height={layout.height} />
        {layout.zones.map((z) => {
          if (z.points.length < 3) return null;
          const pts = z.points.map((p) => p.join(",")).join(" ");
          const c = z.points.reduce((a, p) => [a[0] + p[0] / z.points.length, a[1] + p[1] / z.points.length], [0, 0]);
          return (
            <g key={z.zone_id} onClick={() => onSelect(z.zone_id)} style={{ cursor: "pointer" }}>
              <polygon points={pts} fill={fill(z.zone_id)} fillOpacity={selected === z.zone_id ? 0.55 : 0.35}
                       stroke={selected === z.zone_id ? "#fff" : fill(z.zone_id)} strokeWidth={selected === z.zone_id ? 6 : 3} />
              <text x={c[0]} y={c[1]} fill="#fff" fontSize="34" fontWeight="700" textAnchor="middle" stroke="#000" strokeWidth="1.5">
                {label(z.zone_id)}
              </text>
            </g>
          );
        })}
      </svg>
    );
  }
  const W = layout.nx * 60, H = layout.ny * 60;
  return (
    <svg viewBox={`-4 -4 ${W + 8} ${H + 8}`} style={{ width: "100%", maxHeight: 520, display: "block" }}>
      {layout.zones.map((z) => (
        <g key={z.zone_id} onClick={() => onSelect(z.zone_id)} style={{ cursor: "pointer" }}>
          <rect x={z.ix * 60} y={H - (z.iy + 1) * 60} width="58" height="58" rx="4" fill={fill(z.zone_id)}
                fillOpacity={0.25 + Math.min(0.65, (zmap[z.zone_id]?.density || 0) / 8)}
                stroke={selected === z.zone_id ? "var(--text)" : "none"} strokeWidth="3" />
          <text x={z.ix * 60 + 29} y={H - (z.iy + 1) * 60 + 35} textAnchor="middle" fontSize="15" fontWeight="600" fill="var(--text)">
            {label(z.zone_id)}
          </text>
        </g>
      ))}
    </svg>
  );
}

function ForecastChart({ history, msg, zone }) {
  const W = 520, H = 220, L = 34, B = 22;
  const z = msg?.zones.find((x) => x.zone_id === zone);
  if (!z) return <p className="muted">Click a zone to see its forecast.</p>;
  const now = msg.timestamp;
  const past = history.filter((p) => p.t >= now - 120);
  const f = z.forecast;
  const tMax = now + (f ? Math.max(...f.h) : 60);
  const tMin = now - 120;
  const yMax = Math.max(6, ...past.map((p) => p.d || 0), ...(f ? f.p90.map((v) => v || 0) : [0])) * 1.05;
  const X = (t) => L + ((t - tMin) / (tMax - tMin)) * (W - L - 6);
  const Y = (v) => H - B - (v / yMax) * (H - B - 8);
  const line = past.filter((p) => p.d != null).map((p) => `${X(p.t)},${Y(p.d)}`).join(" ");
  const band = f ? [[now, z.density ?? 0], ...f.h.map((h, i) => [now + h, f.p90[i] ?? 0])]
    .map(([t, v]) => `${X(t)},${Y(v)}`).concat([...f.h].reverse().map((h, j) => {
      const i = f.h.length - 1 - j; return `${X(now + h)},${Y(f.p10[i] ?? 0)}`;
    })).concat([`${X(now)},${Y(z.density ?? 0)}`]).join(" ") : "";
  const mid = f ? [[now, z.density ?? 0], ...f.h.map((h, i) => [now + h, f.p50[i] ?? 0])].map(([t, v]) => `${X(t)},${Y(v)}`).join(" ") : "";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
      {[[2, "var(--medium)"], [4, "var(--high)"], [5, "var(--critical)"]].map(([v, c]) => v < yMax && (
        <g key={v}><line x1={L} x2={W - 6} y1={Y(v)} y2={Y(v)} stroke={c} strokeDasharray="4 4" />
          <text x={L - 4} y={Y(v) + 4} fontSize="10" textAnchor="end" fill={c}>{v}</text></g>))}
      <line x1={X(now)} x2={X(now)} y1={8} y2={H - B} stroke="var(--muted)" />
      <text x={X(now)} y={H - 6} fontSize="10" textAnchor="middle" fill="var(--muted)">now</text>
      <text x={L} y={H - 6} fontSize="10" fill="var(--muted)">−2 min</text>
      <text x={W - 6} y={H - 6} fontSize="10" textAnchor="end" fill="var(--muted)">+{f ? Math.max(...f.h) : 0} s</text>
      {band && <polygon points={band} fill="var(--accent)" fillOpacity="0.18" />}
      {mid && <polyline points={mid} fill="none" stroke="var(--accent)" strokeWidth="2" strokeDasharray="5 3" />}
      <polyline points={line} fill="none" stroke="var(--text)" strokeWidth="2" />
    </svg>
  );
}

function TimeBar({ summary, t, onSeek }) {
  if (!summary) return null;
  const W = 1000, H = 46, { start, end, times, peaks, alerts } = summary;
  const X = (x) => ((x - start) / Math.max(1, end - start)) * W;
  const yMax = Math.max(5, ...peaks);
  const pts = times.map((x, i) => `${X(x)},${H - 6 - (peaks[i] / yMax) * (H - 12)}`).join(" ");
  return (
    <svg className="timebar" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 46 }}
         onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); onSeek(start + ((e.clientX - r.left) / r.width) * (end - start)); }}>
      <rect x="0" y="0" width={W} height={H} fill="var(--bg)" />
      <polyline points={pts} fill="none" stroke="var(--muted)" strokeWidth="1.5" />
      {alerts.map((a, i) => <line key={i} x1={X(a.since)} x2={X(a.since)} y1="0" y2={H} stroke="var(--critical)" strokeWidth="2" />)}
      {t != null && <line x1={X(t)} x2={X(t)} y1="0" y2={H} stroke="var(--accent)" strokeWidth="3" />}
    </svg>
  );
}

export default function Page() {
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState(null);
  const [layout, setLayout] = useState(null);
  const [summary, setSummary] = useState(null);
  const [msg, setMsg] = useState(null);
  const [history, setHistory] = useState({});
  const [zone, setZone] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(10);
  const [error, setError] = useState(null);
  const ws = useRef(null);

  useEffect(() => {
    fetch(`${API}/api/runs`).then((r) => r.json()).then((rs) => { setRuns(rs); if (rs.length) setRun(rs[0].id); })
      .catch(() => setError(`Cannot reach the backend at ${API}. Start it with: uvicorn crowdsafe.server.app:app --port 8000`));
  }, []);

  useEffect(() => {
    if (!run) return;
    stop(); setMsg(null); setHistory({}); setZone(null); setLayout(null); setSummary(null);
    fetch(`${API}/api/runs/${run}/layout`).then((r) => r.json()).then(setLayout);
    fetch(`${API}/api/runs/${run}/summary`).then((r) => r.json()).then((s) => { setSummary(s); play(s.start); });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run]);

  function stop() { if (ws.current) { ws.current.onmessage = null; ws.current.close(); ws.current = null; } setPlaying(false); }

  function play(from, spd = speed) {
    stop();
    const sock = new WebSocket(`${WS}/ws/replay/${run}?speed=${spd}&start=${from ?? 0}`);
    sock.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.done || m.error) { setPlaying(false); if (m.error) setError(m.error); return; }
      setMsg(m);
      setHistory((h) => {
        const n = { ...h };
        for (const z of m.zones) n[z.zone_id] = [...(n[z.zone_id] || []).filter((p) => p.t < m.timestamp).slice(-400), { t: m.timestamp, d: z.density }];
        return n;
      });
      setZone((cur) => cur || (m.zones.reduce((a, b) => ((b.density || 0) > (a.density || 0) ? b : a), m.zones[0]) || {}).zone_id);
    };
    ws.current = sock; setPlaying(true);
  }

  const seek = (t) => { setHistory({}); play(t); };
  const title = runs.find((r) => r.id === run)?.title;
  const peak = msg ? Math.max(...msg.zones.map((z) => z.density || 0)) : null;
  const worst = msg ? msg.zones.reduce((a, z) => (["Low", "Medium", "High", "Critical"].indexOf(z.level) > ["Low", "Medium", "High", "Critical"].indexOf(a) ? z.level : a), "Low") : "Low";

  return (
    <>
      <header>
        <h1>CrowdSafe — replay</h1>
        <select value={run || ""} onChange={(e) => setRun(e.target.value)}>
          {runs.map((r) => <option key={r.id} value={r.id}>{r.title}</option>)}
        </select>
        <button className="primary" onClick={() => (playing ? stop() : play(msg?.timestamp ?? summary?.start))}>{playing ? "Pause" : "Play"}</button>
        <label className="muted">speed&nbsp;
          <select value={speed} onChange={(e) => { const s = Number(e.target.value); setSpeed(s); if (playing) play(msg?.timestamp, s); }}>
            {SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}
          </select>
        </label>
        <span className="muted">{msg ? `t = ${fmt(msg.timestamp)} / ${fmt(summary?.end || 0)}` : ""}</span>
      </header>
      {error && <p style={{ padding: "8px 16px", color: "var(--critical)" }}>{error}</p>}
      <div style={{ padding: "8px 16px 0" }}><TimeBar summary={summary} t={msg?.timestamp} onSeek={seek} /></div>
      <main>
        <section className="panel">
          <h2>{title || "Zones"}</h2>
          <ZoneView layout={layout} msg={msg} selected={zone} onSelect={setZone} />
          <div className="legend">
            <span style={{ "--c": "var(--low)" }}>Low &lt; 2</span><span style={{ "--c": "var(--medium)" }}>Medium 2–4</span>
            <span style={{ "--c": "var(--high)" }}>High 4–5</span><span style={{ "--c": "var(--critical)" }}>Critical ≥ 5 persons/m²</span>
            <span>numbers = persons/m² now · colour = level from the forecast worst case</span>
          </div>
        </section>
        <section style={{ display: "grid", gap: 16, alignContent: "start" }}>
          <div className="panel">
            <h2>Now</h2>
            <div className="stat">
              <div><b>{peak == null ? "–" : peak.toFixed(1)}</b>busiest zone (persons/m²)</div>
              <div><b style={{ color: LEVEL_COLOR[worst] }}>{worst}</b>highest level</div>
              <div><b>{msg?.alerts.length ?? 0}</b>active alerts</div>
            </div>
          </div>
          <div className="panel">
            <h2>Forecast — zone {zone || "–"}</h2>
            <ForecastChart history={history[zone] || []} msg={msg} zone={zone} />
            <p className="muted">Black: measured density · blue band: forecast 10–90 % · dashed: median. Dashed lines: 2 / 4 / 5 persons/m².</p>
          </div>
          <div className="panel">
            <h2>Alerts</h2>
            {msg?.alerts.length ? msg.alerts.map((a, i) => (
              <div key={i} className={`alert ${a.level}`}>
                <b>{a.level}</b> in zone <b>{a.zone_id}</b> since {fmt(a.since)}
                {a.time_to_critical_s != null && <> · critical in ≤ {a.time_to_critical_s} s</>}
                <div className="muted">{a.reason}</div>
              </div>)) : <p className="muted">No active alerts.</p>}
            {summary && <p className="muted">This run: {summary.alerts.length} alert(s) in total (red marks on the time bar). Peak density {summary.peak_density} persons/m².</p>}
          </div>
        </section>
      </main>
    </>
  );
}
