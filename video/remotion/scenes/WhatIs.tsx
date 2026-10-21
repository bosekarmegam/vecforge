import React from 'react';
import {AbsoluteFill, random, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {Backdrop, Eyebrow, Flash, ForgeText, Glass, ramp, Shockwave} from '../fx';
import {C, FONT, MONO} from '../theme';

// 10-17.5 s: text becomes a vector, lands in a 3D cloud of meaning, and a
// query with no shared words still finds it.

const CLUSTERS = [
  {name: 'health', color: C.cyan, c: [-150, -60, 60]},
  {name: 'finance', color: C.amber, c: [170, -110, -90]},
  {name: 'travel', color: C.violet, c: [150, 170, 170]},
  {name: 'code', color: C.green, c: [-90, 130, -170]},
] as const;

type P = {x: number; y: number; z: number; k: number};
const POINTS: P[] = Array.from({length: 168}).map((_, i) => {
  const k = i % 4;
  const [cx, cy, cz] = CLUSTERS[k].c;
  const r = 95 * Math.cbrt(random(`r${i}`));
  const th = random(`t${i}`) * Math.PI * 2;
  const ph = Math.acos(2 * random(`p${i}`) - 1);
  return {x: cx + r * Math.sin(ph) * Math.cos(th), y: cy + r * Math.cos(ph), z: cz + r * Math.sin(ph) * Math.sin(th), k};
});

const CX = 540;
const CY = 880;

function project(p: {x: number; y: number; z: number}, a: number) {
  const x = p.x * Math.cos(a) - p.z * Math.sin(a);
  const z = p.x * Math.sin(a) + p.z * Math.cos(a);
  const y = p.y * Math.cos(0.32) - z * Math.sin(0.32);
  const zz = p.y * Math.sin(0.32) + z * Math.cos(0.32);
  const s = 900 / (900 + zz);
  return {x: CX + x * s * 1.3, y: CY + y * s * 1.05, s, z: zz};
}

const DOC = {x: -150, y: -60, z: 60}; // where the new document lands (health cluster)
const QUERY = {x: -95, y: -20, z: 20};
const NEAR = [...POINTS.map((p, i) => ({i, d: Math.hypot(p.x - QUERY.x, p.y - QUERY.y, p.z - QUERY.z)}))]
  .sort((a, b) => a.d - b.d)
  .slice(0, 3)
  .map((n) => n.i);

export const WhatIs: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const angle = -0.45 + f * 0.0045;
  const cloudIn = ramp(f, [34, 70], [0, 1]);
  const card = spring({frame: f - 2, fps, config: {damping: 15}});
  const bars = ramp(f, [18, 40], [0, 1]);
  const fly = ramp(f, [56, 88], [0, 1]);
  const queryIn = ramp(f, [104, 120], [0, 1]);
  const links = ramp(f, [118, 140], [0, 1]);
  const docP = project(DOC, angle);
  const qP = project(QUERY, angle);
  const startX = 540;
  const startY = 480;
  const dx = startX + (docP.x - startX) * fly;
  const dy = startY + (docP.y - startY) * fly - Math.sin(Math.PI * fly) * 120;
  const projected = POINTS.map((p, i) => ({...project(p, angle), k: p.k, i})).sort((a, b) => b.z - a.z);
  return (
    <AbsoluteFill>
      <Backdrop grid={0.35} glow="#1b1350" />
      {/* title */}
      <div style={{position: 'absolute', top: 150, left: 0, right: 0, textAlign: 'center'}}>
        <Eyebrow>What is VecForge?</Eyebrow>
        <div style={{marginTop: 14, opacity: ramp(f, [0, 12], [0, 1]), transform: `translateY(${ramp(f, [0, 12], [30, 0])}px)`}}>
          <span style={{fontFamily: FONT, fontSize: 84, fontWeight: 800, color: '#fff', letterSpacing: '-0.03em'}}>Search by </span>
          <ForgeText size={84}>meaning</ForgeText>
        </div>
      </div>
      {/* the document and its vector */}
      <div style={{position: 'absolute', top: 360, left: 120, right: 120, opacity: card * (1 - fly), transform: `scale(${(0.9 + 0.1 * card) * (1 - 0.3 * fly)}) translateY(${-fly * 80}px)`}}>
        <Glass style={{padding: '24px 30px', textAlign: 'center'}}>
          <div style={{fontFamily: FONT, fontSize: 40, color: '#fff', fontWeight: 500}}>"Patient has a fever and a cough"</div>
          <div style={{display: 'flex', gap: 7, justifyContent: 'center', alignItems: 'flex-end', height: 70, marginTop: 16}}>
            {Array.from({length: 30}).map((_, i) => {
              const v = random(`b${i}`);
              const h = 10 + v * 60 * bars * (0.75 + 0.25 * Math.sin(f / 4 + i));
              const col = v > 0.5 ? C.cyan : C.violet;
              return <div key={i} style={{width: 14, height: h, borderRadius: 4, background: col, boxShadow: `0 0 10px ${col}`, opacity: bars}} />;
            })}
          </div>
          <div style={{fontFamily: MONO, fontSize: 28, color: C.cyan, marginTop: 12, opacity: bars}}>
            {'[0.12, -0.83, 0.44, 0.07, -0.51, ...]'.slice(0, Math.floor(bars * 40))}
          </div>
        </Glass>
      </div>
      {/* the cloud */}
      <svg width={1080} height={1920} style={{position: 'absolute', inset: 0, opacity: cloudIn}}>
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="6" />
          </filter>
        </defs>
        {/* faint axes */}
        {[
          [{x: -260, y: 0, z: 0}, {x: 260, y: 0, z: 0}],
          [{x: 0, y: -260, z: 0}, {x: 0, y: 260, z: 0}],
          [{x: 0, y: 0, z: -260}, {x: 0, y: 0, z: 260}],
        ].map(([a, b], i) => {
          const pa = project(a, angle);
          const pb = project(b, angle);
          return <line key={i} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y} stroke="#fff" strokeOpacity={0.12} strokeWidth={2} />;
        })}
        <g filter="url(#glow)">
          {projected.map((p) => (
            <circle key={`b${p.i}`} cx={p.x} cy={p.y} r={7 * p.s * cloudIn} fill={CLUSTERS[p.k].color} opacity={0.8} />
          ))}
        </g>
        {projected.map((p) => (
          <circle key={p.i} cx={p.x} cy={p.y} r={4.5 * p.s} fill={CLUSTERS[p.k].color} opacity={0.4 + 0.6 * p.s * p.s} />
        ))}
        {/* cluster labels */}
        {CLUSTERS.map((c, i) => {
          const p = project({x: c.c[0], y: c.c[1] - 125, z: c.c[2]}, angle);
          return (
            <text key={c.name} x={p.x} y={p.y} fill={c.color} fontFamily={FONT} fontSize={30} fontWeight={700} textAnchor="middle" opacity={ramp(f, [60 + i * 5, 75 + i * 5], [0, 1])} letterSpacing={3}>
              {c.name.toUpperCase()}
            </text>
          );
        })}
        {/* nearest-neighbour links from the query */}
        {NEAR.map((i, n) => {
          const p = project(POINTS[i], angle);
          const t = ramp(f, [118 + n * 5, 134 + n * 5], [0, 1]);
          return (
            <g key={i}>
              <line x1={qP.x} y1={qP.y} x2={qP.x + (p.x - qP.x) * t} y2={qP.y + (p.y - qP.y) * t} stroke={C.amber} strokeWidth={4} strokeLinecap="round" opacity={0.9} />
              <circle cx={p.x} cy={p.y} r={14 * t * (1 + 0.2 * Math.sin(f / 3))} fill="none" stroke={C.amber} strokeWidth={3} opacity={t} />
            </g>
          );
        })}
        {/* the query */}
        {queryIn > 0 ? (
          <g opacity={queryIn}>
            <circle cx={qP.x} cy={qP.y} r={22 + 6 * Math.sin(f / 4)} fill={C.amber} opacity={0.3} filter="url(#glow)" />
            <circle cx={qP.x} cy={qP.y} r={11} fill="#fff" />
          </g>
        ) : null}
        {/* the new document flying in */}
        {f >= 50 && f < 100 ? (
          <g>
            <circle cx={dx} cy={dy} r={26} fill={C.cyan} opacity={0.5} filter="url(#glow)" />
            <circle cx={dx} cy={dy} r={12} fill="#fff" />
          </g>
        ) : null}
      </svg>
      {f >= 88 ? <Ring x={docP.x} y={docP.y} at={88} /> : null}
      {/* query label */}
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: 1255,
          textAlign: 'center',
          opacity: queryIn,
          transform: `translateY(${(1 - queryIn) * 20}px)`,
        }}
      >
        <span style={{fontFamily: MONO, fontSize: 34, color: C.amber, padding: '14px 26px', borderRadius: 18, background: 'rgba(255,184,107,0.12)', border: `2px solid ${C.amber}77`}}>
          query: "respiratory symptoms"
        </span>
        <div style={{fontFamily: FONT, fontSize: 30, color: C.ink, marginTop: 22, opacity: links}}>no shared words, still the nearest match</div>
      </div>
      {/* the promise */}
      <div style={{position: 'absolute', left: 0, right: 0, top: 1372, display: 'flex', justifyContent: 'center', gap: 22}}>
        {['No cloud', 'No API keys', '100% local'].map((t, i) => {
          const s = spring({frame: f - (160 + i * 9), fps, config: {damping: 10, stiffness: 180}});
          return (
            <div
              key={t}
              style={{
                fontFamily: FONT,
                fontSize: 34,
                fontWeight: 700,
                color: '#fff',
                padding: '16px 28px',
                borderRadius: 999,
                background: `linear-gradient(135deg, ${C.indigo}, #4B3BD1)`,
                boxShadow: `0 0 30px ${C.indigo}`,
                opacity: s,
                transform: `scale(${s}) translateY(${(1 - s) * 40}px)`,
              }}
            >
              {t}
            </div>
          );
        })}
      </div>
      <Flash at={88} color={C.cyan} length={8} peak={0.18} />
    </AbsoluteFill>
  );
};

const Ring: React.FC<{x: number; y: number; at: number}> = ({x, y, at}) => <Shockwave at={at} x={x} y={y} color={C.cyan} length={22} max={220} />;
