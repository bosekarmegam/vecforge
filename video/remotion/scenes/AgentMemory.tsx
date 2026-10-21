import React from 'react';
import {AbsoluteFill, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {Backdrop, Eyebrow, Flash, ForgeText, Mark, ramp, Rays, Shockwave} from '../fx';
import {C, FONT, MONO} from '../theme';

// 40-47.5 s: agents talk to the VecForge MCP server: one remembers, another recalls.

const HX = 540;
const HY = 800;
const AGENTS = [
  {label: 'Chat assistant', glyph: 'chat', a: -135, color: C.cyan},
  {label: 'Coding agent', glyph: 'code', a: -45, color: C.amber},
  {label: 'Your script', glyph: 'term', a: 135, color: C.green},
  {label: 'Desktop app', glyph: 'win', a: 45, color: C.violet},
] as const;

const pos = (deg: number, f: number) => {
  const a = ((deg + f * 0.08) * Math.PI) / 180;
  return {x: HX + Math.cos(a) * 340, y: HY + Math.sin(a) * 340};
};

const Glyph: React.FC<{kind: string; color: string}> = ({kind, color}) => {
  const s = {stroke: color, strokeWidth: 6, fill: 'none', strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const};
  return (
    <svg width={70} height={70} viewBox="0 0 70 70">
      {kind === 'chat' && <path d="M12 16 h46 a6 6 0 0 1 6 6 v22 a6 6 0 0 1 -6 6 h-26 l-12 10 v-10 h-8 a6 6 0 0 1 -6 -6 v-22 a6 6 0 0 1 6 -6z" {...s} />}
      {kind === 'code' && <path d="M24 20 L10 35 L24 50 M46 20 L60 35 L46 50 M39 14 L31 56" {...s} />}
      {kind === 'term' && <path d="M12 22 L26 35 L12 48 M32 50 H58" {...s} />}
      {kind === 'win' && <path d="M8 14 h54 v42 h-54z M8 26 h54" {...s} />}
    </svg>
  );
};

/** A glowing packet travelling from a to b over [t0, t1]. */
const Packet: React.FC<{from: {x: number; y: number}; to: {x: number; y: number}; t0: number; t1: number; color: string}> = ({from, to, t0, t1, color}) => {
  const f = useCurrentFrame();
  if (f < t0 || f > t1) return null;
  const p = ramp(f, [t0, t1], [0, 1]);
  const trail = [0, 0.04, 0.08, 0.12, 0.16];
  return (
    <>
      {trail.map((d, i) => {
        const q = Math.max(0, p - d);
        const x = from.x + (to.x - from.x) * q;
        const y = from.y + (to.y - from.y) * q;
        const r = 18 - i * 3;
        return (
          <div
            key={i}
            style={{position: 'absolute', left: x - r, top: y - r, width: r * 2, height: r * 2, borderRadius: '50%', background: i === 0 ? '#fff' : color, opacity: 1 - i * 0.18, boxShadow: `0 0 30px ${color}`}}
          />
        );
      })}
    </>
  );
};

const Chip: React.FC<{at: number; out: number; color: string; children: React.ReactNode; top: number}> = ({at, out, color, children, top}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f - at, fps, config: {damping: 13, stiffness: 170}});
  const o = s * ramp(f, [out, out + 10], [1, 0]);
  return (
    <div style={{position: 'absolute', left: 0, right: 0, top, display: 'flex', justifyContent: 'center', opacity: o, transform: `translateY(${(1 - s) * 30}px) scale(${0.9 + 0.1 * s})`}}>
      <div style={{fontFamily: MONO, fontSize: 36, color: '#fff', padding: '18px 30px', borderRadius: 20, background: 'rgba(10,6,34,0.85)', border: `2px solid ${color}`, boxShadow: `0 0 34px ${color}66`}}>{children}</div>
    </div>
  );
};

export const AgentMemory: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const hub = spring({frame: f - 2, fps, config: {damping: 12, stiffness: 110}});
  const ps = AGENTS.map((a) => pos(a.a, f));
  const hubPt = {x: HX, y: HY};
  const memories = (f >= 108 ? 1 : 0) + 3;
  return (
    <AbsoluteFill>
      <Backdrop grid={0.45} glow="#1e1456" speed={0.8} />
      <div style={{position: 'absolute', top: 150, left: 0, right: 0, textAlign: 'center', opacity: ramp(f, [0, 12], [0, 1])}}>
        <Eyebrow color={C.amber}>Built-in MCP server</Eyebrow>
        <div style={{marginTop: 14}}>
          <span style={{fontFamily: FONT, fontSize: 80, fontWeight: 800, color: '#fff', letterSpacing: '-0.03em'}}>Memory for </span>
          <ForgeText size={80}>AI agents</ForgeText>
        </div>
      </div>
      <Rays x={HX} y={HY} size={1200} opacity={0.5 * hub} color={C.indigo} />
      {/* beams */}
      <svg width={1080} height={1920} style={{position: 'absolute', inset: 0}}>
        {ps.map((p, i) => {
          const t = ramp(f, [8 + i * 5, 26 + i * 5], [0, 1]);
          return (
            <g key={i}>
              <line x1={HX} y1={HY} x2={HX + (p.x - HX) * t} y2={HY + (p.y - HY) * t} stroke={AGENTS[i].color} strokeOpacity={0.25} strokeWidth={10} />
              <line
                x1={HX}
                y1={HY}
                x2={HX + (p.x - HX) * t}
                y2={HY + (p.y - HY) * t}
                stroke={AGENTS[i].color}
                strokeWidth={4}
                strokeDasharray="14 18"
                strokeDashoffset={-f * 3}
              />
            </g>
          );
        })}
        {/* halo rings */}
        {[0, 1, 2].map((k) => {
          const p = ((f + k * 20) % 60) / 60;
          return <circle key={k} cx={HX} cy={HY} r={150 + p * 160} fill="none" stroke={C.violet} strokeWidth={3} opacity={(1 - p) * 0.5 * hub} />;
        })}
      </svg>
      {/* agents */}
      {AGENTS.map((a, i) => {
        const s = spring({frame: f - (10 + i * 5), fps, config: {damping: 12}});
        const p = ps[i];
        const active = (i === 0 && f > 56 && f < 100) || (i === 1 && ((f > 128 && f < 160) || (f > 176 && f < 210)));
        return (
          <div key={a.label} style={{position: 'absolute', left: p.x - 90, top: p.y - 90, width: 180, textAlign: 'center', transform: `scale(${s * (active ? 1.12 : 1)})`}}>
            <div
              style={{
                width: 140,
                height: 140,
                margin: '0 auto',
                borderRadius: 40,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'rgba(12,8,40,0.9)',
                border: `3px solid ${a.color}`,
                boxShadow: `0 0 ${active ? 60 : 26}px ${a.color}${active ? 'cc' : '66'}`,
              }}
            >
              <Glyph kind={a.glyph} color={a.color} />
            </div>
            <div style={{fontFamily: FONT, fontSize: 28, fontWeight: 700, color: '#fff', marginTop: 12, whiteSpace: 'nowrap', marginLeft: -40, marginRight: -40}}>{a.label}</div>
          </div>
        );
      })}
      {/* hub */}
      <div style={{position: 'absolute', left: HX - 130, top: HY - 130, transform: `scale(${hub * (1 + (f > 106 && f < 116 ? 0.08 : 0))})`, filter: `drop-shadow(0 0 50px ${C.indigo})`}}>
        <Mark size={260} id="hub" spark={1 + 0.2 * Math.sin(f / 4)} />
      </div>
      {/* memory counter badge on the hub */}
      <div
        style={{
          position: 'absolute',
          left: HX + 70,
          top: HY - 150,
          minWidth: 64,
          height: 64,
          padding: '0 16px',
          borderRadius: 32,
          background: C.amber,
          color: C.bg0,
          fontFamily: FONT,
          fontWeight: 800,
          fontSize: 36,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: `0 0 30px ${C.amber}`,
          transform: `scale(${hub * (f >= 106 && f < 116 ? 1.3 : 1)})`,
        }}
      >
        {memories}
      </div>
      <Packet from={ps[0]} to={hubPt} t0={70} t1={106} color={C.cyan} />
      <Packet from={ps[1]} to={hubPt} t0={140} t1={166} color={C.amber} />
      <Packet from={hubPt} to={ps[1]} t0={168} t1={194} color={C.green} />
      <Shockwave at={106} x={HX} y={HY} color={C.cyan} length={24} max={420} />
      <Shockwave at={166} x={HX} y={HY} color={C.amber} length={24} max={420} />
      <Chip at={58} out={128} color={C.cyan} top={1230}>
        <span style={{color: C.cyan}}>remember</span>("Deploys happen on Fridays")
      </Chip>
      <Chip at={132} out={500} color={C.amber} top={1200}>
        <span style={{color: C.amber}}>recall</span>("when do we deploy?")
      </Chip>
      <Chip at={186} out={500} color={C.green} top={1310}>
        <span style={{color: C.green}}>→</span> Deploys happen on Fridays
      </Chip>
      <Flash at={106} color={C.cyan} length={8} peak={0.15} />
    </AbsoluteFill>
  );
};
