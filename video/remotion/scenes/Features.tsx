import React from 'react';
import {AbsoluteFill, random, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {Backdrop, Flash, Glass, ramp} from '../fx';
import {C, FONT, MONO} from '../theme';

// 47.5-55 s: encrypted, fast, private; then everything is fenced inside "your machine".

const HexField: React.FC = () => {
  const f = useCurrentFrame();
  const r = 46;
  const w = Math.sqrt(3) * r;
  const cells = [];
  for (let row = 0; row < 26; row++) {
    for (let col = 0; col < 14; col++) {
      const x = col * w + (row % 2 ? w / 2 : 0) - 20;
      const y = row * r * 1.5 - 20;
      const d = Math.hypot(x - 540, y - 470);
      const wave = Math.max(0, Math.sin(d / 90 - f / 6));
      const o = 0.04 + 0.22 * wave ** 6 * ramp(f, [0, 30], [0, 1]);
      const pts = Array.from({length: 6})
        .map((_, k) => {
          const a = (Math.PI / 3) * k + Math.PI / 6;
          return `${x + Math.cos(a) * (r - 4)},${y + Math.sin(a) * (r - 4)}`;
        })
        .join(' ');
      cells.push(<polygon key={`${row}-${col}`} points={pts} fill="none" stroke={C.cyan} strokeWidth={2} opacity={o} />);
    }
  }
  return (
    <svg width={1080} height={1920} style={{position: 'absolute', inset: 0}}>
      {cells}
    </svg>
  );
};

const SHIELD = 'M190 20 L340 75 V190 C340 290 275 355 190 390 C105 355 40 290 40 190 V75 Z';

const Shield: React.FC = () => {
  const f = useCurrentFrame();
  const draw = ramp(f, [4, 40], [0, 1]);
  const fill = ramp(f, [30, 50], [0, 1]);
  const scan = ((f - 40) % 50) / 50;
  return (
    <svg width={380} height={410} viewBox="0 0 380 410" style={{overflow: 'visible', filter: `drop-shadow(0 0 30px ${C.cyan})`}}>
      <defs>
        <linearGradient id="sh" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={C.indigo} />
          <stop offset="1" stopColor={C.deep} />
        </linearGradient>
        <clipPath id="shc">
          <path d={SHIELD} />
        </clipPath>
      </defs>
      <path d={SHIELD} fill="url(#sh)" opacity={fill} />
      <path d={SHIELD} fill="none" stroke={C.cyan} strokeWidth={8} strokeDasharray={1200} strokeDashoffset={1200 * (1 - draw)} strokeLinejoin="round" />
      {f > 40 ? <rect x={0} y={scan * 410 - 10} width={380} height={20} fill={C.cyan} opacity={0.35} clipPath="url(#shc)" /> : null}
      <g opacity={fill} transform="translate(190 205)">
        <path d="M-40 -18 V-48 A40 40 0 0 1 40 -48 V-18" fill="none" stroke="#fff" strokeWidth={14} />
        <rect x={-62} y={-20} width={124} height={100} rx={18} fill="#fff" />
        <circle cx={0} cy={22} r={13} fill={C.deep} />
        <rect x={-5} y={24} width={10} height={30} rx={4} fill={C.deep} />
      </g>
    </svg>
  );
};

const Row: React.FC<{at: number; top: number; title: string; color: string; children: React.ReactNode}> = ({at, top, title, color, children}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f - at, fps, config: {damping: 14, stiffness: 150}});
  return (
    <div style={{position: 'absolute', left: 90, right: 90, top, opacity: s, transform: `translateX(${(1 - s) * 260}px) skewX(${(1 - s) * -12}deg)`}}>
      <Glass style={{padding: '30px 40px', display: 'flex', alignItems: 'center', gap: 34, borderColor: `${color}66`}}>
        <div style={{fontFamily: FONT, fontSize: 70, fontWeight: 800, color, width: 280, letterSpacing: '-0.03em', textShadow: `0 0 30px ${color}88`}}>{title}</div>
        <div style={{flex: 1}}>{children}</div>
      </Glass>
    </div>
  );
};

const Sub: React.FC<{children: React.ReactNode}> = ({children}) => (
  <div style={{fontFamily: FONT, fontSize: 32, color: C.ink, lineHeight: 1.3}}>{children}</div>
);

export const Features: React.FC = () => {
  const f = useCurrentFrame();
  const ms = ramp(f, [80, 110], [80, 7.3]);
  const fence = ramp(f, [118, 150], [0, 1]);
  const box = {l: 60, t: 140, r: 1020, b: 1260};
  return (
    <AbsoluteFill>
      <Backdrop grid={0.3} glow="#141046" />
      <HexField />
      <div style={{position: 'absolute', left: 0, right: 0, top: 200, display: 'flex', justifyContent: 'center', transform: `scale(${ramp(f, [0, 20], [0.6, 1])})`}}>
        <Shield />
      </div>
      <div style={{position: 'absolute', left: 0, right: 0, top: 630, textAlign: 'center', opacity: ramp(f, [34, 48], [0, 1])}}>
        <span style={{fontFamily: MONO, fontSize: 44, fontWeight: 700, color: C.cyan, letterSpacing: 6}}>AES-256</span>
        <div style={{fontFamily: FONT, fontSize: 32, color: C.ink, marginTop: 8}}>encrypted at rest, indexes included</div>
      </div>
      <Row at={78} top={800} title="Fast" color={C.amber}>
        <div style={{fontFamily: MONO, fontSize: 64, fontWeight: 700, color: '#fff'}}>{ms.toFixed(1)} ms</div>
        <Sub>hybrid search over 100k docs</Sub>
      </Row>
      <Row at={98} top={1040} title="Private" color={C.green}>
        <div style={{fontFamily: MONO, fontSize: 64, fontWeight: 700, color: '#fff'}}>0 bytes</div>
        <Sub>sent to any server</Sub>
      </Row>
      {/* the fence: everything stays on your machine */}
      {fence > 0 ? (
        <>
          <svg width={1080} height={1920} style={{position: 'absolute', inset: 0}}>
            <rect
              x={box.l}
              y={box.t}
              width={box.r - box.l}
              height={box.b - box.t}
              rx={46}
              fill="none"
              stroke={C.violet}
              strokeWidth={5}
              strokeDasharray={4400}
              strokeDashoffset={4400 * (1 - fence)}
              style={{filter: `drop-shadow(0 0 16px ${C.violet})`}}
            />
          </svg>
          <div
            style={{
              position: 'absolute',
              left: 110,
              top: box.t - 26,
              padding: '8px 22px',
              borderRadius: 12,
              background: C.violet,
              color: C.bg0,
              fontFamily: FONT,
              fontWeight: 800,
              fontSize: 30,
              letterSpacing: 5,
              opacity: fence,
            }}
          >
            YOUR MACHINE
          </div>
          {/* data bouncing inside the fence, never leaving it */}
          {Array.from({length: 26}).map((_, i) => {
            const t = (f - 118) * (2 + random(`bv${i}`) * 4);
            const wv = box.r - box.l - 60;
            const hv = box.b - box.t - 60;
            const bx = (random(`bx${i}`) * wv + t * Math.cos(i)) % (2 * wv);
            const by = (random(`by${i}`) * hv + t * Math.sin(i * 1.7)) % (2 * hv);
            const x = box.l + 30 + (Math.abs(bx) > wv ? 2 * wv - Math.abs(bx) : Math.abs(bx));
            const y = box.t + 30 + (Math.abs(by) > hv ? 2 * hv - Math.abs(by) : Math.abs(by));
            const col = [C.cyan, C.violet, C.amber, C.green][i % 4];
            return <div key={i} style={{position: 'absolute', left: x - 6, top: y - 6, width: 12, height: 12, borderRadius: 6, background: col, boxShadow: `0 0 16px ${col}`, opacity: fence}} />;
          })}
        </>
      ) : null}
      <Flash at={78} color={C.amber} length={7} peak={0.12} />
      <Flash at={98} color={C.green} length={7} peak={0.12} />
    </AbsoluteFill>
  );
};
