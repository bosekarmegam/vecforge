import React from 'react';
import {AbsoluteFill, random, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {Backdrop, Flash, ForgeText, Mark, ramp, Rays, Shockwave, Streak} from '../fx';
import {C, FONT, MONO} from '../theme';

// 55-60 s: particles converge into the mark; install command and links.

const LX = 540;
const LY = 600;

export const EndCard: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const logo = spring({frame: f - 20, fps, config: {damping: 11, stiffness: 120}});
  const cmd = '$ pip install vecforge';
  const typed = cmd.slice(0, Math.floor(ramp(f, [34, 58], [0, cmd.length], (t) => t)));
  const cursor = Math.floor(f / 8) % 2 === 0;
  const sweep = ramp(f, [60, 90], [-0.3, 1.3]);
  return (
    <AbsoluteFill>
      <Backdrop grid={0.55} speed={1.2} />
      <Rays x={LX} y={LY} size={1500} opacity={0.8 * logo} color={C.violet} />
      {/* converging particles */}
      {Array.from({length: 90}).map((_, i) => {
        const a = random(`ea${i}`) * Math.PI * 2;
        const d = 600 + random(`ed${i}`) * 700;
        const p = ramp(f, [0 + random(`et${i}`) * 6, 22], [0, 1], (t) => t * t * t);
        const x = LX + Math.cos(a) * d * (1 - p);
        const y = LY + Math.sin(a) * d * (1 - p);
        const col = [C.cyan, C.violet, C.amber][i % 3];
        if (p >= 1) return null;
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: x - 4,
              top: y - 4,
              width: 8 + 30 * p,
              height: 8,
              borderRadius: 4,
              background: col,
              boxShadow: `0 0 14px ${col}`,
              transform: `rotate(${(a * 180) / Math.PI}deg)`,
              transformOrigin: '4px 4px',
            }}
          />
        );
      })}
      <div style={{position: 'absolute', left: LX - 190, top: LY - 190, transform: `scale(${logo})`, filter: `drop-shadow(0 0 60px ${C.indigo})`}}>
        <Mark size={380} id="end" spark={1 + 0.15 * Math.sin(f / 5)} />
        <div style={{position: 'absolute', left: 14, top: 14, width: 352, height: 352, borderRadius: 84, overflow: 'hidden'}}>
          <div style={{position: 'absolute', top: -150, bottom: -150, width: 140, left: `${sweep * 100}%`, transform: 'rotate(20deg)', background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent)'}} />
        </div>
      </div>
      <div style={{position: 'absolute', top: 860, left: 0, right: 0, textAlign: 'center', opacity: ramp(f, [24, 38], [0, 1]), transform: `translateY(${ramp(f, [24, 38], [40, 0])}px)`, filter: `blur(${ramp(f, [24, 38], [14, 0])}px)`}}>
        <ForgeText size={170}>VecForge</ForgeText>
      </div>
      <div style={{position: 'absolute', top: 1090, left: 0, right: 0, display: 'flex', justifyContent: 'center', opacity: ramp(f, [30, 38], [0, 1])}}>
        <div style={{fontFamily: MONO, fontSize: 46, color: '#fff', padding: '22px 40px', borderRadius: 24, background: 'rgba(5,3,20,0.85)', border: `2px solid ${C.cyan}88`, boxShadow: `0 0 40px ${C.cyan}44`, minWidth: 720}}>
          <span style={{color: C.cyan}}>{typed.slice(0, 1)}</span>
          {typed.slice(1)}
          <span style={{opacity: cursor ? 1 : 0, color: C.cyan}}>▍</span>
        </div>
      </div>
      <div style={{position: 'absolute', top: 1240, left: 0, right: 0, textAlign: 'center', fontFamily: FONT, fontSize: 36, fontWeight: 600, color: C.ink, opacity: ramp(f, [56, 70], [0, 1])}}>
        github.com/bosekarmegam/vecforge
      </div>
      <div style={{position: 'absolute', top: 1330, left: 0, right: 0, textAlign: 'center', fontFamily: FONT, fontSize: 30, fontWeight: 700, letterSpacing: 6, color: C.amber, opacity: ramp(f, [66, 80], [0, 1])}}>
        POWERED BY ARCGX TECHLABS
      </div>
      <Shockwave at={22} x={LX} y={LY} color={C.violet} length={30} />
      <Streak y={LY} at={20} length={28} color={C.violet} />
      <Flash at={21} length={10} peak={0.7} />
    </AbsoluteFill>
  );
};
