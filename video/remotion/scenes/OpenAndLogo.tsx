import React from 'react';
import {AbsoluteFill, random, Sequence, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {Backdrop, Bokeh, Eyebrow, Flash, ForgeText, GlitchText, Glass, Mark, ramp, Rays, Shockwave, Streak, easeIn} from '../fx';
import {C, FONT} from '../theme';

// 0-5 s: a chat that forgets. 5-10 s: the VecForge mark is forged.
export const OpenAndLogo: React.FC = () => (
  <AbsoluteFill>
    <Sequence durationInFrames={150}>
      <ColdOpen />
    </Sequence>
    <Sequence from={150} durationInFrames={150}>
      <LogoReveal />
    </Sequence>
  </AbsoluteFill>
);

const Bubble: React.FC<{me?: boolean; text: string; at: number; tone?: 'bad'}> = ({me, text, at, tone}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f - at, fps, config: {damping: 14, stiffness: 160}});
  const bg = me
    ? `linear-gradient(135deg, ${C.indigo}, #4B3BD1)`
    : tone === 'bad'
      ? 'linear-gradient(135deg, rgba(255,122,154,0.22), rgba(255,122,154,0.08))'
      : 'rgba(255,255,255,0.09)';
  return (
    <div style={{display: 'flex', justifyContent: me ? 'flex-end' : 'flex-start', margin: '16px 0'}}>
      <div
        style={{
          maxWidth: 640,
          padding: '26px 34px',
          borderRadius: 34,
          borderBottomRightRadius: me ? 10 : 34,
          borderBottomLeftRadius: me ? 34 : 10,
          background: bg,
          border: tone === 'bad' ? `2px solid ${C.rose}88` : '2px solid rgba(255,255,255,0.08)',
          color: '#fff',
          fontFamily: FONT,
          fontSize: 38,
          fontWeight: 500,
          lineHeight: 1.3,
          opacity: s,
          transform: `translateY(${(1 - s) * 40}px) scale(${0.85 + 0.15 * s})`,
          transformOrigin: me ? '100% 100%' : '0% 100%',
        }}
      >
        {text}
      </div>
    </div>
  );
};

/** Dots that burst out of the first chat when it is wiped. */
const Disintegrate: React.FC<{at: number}> = ({at}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0 || t > 70) return null;
  const cols = 26;
  const rows = 12;
  return (
    <AbsoluteFill style={{mixBlendMode: 'screen'}}>
      {Array.from({length: cols * rows}).map((_, i) => {
        const cx = i % cols;
        const cy = Math.floor(i / cols);
        const x0 = 130 + (cx / (cols - 1)) * 820;
        const y0 = 360 + (cy / (rows - 1)) * 420;
        const delay = (cx / cols) * 14 + random(`d${i}`) * 6;
        const tt = Math.max(0, t - delay);
        const vx = (random(`vx${i}`) - 0.3) * 26;
        const vy = (random(`vy${i}`) - 0.8) * 18;
        const x = x0 + vx * tt + tt * tt * 0.25;
        const y = y0 + vy * tt;
        const o = tt === 0 ? 0 : Math.max(0, 1 - tt / 40);
        const size = 4 + random(`s${i}`) * 7;
        const col = cy < 6 ? C.violet : C.cyan;
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: x,
              top: y,
              width: size,
              height: size,
              borderRadius: '50%',
              background: col,
              boxShadow: `0 0 12px ${col}`,
              opacity: o,
            }}
          />
        );
      })}
    </AbsoluteFill>
  );
};

const ColdOpen: React.FC = () => {
  const f = useCurrentFrame();
  const wipe = 72; // the chat ends
  const glitch = f >= wipe - 6 && f < wipe + 8 ? 1 - Math.abs(f - wipe) / 8 : 0;
  const oldChat = f < wipe ? 1 : ramp(f, [wipe, wipe + 14], [1, 0]);
  const implode = ramp(f, [128, 150], [0, 1], easeIn);
  const shake = glitch * 18 * (random(`sh${f}`) - 0.5);
  const headline = ramp(f, [24, 40], [0, 1]);
  const forget = f >= 84 ? Math.max(0, 1 - (f - 84) / 12) * 0.9 : 0;
  return (
    <AbsoluteFill style={{transform: `translateX(${shake}px) scale(${1 - implode * 0.6})`, opacity: 1 - implode * 0.9}}>
      <Backdrop grid={0.25} glow="#1a1240" />
      {/* the old conversation */}
      <div style={{position: 'absolute', left: 90, right: 90, top: 300}}>
        <div style={{opacity: oldChat, filter: glitch ? `hue-rotate(${glitch * 90}deg)` : undefined}}>
          <Glass style={{padding: '34px 30px 26px'}}>
            <div style={{display: 'flex', alignItems: 'center', gap: 16, marginBottom: 10}}>
              <div style={{width: 18, height: 18, borderRadius: 9, background: C.green, boxShadow: `0 0 14px ${C.green}`}} />
              <div style={{fontFamily: FONT, fontSize: 26, letterSpacing: 5, fontWeight: 700, color: 'rgba(233,228,255,0.6)'}}>CHAT · MONDAY</div>
            </div>
            <Bubble me text="Our API runs on Postgres. Remember that." at={6} />
            <Bubble text="Got it. Postgres it is." at={30} />
          </Glass>
        </div>
      </div>
      <Disintegrate at={wipe} />
      {/* the new conversation */}
      <div style={{position: 'absolute', left: 90, right: 90, top: 330, opacity: ramp(f, [wipe + 10, wipe + 20], [0, 1])}}>
        <Glass style={{padding: '34px 30px 26px'}}>
          <div style={{display: 'flex', alignItems: 'center', gap: 16, marginBottom: 10}}>
            <div style={{width: 18, height: 18, borderRadius: 9, background: C.amber, boxShadow: `0 0 14px ${C.amber}`}} />
            <div style={{fontFamily: FONT, fontSize: 26, letterSpacing: 5, fontWeight: 700, color: 'rgba(233,228,255,0.6)'}}>NEW CHAT · FRIDAY</div>
          </div>
          <Bubble me text="Which database does our API use?" at={wipe + 14} />
          <Bubble text="Sorry, I don't have any memory of that." at={wipe + 34} tone="bad" />
        </Glass>
      </div>
      {/* headline */}
      <div style={{position: 'absolute', left: 0, right: 0, top: 1000, opacity: headline, transform: `translateY(${(1 - headline) * 30}px)`}}>
        <GlitchText text={'Your AI\nforgets.'} size={150} amount={Math.max(glitch, forget, f > 120 ? 0.4 : 0)} seed="fg" />
      </div>
      <Flash at={wipe} color={C.rose} length={10} peak={0.35} />
      {/* everything is pulled to a point before the spark */}
      {implode > 0 ? (
        <AbsoluteFill style={{background: `radial-gradient(circle at 50% 50%, #fff ${implode * 6}%, transparent ${implode * 30}%)`, opacity: implode}} />
      ) : null}
    </AbsoluteFill>
  );
};

const Orbiters: React.FC<{cx: number; cy: number; count: number; start: number}> = ({cx, cy, count, start}) => {
  const f = useCurrentFrame();
  const grow = ramp(f, [start, start + 30], [0, 1]);
  return (
    <AbsoluteFill style={{mixBlendMode: 'screen'}}>
      {Array.from({length: count}).map((_, i) => {
        const r = 300 + random(`or${i}`) * 220;
        const a = random(`oa${i}`) * Math.PI * 2 + f * (0.012 + random(`ov${i}`) * 0.02);
        const tilt = 0.35;
        const z = Math.sin(a);
        const x = cx + Math.cos(a) * r * grow;
        const y = cy + z * r * tilt * grow;
        const size = 4 + (z + 1) * 4;
        const col = [C.cyan, C.violet, C.amber][i % 3];
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: x - size / 2,
              top: y - size / 2,
              width: size,
              height: size,
              borderRadius: '50%',
              background: col,
              opacity: (0.35 + 0.65 * (z + 1) / 2) * grow,
              boxShadow: `0 0 ${10 + size * 2}px ${col}`,
            }}
          />
        );
      })}
    </AbsoluteFill>
  );
};

const LogoReveal: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const cx = 540;
  const cy = 640;
  const s = spring({frame: f - 2, fps, config: {damping: 11, stiffness: 120, mass: 0.9}});
  const sweep = ramp(f, [26, 52], [-1, 1.4]);
  const word = 'VecForge';
  const pulse = 1 + 0.04 * Math.sin(f / 7);
  const settle = ramp(f, [100, 150], [0, 1]);
  return (
    <AbsoluteFill style={{transform: `scale(${1 + settle * 0.04})`}}>
      <Backdrop grid={0.6} speed={1.6} />
      <Rays x={cx} y={cy} size={1700} opacity={ramp(f, [0, 20], [0, 0.9])} />
      <div
        style={{
          position: 'absolute',
          left: cx - 420,
          top: cy - 420,
          width: 840,
          height: 840,
          borderRadius: '50%',
          background: `radial-gradient(circle, ${C.indigo}aa 0%, ${C.amber}22 35%, transparent 65%)`,
          transform: `scale(${pulse})`,
          mixBlendMode: 'screen',
        }}
      />
      <Orbiters cx={cx} cy={cy} count={70} start={6} />
      {/* the mark */}
      <div
        style={{
          position: 'absolute',
          left: cx - 230,
          top: cy - 230,
          width: 460,
          height: 460,
          transform: `scale(${0.15 + 0.85 * s}) rotate(${(1 - s) * -28}deg)`,
          filter: `drop-shadow(0 0 ${40 + 20 * Math.sin(f / 9)}px ${C.indigo}) drop-shadow(0 30px 60px rgba(0,0,0,0.5))`,
        }}
      >
        <Mark size={460} id="lg" spark={1 + 0.15 * Math.sin(f / 5)} />
        {/* light sweep across the tile */}
        <div style={{position: 'absolute', left: 18, top: 18, width: 424, height: 424, borderRadius: 100, overflow: 'hidden'}}>
          <div
            style={{
              position: 'absolute',
              top: -200,
              bottom: -200,
              width: 180,
              left: `${sweep * 100}%`,
              transform: 'rotate(20deg)',
              background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.55), transparent)',
            }}
          />
        </div>
      </div>
      {/* wordmark */}
      <div style={{position: 'absolute', top: 960, left: 0, right: 0, display: 'flex', justifyContent: 'center'}}>
        {word.split('').map((ch, i) => {
          const p = ramp(f, [18 + i * 2.5, 34 + i * 2.5], [0, 1]);
          return (
            <span key={i} style={{display: 'inline-block', opacity: p, transform: `translateY(${(1 - p) * 70}px)`, filter: `blur(${(1 - p) * 18}px)`}}>
              <ForgeText size={190} style={{backgroundSize: '1080px 100%', backgroundPosition: `${-i * 100 + 300}px 0`}}>
                {ch}
              </ForgeText>
            </span>
          );
        })}
      </div>
      <div
        style={{
          position: 'absolute',
          top: 1210,
          left: 0,
          right: 0,
          textAlign: 'center',
          opacity: ramp(f, [44, 60], [0, 1]),
          transform: `translateY(${ramp(f, [44, 60], [24, 0])}px)`,
        }}
      >
        <Eyebrow color={C.ink}>Local-first vector database</Eyebrow>
        <div style={{marginTop: 18, fontFamily: FONT, fontSize: 44, fontWeight: 600, color: C.amber}}>with memory for AI agents</div>
      </div>
      <Shockwave at={0} x={cx} y={cy} color={C.cyan} length={30} />
      <Shockwave at={4} x={cx} y={cy} color={C.amber} length={34} max={1200} />
      <Streak y={cy} at={0} length={26} />
      <Streak y={1060} at={30} length={24} color={C.amber} />
      <Bokeh count={14} seed="fg2" color={C.amber} />
      <Flash at={0} length={12} />
    </AbsoluteFill>
  );
};
