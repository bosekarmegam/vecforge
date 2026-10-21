import React from 'react';
import {AbsoluteFill, Easing, interpolate, random, useCurrentFrame} from 'remotion';
import {C, FONT, FORGE} from './theme';

export const ease = Easing.bezier(0.16, 1, 0.3, 1); // expo-out
export const easeIn = Easing.bezier(0.7, 0, 0.84, 0);

/** interpolate with clamping and an expo-out curve by default. */
export const ramp = (
  f: number,
  input: [number, number],
  output: [number, number],
  easing: (t: number) => number = ease,
) => interpolate(f, input, output, {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing});

/** Deep-space backdrop: radial glow, drifting perspective grid and soft bokeh. */
export const Backdrop: React.FC<{glow?: string; grid?: number; speed?: number}> = ({
  glow = C.bg2,
  grid = 0.5,
  speed = 1,
}) => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{background: C.bg0, overflow: 'hidden'}}>
      <AbsoluteFill
        style={{background: `radial-gradient(1300px 1500px at 50% 38%, ${glow} 0%, ${C.bg1} 50%, ${C.bg0} 100%)`}}
      />
      {/* floor grid in perspective, scrolling towards the viewer */}
      <div
        style={{
          position: 'absolute',
          left: -900,
          right: -900,
          top: 1180,
          height: 1500,
          opacity: grid,
          transform: 'perspective(700px) rotateX(72deg)',
          transformOrigin: '50% 0%',
          backgroundImage: `linear-gradient(${C.cyan}55 2px, transparent 2px), linear-gradient(90deg, ${C.cyan}40 2px, transparent 2px)`,
          backgroundSize: '120px 120px',
          backgroundPosition: `0px ${(f * 4 * speed) % 120}px`,
          maskImage: 'linear-gradient(to bottom, transparent, black 35%, black 70%, transparent)',
          WebkitMaskImage: 'linear-gradient(to bottom, transparent, black 35%, black 70%, transparent)',
        }}
      />
      {/* horizon glow */}
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: 1080,
          height: 260,
          opacity: grid,
          background: `radial-gradient(60% 50% at 50% 50%, ${C.indigo}88, transparent 70%)`,
        }}
      />
      <Bokeh count={34} seed="bg" />
    </AbsoluteFill>
  );
};

/** Soft out-of-focus lights drifting upwards. Radial gradients, no blur filter. */
export const Bokeh: React.FC<{count: number; seed: string; color?: string}> = ({count, seed, color}) => {
  const f = useCurrentFrame();
  const palette = [C.cyan, C.violet, C.amber, C.indigo];
  return (
    <AbsoluteFill style={{mixBlendMode: 'screen'}}>
      {Array.from({length: count}).map((_, i) => {
        const r = 14 + random(`${seed}r${i}`) * 60;
        const x = random(`${seed}x${i}`) * 1080;
        const y0 = random(`${seed}y${i}`) * 2100;
        const v = 0.4 + random(`${seed}v${i}`) * 1.6;
        const y = ((y0 - f * v) % 2100 + 2100) % 2100 - 90;
        const c = color ?? palette[i % palette.length];
        const tw = 0.35 + 0.35 * Math.sin(f / 18 + i);
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: x - r,
              top: y - r,
              width: r * 2,
              height: r * 2,
              borderRadius: '50%',
              opacity: tw * 0.55,
              background: `radial-gradient(circle, ${c}cc 0%, ${c}33 40%, transparent 70%)`,
            }}
          />
        );
      })}
    </AbsoluteFill>
  );
};

/** The VecForge mark, inlined so it scales without loss. */
export const Mark: React.FC<{size: number; id: string; spark?: number}> = ({size, id, spark = 1}) => (
  <svg width={size} height={size} viewBox="0 0 512 512" style={{display: 'block', overflow: 'visible'}}>
    <defs>
      <linearGradient id={`${id}t`} x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stopColor="#2A1F7A" />
        <stop offset="0.55" stopColor="#4B3BD1" />
        <stop offset="1" stopColor="#6C5CE7" />
      </linearGradient>
      <linearGradient id={`${id}a`} x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stopColor="#67E8F9" />
        <stop offset="0.6" stopColor="#C4B5FD" />
        <stop offset="1" stopColor="#FFB86B" />
      </linearGradient>
      <radialGradient id={`${id}s`} cx="0.5" cy="0.5" r="0.5">
        <stop offset="0" stopColor="#FFF4D6" />
        <stop offset="0.35" stopColor="#FFB547" />
        <stop offset="1" stopColor="#FF6A00" stopOpacity="0" />
      </radialGradient>
      <linearGradient id={`${id}h`} x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stopColor="#FFFFFF" stopOpacity="0.16" />
        <stop offset="0.5" stopColor="#FFFFFF" stopOpacity="0" />
      </linearGradient>
    </defs>
    <rect x="16" y="16" width="480" height="480" rx="112" fill={`url(#${id}t)`} />
    <rect x="16" y="16" width="480" height="240" rx="112" fill={`url(#${id}h)`} />
    <g stroke="#FFFFFF" strokeOpacity="0.07" strokeWidth="2">
      <line x1="96" y1="176" x2="416" y2="176" />
      <line x1="96" y1="256" x2="416" y2="256" />
      <line x1="96" y1="336" x2="416" y2="336" />
    </g>
    <g fill="none" stroke={`url(#${id}a)`} strokeLinecap="round" strokeLinejoin="round">
      <path d="M128 132 L256 384" strokeWidth="52" />
      <path d="M384 132 L256 384" strokeWidth="52" />
    </g>
    <g fill="#FFFFFF">
      <circle cx="128" cy="132" r="26" />
      <circle cx="384" cy="132" r="26" />
    </g>
    <g fill="#2A1F7A">
      <circle cx="128" cy="132" r="11" />
      <circle cx="384" cy="132" r="11" />
    </g>
    <g fill="#FFFFFF" fillOpacity="0.9">
      <circle cx="176" cy="226" r="9" />
      <circle cx="336" cy="226" r="9" />
    </g>
    <circle cx="256" cy="384" r={78 * spark} fill={`url(#${id}s)`} />
    <path d="M256 338 L266 374 L302 384 L266 394 L256 430 L246 394 L210 384 L246 374 Z" fill="#FFF7E6" />
  </svg>
);

/** Text with RGB-split glitch. `amount` 0 = clean, 1 = heavy. */
export const GlitchText: React.FC<{
  text: string;
  size: number;
  amount: number;
  color?: string;
  weight?: number;
  seed?: string;
}> = ({text, size, amount, color = '#fff', weight = 800, seed = 'g'}) => {
  const f = useCurrentFrame();
  const jit = (k: string) => (random(`${seed}${k}${f}`) - 0.5) * 2;
  const dx = amount * 22 * jit('x');
  const slice = amount > 0.05 ? random(`${seed}s${f}`) : 0;
  const base: React.CSSProperties = {
    position: 'absolute',
    inset: 0,
    fontFamily: FONT,
    fontSize: size,
    fontWeight: weight,
    letterSpacing: '-0.03em',
    lineHeight: 1.05,
    textAlign: 'center',
    whiteSpace: 'pre-line',
  };
  return (
    <div style={{position: 'relative', width: '100%', height: size * 1.25 * text.split('\n').length}}>
      <div style={{...base, color: '#ff2e63', transform: `translateX(${-dx - amount * 6}px)`, mixBlendMode: 'screen', opacity: amount > 0 ? 0.9 : 0}}>
        {text}
      </div>
      <div style={{...base, color: '#08f7fe', transform: `translateX(${dx + amount * 6}px)`, mixBlendMode: 'screen', opacity: amount > 0 ? 0.9 : 0}}>
        {text}
      </div>
      <div
        style={{
          ...base,
          color,
          transform: `translateX(${amount * 10 * jit('m')}px)`,
          clipPath: slice ? `inset(${slice * 70}% 0 ${(1 - slice) * 20}% 0)` : undefined,
        }}
      >
        {text}
      </div>
      {slice ? (
        <div style={{...base, color, transform: `translateX(${amount * 40 * jit('k')}px)`, clipPath: `inset(0 0 ${100 - slice * 70}% 0)`}}>{text}</div>
      ) : null}
    </div>
  );
};

/** Gradient headline text in the brand's cyan-violet-amber. */
export const ForgeText: React.FC<{children: React.ReactNode; size: number; weight?: number; style?: React.CSSProperties}> = ({
  children,
  size,
  weight = 800,
  style,
}) => (
  <span
    style={{
      fontFamily: FONT,
      fontSize: size,
      fontWeight: weight,
      letterSpacing: '-0.03em',
      backgroundImage: FORGE,
      WebkitBackgroundClip: 'text',
      backgroundClip: 'text',
      color: 'transparent',
      ...style,
    }}
  >
    {children}
  </span>
);

/** Full-frame flash that decays from `at`. */
export const Flash: React.FC<{at: number; color?: string; length?: number; peak?: number}> = ({
  at,
  color = '#ffffff',
  length = 10,
  peak = 1,
}) => {
  const f = useCurrentFrame();
  const o = f < at ? 0 : peak * Math.exp(-(f - at) / (length / 3));
  if (o < 0.01) return null;
  return <AbsoluteFill style={{background: color, opacity: o, mixBlendMode: 'screen'}} />;
};

/** Expanding ring shockwave centred at (x, y). */
export const Shockwave: React.FC<{at: number; x: number; y: number; color?: string; length?: number; max?: number}> = ({
  at,
  x,
  y,
  color = C.cyan,
  length = 28,
  max = 1500,
}) => {
  const f = useCurrentFrame();
  if (f < at || f > at + length) return null;
  const p = (f - at) / length;
  const r = ease(p) * max;
  return (
    <div
      style={{
        position: 'absolute',
        left: x - r,
        top: y - r,
        width: r * 2,
        height: r * 2,
        borderRadius: '50%',
        border: `${Math.max(2, 26 * (1 - p))}px solid ${color}`,
        opacity: 1 - p,
        boxShadow: `0 0 60px ${color}, inset 0 0 60px ${color}`,
      }}
    />
  );
};

/** Rotating god-rays behind a focal point. */
export const Rays: React.FC<{x: number; y: number; size: number; opacity: number; color?: string}> = ({
  x,
  y,
  size,
  opacity,
  color = C.violet,
}) => {
  const f = useCurrentFrame();
  return (
    <div
      style={{
        position: 'absolute',
        left: x - size / 2,
        top: y - size / 2,
        width: size,
        height: size,
        opacity,
        borderRadius: '50%',
        transform: `rotate(${f * 0.6}deg)`,
        background: `repeating-conic-gradient(from 0deg, ${color}55 0deg 4deg, transparent 4deg 18deg)`,
        maskImage: 'radial-gradient(circle, black 0%, transparent 65%)',
        WebkitMaskImage: 'radial-gradient(circle, black 0%, transparent 65%)',
        mixBlendMode: 'screen',
      }}
    />
  );
};

/** Horizontal anamorphic lens streak. */
export const Streak: React.FC<{y: number; at: number; length?: number; color?: string}> = ({y, at, length = 30, color = C.cyan}) => {
  const f = useCurrentFrame();
  if (f < at || f > at + length) return null;
  const p = (f - at) / length;
  const o = Math.sin(Math.PI * p);
  return (
    <div
      style={{
        position: 'absolute',
        left: -200,
        right: -200,
        top: y - 6,
        height: 12,
        opacity: o,
        background: `linear-gradient(90deg, transparent, ${color} 30%, #fff 50%, ${color} 70%, transparent)`,
        filter: 'blur(3px)',
        transform: `scaleX(${0.3 + p})`,
        mixBlendMode: 'screen',
      }}
    />
  );
};

/** A glass card. */
export const Glass: React.FC<{style?: React.CSSProperties; children?: React.ReactNode}> = ({style, children}) => (
  <div
    style={{
      borderRadius: 34,
      background: 'linear-gradient(160deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03))',
      border: '2px solid rgba(196,181,253,0.28)',
      boxShadow: '0 30px 80px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.15)',
      ...style,
    }}
  >
    {children}
  </div>
);

/** Small uppercase label above a headline. */
export const Eyebrow: React.FC<{children: React.ReactNode; color?: string}> = ({children, color = C.cyan}) => (
  <div style={{fontFamily: FONT, fontSize: 30, fontWeight: 700, letterSpacing: 8, color, textTransform: 'uppercase'}}>{children}</div>
);
