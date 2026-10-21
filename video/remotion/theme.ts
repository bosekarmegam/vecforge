// Brand colours, shared with the HyperFrames scenes (../hf/index.html).
export const C = {
  bg0: '#07051a',
  bg1: '#110b36',
  bg2: '#1d1558',
  deep: '#2A1F7A',
  indigo: '#6C5CE7',
  cyan: '#67E8F9',
  violet: '#C4B5FD',
  amber: '#FFB86B',
  green: '#7CF2B0',
  rose: '#FF7A9A',
  ink: '#E9E4FF',
};

export const FONT = 'Inter, "Inter Display", system-ui, sans-serif';
export const MONO = '"DejaVu Sans Mono", ui-monospace, monospace';
export const FORGE = `linear-gradient(90deg, ${C.cyan}, ${C.violet} 55%, ${C.amber})`;

// Layout: scene content stays above y = 1400; HyperFrames draws captions in 1460-1660.
export const W = 1080;
export const H = 1920;
