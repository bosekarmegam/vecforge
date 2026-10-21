import React from 'react';
import {Composition} from 'remotion';
import {OpenAndLogo} from './scenes/OpenAndLogo';
import {WhatIs} from './scenes/WhatIs';
import {AgentMemory} from './scenes/AgentMemory';
import {Features} from './scenes/Features';
import {EndCard} from './scenes/EndCard';
import {H, W} from './theme';

// Each composition is one segment of the 60 s reel; the HyperFrames master
// (../hf/index.html) places them on its timeline at the times noted here.
const SEGMENTS = [
  {id: 'open-and-logo', component: OpenAndLogo, seconds: 10}, // 0-10 s
  {id: 'what-is', component: WhatIs, seconds: 7.5}, // 10-17.5 s
  {id: 'agent-memory', component: AgentMemory, seconds: 7.5}, // 40-47.5 s
  {id: 'features', component: Features, seconds: 7.5}, // 47.5-55 s
  {id: 'end-card', component: EndCard, seconds: 5}, // 55-60 s
];

export const Root: React.FC = () => (
  <>
    {SEGMENTS.map((s) => (
      <Composition
        key={s.id}
        id={s.id}
        component={s.component}
        durationInFrames={Math.round(s.seconds * 30)}
        fps={30}
        width={W}
        height={H}
      />
    ))}
  </>
);
