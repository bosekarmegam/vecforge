// Render the Remotion segments of the reel to hf/assets/remotion/<id>.mp4.
//   node render-remotion.mjs                 # every segment
//   node render-remotion.mjs what-is         # one segment
//   node render-remotion.mjs what-is --still 40,120 --out dir   # PNG stills for review
import path from 'node:path';
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import {bundle} from '@remotion/bundler';
import {renderMedia, renderStill, selectComposition} from '@remotion/renderer';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined);
const ids = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
// Any Chrome Headless Shell works; set REMOTION_BROWSER to use one already on the machine.
const browserExecutable = process.env.REMOTION_BROWSER || null;

const serveUrl = await bundle({entryPoint: path.join(here, 'remotion', 'index.ts')});
const all = ['open-and-logo', 'what-is', 'agent-memory', 'features', 'end-card'];
const outDir = flag('--out') || path.join(here, 'hf', 'assets', 'remotion');
fs.mkdirSync(outDir, {recursive: true});

for (const id of ids.length ? ids : all) {
  const composition = await selectComposition({serveUrl, id, browserExecutable});
  const stills = flag('--still');
  if (stills) {
    for (const frame of stills.split(',').map(Number)) {
      const output = path.join(outDir, `${id}-${frame}.png`);
      await renderStill({serveUrl, composition, frame, output, browserExecutable});
      console.log('still', output);
    }
    continue;
  }
  const output = path.join(outDir, `${id}.mp4`);
  let last = -1;
  await renderMedia({
    serveUrl,
    composition,
    codec: 'h264',
    crf: 14,
    pixelFormat: 'yuv420p',
    outputLocation: output,
    browserExecutable,
    concurrency: Number(process.env.REMOTION_CONCURRENCY || 4),
    onProgress: ({progress}) => {
      const pct = Math.floor(progress * 10) * 10;
      if (pct !== last) console.log(`${id} ${pct}%`), (last = pct);
    },
  });
  console.log('wrote', output);
}
