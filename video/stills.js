// Renders stills (SVG/JPG/PNG) into 1920x1080 mp4 clips with a slow zoom.
// SVGs are rasterized at 3x via playwright first, since ffmpeg cannot read svg.
// Run: NODE_PATH=~/node_modules node video/stills.js
const { chromium } = require('playwright');
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

const CLIPS = path.join(__dirname, 'clips');
const TMP = path.join(__dirname, 'rec');
const ROOT = path.join(__dirname, '..');

// [source, clip name, seconds]
const STILLS = [
  ['docs/agent_loop.svg', 'still_agent_loop', 26],
  ['docs/architecture.svg', 'still_architecture', 18],
  ['results/exhibit_before.jpg', 'still_exhibit_before', 16],
  ['results/exhibit_after.jpg', 'still_exhibit_after', 12],
  ['results/gallery/registration_failed_1.jpg', 'still_gal_registration', 9],
  ['results/gallery/night_abstain_1.jpg', 'still_gal_night', 6],
  ['results/gallery/substitution_1.jpg', 'still_gal_substitution', 6],
  ['results/gallery/largest_agent_error_1.jpg', 'still_gal_largest_error', 6],
];

const ZOOM = "zoompan=z='min(1.0+on*0.0011,1.14)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'";

async function rasterize(src) {
  const browser = await chromium.launch();
  const page = await browser.newPage({ deviceScaleFactor: 3 });
  await page.goto(pathToFileURL(path.resolve(src)).href);
  const el = await page.$('svg');
  const out = path.join(TMP, path.basename(src, '.svg') + '.png');
  await el.screenshot({ path: out });
  await browser.close();
  return out;
}

(async () => {
  fs.mkdirSync(CLIPS, { recursive: true });
  fs.mkdirSync(TMP, { recursive: true });
  for (const [src, name, secs] of STILLS) {
    let inPath = path.join(ROOT, src);
    if (!fs.existsSync(inPath)) { console.log('MISSING', src); continue; }
    if (src.endsWith('.svg')) {
      inPath = await rasterize(inPath);
      console.log('rasterized', src);
    }
    const frames = Math.round(secs * 30);
    const out = path.join(CLIPS, name + '.mp4');
    execFileSync('ffmpeg', [
      '-y', '-loop', '1', '-i', inPath,
      '-vf', `scale=3840:2160:force_original_aspect_ratio=decrease,pad=3840:2160:(ow-iw)/2:(oh-ih)/2:color=0xf6f4ee,${ZOOM}:d=${frames}:s=1920x1080:fps=30,format=yuv420p`,
      '-t', String(secs), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
      '-movflags', '+faststart', out,
    ], { stdio: 'pipe' });
    const dur = execFileSync('ffprobe', ['-v', 'quiet', '-show_entries', 'format=duration',
      '-of', 'csv=p=0', out]).toString().trim();
    console.log(name, dur + 's');
  }
  console.log('DONE');
})();
