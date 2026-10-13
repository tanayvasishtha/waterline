// Records the dashboard on :8711 as one clip per shot into video/clips/.
// Each clip is a playwright page-video (webm) converted to mp4 by ffmpeg.
// Run: NODE_PATH=~/node_modules node video/capture.js
const { chromium } = require('playwright');
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const BASE = 'http://localhost:8711';
const CLIPS = path.join(__dirname, 'clips');
const TMP = path.join(__dirname, 'rec');

const W = 1920, H = 1080;

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function smoothScroll(page, px, ms) {
  const steps = Math.max(10, Math.floor(ms / 60));
  await page.evaluate(async ({ px, steps }) => {
    for (let i = 1; i <= steps; i++) {
      window.scrollBy(0, px / steps);
      await new Promise(r => requestAnimationFrame(r));
    }
  }, { px, steps });
  await sleep(ms - steps * 17 > 0 ? 200 : 0);
}

async function clip(browser, name, fn) {
  const ctx = await browser.newContext({
    viewport: { width: W, height: H },
    recordVideo: { dir: TMP, size: { width: W, height: H } },
    deviceScaleFactor: 1,
  });
  const page = await ctx.newPage();
  await fn(page);
  await sleep(400);
  const vid = page.video();
  await ctx.close();
  const webm = await vid.path();
  const out = path.join(CLIPS, name + '.mp4');
  execFileSync('ffmpeg', ['-y', '-i', webm, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-r', '30', '-movflags', '+faststart', out], { stdio: 'pipe' });
  const dur = execFileSync('ffprobe', ['-v', 'quiet', '-show_entries', 'format=duration',
    '-of', 'csv=p=0', out]).toString().trim();
  console.log(name, dur + 's');
  fs.unlinkSync(webm);
}

async function clipMobile(browser, name, fn) {
  const ctx = await browser.newContext({
    viewport: { width: 390, height: 844 },
    recordVideo: { dir: TMP, size: { width: 390, height: 844 } },
    deviceScaleFactor: 1,
  });
  const page = await ctx.newPage();
  await fn(page);
  await sleep(400);
  const vid = page.video();
  await ctx.close();
  const webm = await vid.path();
  const out = path.join(CLIPS, name + '.mp4');
  execFileSync('ffmpeg', ['-y', '-i', webm, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-r', '30', '-movflags', '+faststart', out], { stdio: 'pipe' });
  const dur = execFileSync('ffprobe', ['-v', 'quiet', '-show_entries', 'format=duration',
    '-of', 'csv=p=0', out]).toString().trim();
  console.log(name, dur + 's');
  fs.unlinkSync(webm);
}

(async () => {
  fs.mkdirSync(CLIPS, { recursive: true });
  fs.mkdirSync(TMP, { recursive: true });
  const browser = await chromium.launch();

  // 1. hancock "all" with a slow scroll through the panels
  await clip(browser, 'shot01_hancock_all', async (page) => {
    await page.goto(BASE + '/?site=hancock&window=all', { waitUntil: 'networkidle' });
    await sleep(1200);
    await smoothScroll(page, 400, 4000);
    await sleep(600);
    await smoothScroll(page, -400, 3000);
    await sleep(600);
  });

  // 2. window switch to the September window
  await clip(browser, 'shot02_window_switch', async (page) => {
    await page.goto(BASE + '/?site=hancock&window=all', { waitUntil: 'networkidle' });
    await sleep(800);
    await page.selectOption('#winSel', '2026-09-01_2026-09-16');
    await page.waitForTimeout(1500);
    await sleep(1500);
  });

  // 3. open a non-trivial trace, read the step detail
  await clip(browser, 'shot03_trace_detail', async (page) => {
    await page.goto(BASE + '/?site=hancock&window=2026-09-01_2026-09-16', { waitUntil: 'networkidle' });
    await sleep(600);
    await page.evaluate(() => {
      document.querySelector('#traces').scrollIntoView({ block: 'start' });
    });
    await sleep(500);
    const li = page.locator('#traces li:not(.none)').first();
    await li.click();
    await sleep(1500);
    await page.evaluate(() => {
      const d = document.querySelector('#traceDetail');
      if (d) d.scrollIntoView({ block: 'nearest' });
    });
    await sleep(1500);
  });

  // 4. flags panel with the show-all toggle
  await clip(browser, 'shot04_flags_toggle', async (page) => {
    await page.goto(BASE + '/?site=hancock&window=all', { waitUntil: 'networkidle' });
    await sleep(600);
    await page.evaluate(() => {
      document.querySelector('#flags').scrollIntoView({ block: 'center' });
    });
    await sleep(1200);
    await page.check('#flagAll');
    await sleep(1800);
  });

  // 5. the fault test table
  await clip(browser, 'shot05_fault_table', async (page) => {
    await page.goto(BASE + '/?site=hancock&window=all', { waitUntil: 'networkidle' });
    await sleep(600);
    await page.evaluate(() => {
      const els = [...document.querySelectorAll('h2')];
      const t = els.find(e => e.textContent.includes('Fault test'));
      if (t) t.scrollIntoView({ block: 'start' });
    });
    await sleep(2500);
  });

  // 6. the failure gallery
  await clip(browser, 'shot06_gallery', async (page) => {
    await page.goto(BASE + '/?site=hancock&window=all', { waitUntil: 'networkidle' });
    await sleep(600);
    await page.evaluate(() => {
      const els = [...document.querySelectorAll('h2')];
      const t = els.find(e => e.textContent.includes('Where it fails'));
      if (t) t.scrollIntoView({ block: 'start' });
    });
    await sleep(2000);
    await smoothScroll(page, 500, 3000);
    await sleep(600);
  });

  // 7. mobile 390px
  await clipMobile(browser, 'shot07_mobile_390', async (page) => {
    await page.goto(BASE + '/?site=hancock&window=all', { waitUntil: 'networkidle' });
    await sleep(1200);
    await smoothScroll(page, 1200, 5000);
    await sleep(800);
  });

  await browser.close();
  console.log('DONE');
})();
