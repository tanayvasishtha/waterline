// Records the dashboard on :8711 as one clip per shot into video/clips/.
// Clip lengths cover the longest EDL segment using them plus 1 s; motion is
// slow pans, scrolls and highlights, never a frozen frame longer than ~2 s.
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
}

// horizontal pan inside .chartwrap (the chart's internal scroller)
async function panChart(page, px, ms) {
  const steps = Math.max(10, Math.floor(ms / 60));
  await page.evaluate(async ({ px, steps }) => {
    const el = document.querySelector('.chartwrap');
    if (!el) return;
    for (let i = 1; i <= steps; i++) {
      el.scrollLeft += px / steps;
      await new Promise(r => requestAnimationFrame(r));
    }
  }, { px, steps });
}

async function record(page, name) {
  const vid = page.video();
  await page.context().close();
  const webm = await vid.path();
  const out = path.join(CLIPS, name + '.mp4');
  execFileSync('ffmpeg', ['-y', '-i', webm, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-r', '30', '-movflags', '+faststart', out], { stdio: 'pipe' });
  const dur = execFileSync('ffprobe', ['-v', 'quiet', '-show_entries', 'format=duration',
    '-of', 'csv=p=0', out]).toString().trim();
  console.log(name, dur + 's');
  fs.unlinkSync(webm);
}

async function clip(browser, name, fn, mobile = false) {
  const ctx = await browser.newContext({
    viewport: mobile ? { width: 390, height: 844 } : { width: W, height: H },
    recordVideo: { dir: TMP, size: mobile ? { width: 390, height: 844 } : { width: W, height: H } },
    deviceScaleFactor: 1,
  });
  const page = await ctx.newPage();
  await fn(page);
  await record(page, name);
}

async function scrollToEl(page, sel, block = 'start') {
  await page.evaluate(([s, b]) => {
    const e = document.querySelector(s);
    if (e) e.scrollIntoView({ block: b });
  }, [sel, block]);
}

async function scrollToH2(page, text) {
  await page.evaluate((t) => {
    const e = [...document.querySelectorAll('h2')].find(h => h.textContent.includes(t));
    if (e) e.scrollIntoView({ block: 'start' });
  }, text);
}

(async () => {
  fs.mkdirSync(CLIPS, { recursive: true });
  fs.mkdirSync(TMP, { recursive: true });
  const browser = await chromium.launch();

  // shot01: hancock "all"  -  slow chart pan, then slow page scroll (~26 s)
  await clip(browser, 'shot01_hancock_all', async (page) => {
    await page.goto(BASE + '/?site=hancock&window=all', { waitUntil: 'networkidle' });
    // playwright video records at ~0.45x wall time; schedule ~60 s of action
    await sleep(2000);
    await panChart(page, 500, 12000);
    await panChart(page, -350, 10000);
    await sleep(500);
    await smoothScroll(page, 450, 12000);
    await smoothScroll(page, -300, 9000);
    await smoothScroll(page, 250, 8000);
    await panChart(page, 250, 6000);
    await smoothScroll(page, -200, 5000);
    await sleep(2500);
  });

  // shot02: window switch to September (~11 s)
  await clip(browser, 'shot02_window_switch', async (page) => {
    await page.goto(BASE + '/?site=hancock&window=all', { waitUntil: 'networkidle' });
    await sleep(1500);
    await page.selectOption('#winSel', '2026-09-01_2026-09-16');
    await page.waitForTimeout(2500);
    await sleep(3500);
    await panChart(page, 300, 4000);
    await panChart(page, -150, 2500);
    await sleep(1500);
  });

  // shot03: the exhibit trace, deep-linked, steps highlighted ~3 s apart (~26 s)
  await clip(browser, 'shot03_trace_detail', async (page) => {
    await page.goto(
      BASE + '/?site=in_overland&window=2026-04-04_2026-04-05&trace=in_overland-20260404T2045Z',
      { waitUntil: 'networkidle' });
    await sleep(2500);
    await scrollToEl(page, '#traces', 'start');
    await sleep(2500);
    const n = await page.evaluate(() =>
      document.querySelectorAll('#traceDetail .tstep').length);
    for (let i = 0; i < n; i++) {
      await page.evaluate((i) => {
        document.querySelectorAll('#traceDetail .tstep').forEach((el, j) =>
          el.classList.toggle('hl', j === i));
      }, i);
      await sleep(4000);
    }
    await page.evaluate(() =>
      document.querySelectorAll('#traceDetail .tstep').forEach(el => el.classList.remove('hl')));
    await sleep(2000);
    await page.evaluate(() => {
      const t = document.querySelector('#traceDetail .tfinal');
      if (t) { t.style.background = '#e4efef'; t.scrollIntoView({ block: 'nearest' }); }
    });
    await sleep(4000);
    await smoothScroll(page, 120, 2500);
    await smoothScroll(page, -120, 2500);
    await sleep(2000);
  });

  // shot04: flags panel, open the toggle (~13 s)
  await clip(browser, 'shot04_flags_toggle', async (page) => {
    await page.goto(BASE + '/?site=hancock&window=all', { waitUntil: 'networkidle' });
    await sleep(2000);
    await scrollToEl(page, '#flags', 'center');
    await sleep(3000);
    await page.check('#flagAll');
    await sleep(5000);
    await smoothScroll(page, 250, 5000);
    await smoothScroll(page, -200, 3000);
    await sleep(1500);
  });

  // shot05: fault table, slow hold with a highlight walking the rows (~24 s)
  await clip(browser, 'shot05_fault_table', async (page) => {
    await page.goto(BASE + '/?site=hancock&window=all', { waitUntil: 'networkidle' });
    await sleep(1500);
    await scrollToH2(page, 'Fault test');
    await sleep(1200);
    const rows = await page.evaluate(() =>
      document.querySelectorAll('#faultTable tr').length);
    for (let i = 0; i < rows; i++) {
      await page.evaluate((i) => {
        document.querySelectorAll('#faultTable tr').forEach((r, j) => {
          r.style.background = j === i ? '#e4efef' : '';
        });
      }, i);
      await sleep(3500);
    }
    await page.evaluate(() =>
      document.querySelectorAll('#faultTable tr').forEach(r => r.style.background = ''));
    await sleep(2000);
  });

  // shot06: failure gallery, slow pan over the cards (~13 s)
  await clip(browser, 'shot06_gallery', async (page) => {
    await page.goto(BASE + '/?site=in_overland&window=all', { waitUntil: 'networkidle' });
    await sleep(2000);
    await scrollToH2(page, 'Where it fails');
    await sleep(3000);
    await smoothScroll(page, 500, 6000);
    await sleep(2500);
    await smoothScroll(page, -250, 4000);
    await sleep(2000);
  });

  // shot07: mobile 390px (~7 s)
  await clip(browser, 'shot07_mobile_390', async (page) => {
    await page.goto(BASE + '/?site=hancock&window=all', { waitUntil: 'networkidle' });
    await sleep(2500);
    await smoothScroll(page, 900, 5000);
    await smoothScroll(page, -400, 3000);
    await sleep(1500);
  }, true);

  await browser.close();
  console.log('DONE');
})();
