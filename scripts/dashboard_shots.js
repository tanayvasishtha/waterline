const { chromium } = require('playwright');
const path = require('path');

const shots = [
  ['hancock', 'all', 1440],
  ['hancock', 'all', 390],
  ['hancock', '2026-09-01_2026-09-16', 1440],
  ['hancock', '2026-09-01_2026-09-16', 390],
  ['in_overland', '2023', 1440],
  ['in_overland', '2023', 390],
];

(async () => {
  const browser = await chromium.launch();
  const outDir = path.join(__dirname, '..', 'shots');
  for (const [site, win, w] of shots) {
    const page = await browser.newPage({ viewport: { width: w, height: 1200 } });
    await page.goto(`http://localhost:8711/?site=${site}`, { waitUntil: 'networkidle' });
    if (win !== 'all') {
      await page.selectOption('#winSel', win);
      await page.waitForTimeout(800);
    }
    await page.waitForTimeout(400);
    const sw = await page.evaluate(() => document.documentElement.scrollWidth);
    const name = `final_${site}_${win}_${w}.png`;
    await page.screenshot({ path: path.join(outDir, name), fullPage: true });
    console.log(`${name} scrollWidth=${sw} viewport=${w} overflow=${sw > w}`);
    await page.close();
  }
  // ?window= deep link must select the requested window and filter the
  // trace list (io windows are year aggregates: 2026 = 2026-03/2026-04 rows)
  const dp = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
  await dp.goto('http://localhost:8711/?site=in_overland&window=2026&trace=in_overland-20260404T2045Z', { waitUntil: 'networkidle' });
  await dp.waitForTimeout(800);
  const sel = await dp.evaluate(() => document.querySelector('#winSel').value);
  const rows = await dp.evaluate(() => [...document.querySelectorAll('#traces li:not(.none)')].map(x => x.textContent.slice(0, 10)));
  const bad = rows.filter(r => !/^(03-|04-)/.test(r));
  const detail = await dp.evaluate(() => document.querySelectorAll('#traceDetail .tstep').length);
  console.log(`deep-link: winSel=${sel} rows=${rows.length} out-of-window=${bad.length} steps=${detail}`);
  if (sel !== '2026' || bad.length || detail < 3) console.log('DEEP-LINK CHECK FAILED');
  await dp.close();
  await browser.close();
})();
