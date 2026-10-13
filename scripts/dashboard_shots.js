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
  await browser.close();
})();
