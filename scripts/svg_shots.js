const { chromium } = require('playwright');
const path = require('path');
const { pathToFileURL } = require('url');

(async () => {
  const browser = await chromium.launch();
  for (const name of ['architecture', 'agent_loop']) {
    const page = await browser.newPage({ viewport: { width: 1020, height: 640 } });
    const url = pathToFileURL(path.resolve(__dirname, '..', 'docs', `${name}.svg`)).href;
    await page.goto(url);
    await page.waitForTimeout(500);
    const info = await page.evaluate(() => {
      const s = document.querySelector('svg');
      const t = document.querySelectorAll('text');
      const missing = [...t].filter(n => !n.textContent.trim()).length;
      return { svg: !!s, texts: t.length, missing, w: s ? s.getBoundingClientRect().width : 0 };
    });
    console.log(name, JSON.stringify(info));
    const s = await page.$('svg');
    await s.screenshot({ path: path.join(__dirname, '..', 'shots', `${name}_svg.png`) });
    await page.close();
  }
  await browser.close();
})();
