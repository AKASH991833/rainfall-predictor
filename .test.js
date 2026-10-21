module.exports.run = async (page, base, shot) => {
  const out = [];
  await page.waitForFunction(() => /%/.test(document.querySelector('#prob').textContent), {timeout: 10000});
  const p1 = await page.$eval('#prob', e => e.textContent);
  out.push(['probability rendered: ' + p1, /\d+%/.test(p1)]);
  const brier = await page.$eval('#m-brier', e => parseFloat(e.textContent));
  out.push(['Brier sane: ' + brier, brier > 0.05 && brier < 0.25]);
  const acc = await page.$eval('#m-acc', e => parseInt(e.textContent));
  out.push(['accuracy sane: ' + acc + '%', acc > 65 && acc < 98]);
  const factors = await page.$$eval('.factor', r => r.length);
  out.push(['5 factor rows', factors === 5]);
  // crank humidity + cloud + drop pressure -> probability must rise
  await page.$eval('#hum', e => {e.value = 98; e.dispatchEvent(new Event('input'));});
  await page.$eval('#cloud', e => {e.value = 95; e.dispatchEvent(new Event('input'));});
  await page.$eval('#pres', e => {e.value = 982; e.dispatchEvent(new Event('input'));});
  await page.select('#season', '1');
  await new Promise(r=>setTimeout(r,300));
  const p2 = parseInt(await page.$eval('#prob', e => e.textContent));
  out.push(['rainy conditions => high prob: ' + p2 + '%', p2 > 70]);
  const verdict = await page.$eval('#verdict', e => e.textContent);
  out.push(['verdict: ' + verdict, /Rain is likely/.test(verdict)]);
  if (shot) await page.screenshot({path: shot});
  return out;
};
