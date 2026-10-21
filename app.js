/* RainWise - logistic regression rain probability with honest calibration. */
(()=>{
const $=s=>document.querySelector(s);
function mulberry(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const r=mulberry(42024);
const N=1095; // 3 years
// simulate days: latent moisture drives humidity/cloud/pressure/rain
const DAYS=[];
for(let i=0;i<N;i++){
  const season=Math.floor((i%365)/91.25)%4; // 0 spring 1 monsoon 2 autumn 3 winter
  const seasBoost=[0,0.9,-0.1,-0.35][season];
  const m=seasBoost+(r()*2-1)*1.1; // latent moisture
  const hum=Math.max(8,Math.min(100,55+m*17+(r()*2-1)*9));
  const cloud=Math.max(0,Math.min(100,42+m*22+(r()*2-1)*14));
  const pres=Math.max(975,Math.min(1042,1012-m*9+(r()*2-1)*5));
  const wind=Math.max(0,Math.min(60,12+m*4+r()*14));
  const temp=Math.max(-5,Math.min(42,21+seasBoost*3+(r()*2-1)*8+ (season===1?-3:0)));
  const z=-3.1+hum*0.028+cloud*0.024+(1012-pres)*0.055+wind*0.012+Math.abs(temp-19)*(-0.02)+[0,1.15,-0.1,-0.5][season];
  const p=1/(1+Math.exp(-z));
  DAYS.push({hum,cloud,pres,wind,temp,season,rain:r()<p?1:0});
}
// features normalized
const F=d=>[1,d.hum/100,d.cloud/100,(1012-d.pres)/30,d.wind/60,d.temp/42,[0,1,0,0,0][0]&&0,[0,1.15,-0.1,-0.5][d.season]/1.2];
function trainLogit(rows,lr=0.5,iters=900){
  const nf=F(rows[0]).length,w=new Float64Array(nf);
  for(let it=0;it<iters;it++){
    const g=new Float64Array(nf);
    rows.forEach(d=>{const f=F(d);let z=0;for(let j=0;j<nf;j++)z+=w[j]*f[j];
      const p=1/(1+Math.exp(-z)),e=p-d.rain;
      for(let j=0;j<nf;j++)g[j]+=e*f[j];});
    for(let j=0;j<nf;j++)w[j]-=lr*g[j]/rows.length;
  }
  return w;
}
const t0=performance.now();
const W=trainLogit(DAYS);
const trainMs=Math.max(1,Math.round(performance.now()-t0));
const prob=d=>{const f=F(d);let z=0;for(let j=0;j<W.length;j++)z+=W[j]*f[j];return 1/(1+Math.exp(-z));};
// metrics + calibration on all days
let brier=0,acc=0;const bins=Array.from({length:10},()=>({n:0,p:0,r:0}));
DAYS.forEach(d=>{const p=prob(d);brier+=(p-d.rain)**2;if(p>0.5===!!d.rain)acc++;
  const b=Math.min(9,Math.floor(p*10));bins[b].n++;bins[b].p+=p;bins[b].r+=d.rain;});
brier/=N;acc/=N;
$('#m-brier').textContent=brier.toFixed(3);
$('#m-acc').textContent=Math.round(acc*100)+'%';
$('#m-n').textContent=N;
$('#m-ms').textContent=trainMs+' ms';
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function drawCalib(){
  const cv=$('#calib'),dpr=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;
  cv.width=w*dpr;cv.height=h*dpr;const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);
  const L=36,R=w-10,T=10,B=h-24;
  const sx=v=>L+v*(R-L),sy=v=>B-v*(B-T);
  ctx.strokeStyle=css('--border');ctx.fillStyle=css('--muted');ctx.font='10px system-ui';
  [0,.25,.5,.75,1].forEach(g=>{ctx.beginPath();ctx.moveTo(L,sy(g));ctx.lineTo(R,sy(g));ctx.stroke();
    ctx.fillText(Math.round(g*100)+'%',6,sy(g)+3);ctx.fillText(Math.round(g*100)+'%',sx(g)-10,h-8);});
  // perfect line
  ctx.setLineDash([5,5]);ctx.strokeStyle='#8a97ad';ctx.beginPath();ctx.moveTo(sx(0),sy(0));ctx.lineTo(sx(1),sy(1));ctx.stroke();ctx.setLineDash([]);
  ctx.fillText('perfect',sx(.72),sy(.72)-8);
  // model curve
  ctx.strokeStyle=css('--accent');ctx.lineWidth=2.6;ctx.beginPath();
  let started=false;
  bins.forEach(b=>{if(!b.n)return;const x=b.p/b.n,y=b.r/b.n;
    started?ctx.lineTo(sx(x),sy(y)):(ctx.moveTo(sx(x),sy(y)),started=true);});
  ctx.stroke();
  bins.forEach(b=>{if(!b.n)return;const x=b.p/b.n,y=b.r/b.n;
    ctx.fillStyle=css('--accent');ctx.beginPath();ctx.arc(sx(x),sy(y),4.5,0,7);ctx.fill();
    ctx.fillStyle=css('--muted');ctx.fillText('n='+b.n,sx(x)-12,sy(y)+16);});
  ctx.fillStyle=css('--muted');ctx.fillText('predicted ->',R-66,h-8);ctx.save();ctx.translate(10,T+44);ctx.rotate(-Math.PI/2);ctx.fillText('actual rain rate',0,0);ctx.restore();
}
const NAMES=['hum','pres','cloud','wind','temp'];
const LABELS={hum:'Humidity',pres:'Low pressure',cloud:'Cloud cover',wind:'Wind',temp:'Temperature'};
const COEFS=[W[1],W[3],W[2],W[4],W[5]];
function current(){return{hum:+$('#hum').value,pres:+$('#pres').value,cloud:+$('#cloud').value,wind:+$('#wind').value,temp:+$('#temp').value,season:+$('#season').value};}
function render(){
  const d=current(),p=prob(d);
  const pct=Math.round(p*100);
  $('#prob').textContent=pct+'%';
  const arc=$('#arc'),C=326.7;
  arc.style.transition='stroke-dashoffset .5s cubic-bezier(.2,.8,.2,1)';
  arc.style.strokeDashoffset=String(C*(1-p));
  const v=$('#verdict'),vs=$('#verdictSub');
  if(p>=0.65){v.textContent='Rain is likely';v.className='verdict rain';vs.textContent='Carry an umbrella - conditions strongly favor rain.';}
  else if(p>=0.4){v.textContent='A real chance';v.className='verdict rain';vs.textContent='Toss-up conditions. An umbrella is cheap insurance.';}
  else if(p>=0.2){v.textContent='Probably dry';v.className='verdict dry';vs.textContent='Some moisture around, but rain is unlikely.';}
  else{v.textContent='Sunglasses weather';v.className='verdict dry';vs.textContent='Dry and stable - leave the umbrella home.';}
  $('#advice').innerHTML=p>=0.5?'High humidity and low pressure are the classic pre-rain mix. Outdoor plans after a few hours may beat the rain window.':'Stable pressure keeps clouds from condensing into rain. Good day for outdoor plans.';
  // factor contributions
  const contribs=NAMES.map((n,i)=>({n:LABELS[n],c:COEFS[i]*F(d)[i===1?3:i===2?2:i===3?4:i===0?1:5]}));
  const mx=Math.max(...contribs.map(c=>Math.abs(c.c)),0.01);
  $('#factors').innerHTML=contribs.map(c=>{
    const pos=c.c>0;
    return`<div class="factor"><span>${c.n}</span>
      <div class="fb"><i style="width:${Math.round(Math.abs(c.c)/mx*100)}%;background:${pos?'linear-gradient(90deg,var(--accent),var(--accent-2))':'#16a34a'}"></i></div>
      <span class="fv" style="color:${pos?'var(--accent)':'#16a34a'}">${pos?'+':''}${(c.c*100).toFixed(0)}</span></div>`;}).join('')
    +'<p style="font-size:11.5px;color:var(--muted);margin-top:6px">Contribution of each input to the rain log-odds (positive = more rain, green = suppressing rain).</p>';
}
NAMES.forEach(n=>{
  $('#'+n).oninput=e=>{
    const u={hum:'%',pres:' hPa',cloud:'%',wind:' km/h',temp:'°C'}[n];
    $('#'+n+'Out').textContent=e.target.value+u;
    render();};
});
$('#season').onchange=render;
window.addEventListener('themechange',drawCalib);
let rz;window.addEventListener('resize',()=>{clearTimeout(rz);rz=setTimeout(drawCalib,150);});
render();drawCalib();
})();
