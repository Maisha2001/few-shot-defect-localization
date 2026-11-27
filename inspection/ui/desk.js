"use strict";
const el = id => document.getElementById(id);
const arms = ["active-32", "random-32"];
const prefix = arm => arm.split("-")[0];
const canvas = arm => el(prefix(arm));
let token = "", busy = false, image = null, cases = {}, grid = new Uint8Array(3136);
let dirty = false, stroke = null, revision = 0, selected = "active-32";
function message(text, error=false) {
  el("status").textContent = text;
  el("status").classList.toggle("error", error);
}
async function api(path, options={}, binary=false) {
  const headers = new Headers(options.headers);
  headers.set("Authorization","Bearer "+token);
  const response = await fetch("/api"+path,{...options,headers});
  if (!response.ok) throw Error(response.status === 409 ? "Another reviewer saved a revision. Reopen the case before editing again." :
    "Request failed ("+response.status+"). "+(response.status===401 ? "Check the local access key." : "Check the input and try again."));
  return binary ? response.blob() : response.json();
}
async function run(action) {
  if (busy) return;
  busy=true; el("desk").disabled=true;
  try { await action(); } catch(e) { message(e.message,true); }
  finally { busy=false; el("desk").disabled=!el("connect").hidden; }
}
function mayLeave() { return !dirty || window.confirm("Discard the unsaved mask?"); }
function render() {
  for (const arm of arms) {
    const node=canvas(arm), row=cases[arm], ctx=node.getContext("2d");
    node.parentElement.classList.toggle("selected",!!row && arm===selected);
    const count=row ? row.probability.reduce((n,line)=>n+line.filter(p=>p>=row.threshold).length,0) : 0;
    el(prefix(arm)+"-read").textContent = row ? count+" flagged cells" : "Unavailable";
    el(prefix(arm)+"-read").title=row ? "Image ranking score "+row.score.toPrecision(4)+" · pixel threshold "+row.threshold.toFixed(3) : "";
    if (!row || !image) {ctx.clearRect(0,0,node.width,node.height);continue;}
    const scale=840/Math.max(image.width,image.height);
    node.width=Math.max(1,Math.round(image.width*scale));node.height=Math.max(1,Math.round(image.height*scale));
    ctx.drawImage(image,0,0,node.width,node.height);
    const layer=document.createElement("canvas");layer.width=layer.height=56;
    const paint=layer.getContext("2d");const pixels=paint.createImageData(56,56);
    for(let i=0;i<3136;i++) {
      const idx=i*4;
      if(arm===selected && grid[i]) pixels.data.set([40,202,204,190],idx);
      else if(el("overlay").checked && row.probability[Math.floor(i/56)][i%56]>=row.threshold)
        pixels.data.set([255,140,67,150],idx);
    }
    paint.putImageData(pixels,0,0);ctx.imageSmoothingEnabled=false;
    ctx.drawImage(layer,0,0,node.width,node.height);
  }
  el("revision").textContent = cases[selected] ? "Revision "+revision+(dirty?" · unsaved":"") : "";
}
async function activate(arm) {
  const row=cases[arm];
  const annotation=await api("/cases/"+row.id+"/annotation");
  const cells=new Uint8Array(3136);
  for(const [start,length] of annotation.runs) cells.fill(1,start,start+length);
  selected=arm;el("arm").value=arm;revision=annotation.revision;grid=cells;dirty=false;
  render();
}
async function display(rows,blob,title,arm) {
  const bitmap=await createImageBitmap(blob);
  if(image) image.close(); image=bitmap;cases=rows;
  el("name").textContent=title;
  for(const option of el("arm").options) option.disabled=!rows[option.value];
  await activate(rows[arm]?arm:Object.keys(rows)[0]);
}
async function inspect(blob,title) {
  const rows={};
  for(const arm of arms) rows[arm]=await api("/cases?arm="+arm,{method:"POST",headers:{"Content-Type":blob.type},body:blob});
  await display(rows,blob,title,selected);
  await queue(); message("Both predictions are ready.");
}
async function queue() {
  const rows=await api("/queue?limit=100"),list=el("queue");
  list.replaceChildren();
  for(const row of rows) {
    const arm=row.version.split(":").pop();
    const button=document.createElement("button");
    button.textContent=prefix(arm)+" · "+row.id.slice(0,8)+" · "+row.uncertainty.toFixed(3);
    button.onclick=()=>run(async()=>{
      if(!mayLeave()) return;
      const ids=await api("/cases/"+row.id+"/siblings"),siblings={};
      for(const [key,id] of Object.entries(ids)) siblings[key]=await api("/cases/"+id);
      if(!Object.keys(siblings).length) throw Error("This case belongs to a different model version.");
      await display(siblings,await api("/cases/"+row.id+"/image",{},true),"Case "+row.id.slice(0,8),arm);
      message("Queued prediction opened.");
    }); list.append(button);
  }
  if(!rows.length) list.textContent="Queue empty.";
}
el("connect").onsubmit=e=>{e.preventDefault();run(async()=>{
  token=el("key").value.trim();await api("/models");
  const resp=await fetch("/examples/index.json");if(!resp.ok) throw Error("Sample list unavailable.");
  const examples=await resp.json();el("sample").replaceChildren();
  for(const example of examples) el("sample").add(new Option(example.title,example.file));
  await queue();el("connect").hidden=true;el("key").value="";el("connection").textContent="Connected locally";
  message("Choose a sample or upload a PNG or JPEG.");
});};
el("open").onclick=()=>run(async()=>{
  if(!mayLeave())return;
  const option=el("sample").selectedOptions[0],response=await fetch("/examples/"+encodeURIComponent(option.value));
  if(!response.ok)throw Error("Sample image unavailable.");
  await inspect(await response.blob(),option.text);
});
el("upload").onchange=e=>{const file=e.target.files[0];e.target.value="";if(file)run(async()=>{if(mayLeave())await inspect(file,file.name);});};
el("arm").onchange=e=>{const next=e.target.value;e.target.value=selected;run(async()=>{if(cases[next] && mayLeave())await activate(next);});};
el("overlay").onchange=render;
function pos(node,e) {
  const r=node.getBoundingClientRect();
  return [Math.min(55,Math.max(0,Math.floor(56*(e.clientX-r.left)/r.width))),
          Math.min(55,Math.max(0,Math.floor(56*(e.clientY-r.top)/r.height)))];
}
function mark(node,e) {
  const point=pos(node,e),last=stroke || point, steps=Math.max(1,Math.abs(point[0]-last[0]),Math.abs(point[1]-last[1]));
  for(let i=0;i<=steps;i++) {
    const x=Math.round(last[0]+(point[0]-last[0])*i/steps), y=Math.round(last[1]+(point[1]-last[1])*i/steps);
    grid[56*y+x]=Number(el("tool").value==="paint");
  }
  dirty=true;stroke=point;render();
}
for(const arm of arms) {
  const node=canvas(arm);
  node.onpointerdown=e=>{if(busy || selected!==arm || !cases[arm])return;node.setPointerCapture(e.pointerId);stroke=null;mark(node,e);};
  node.onpointermove=e=>{if(stroke && node.hasPointerCapture(e.pointerId))mark(node,e);};
  node.onlostpointercapture=()=>{stroke=null;};
}
el("clear").onclick=()=>{if(cases[selected]) {grid.fill(0);dirty=true;render();}};
el("save").onclick=()=>run(async()=>{
  const row=cases[selected];if(!row)return;
  const runs=[];for(let i=0;i<grid.length;i++){if(!grid[i])continue;const start=i;while(i+1<grid.length && grid[i+1])i++;runs.push([start,i-start+1]);}
  const label=runs.length?"defect":"normal";
  const receipt=await api("/cases/"+row.id+"/review",{method:"PUT",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({request_id:crypto.randomUUID(),expected_revision:revision,label,runs})});
  revision=receipt.revision;dirty=false;render();await queue();message("Saved "+label+" review, revision "+revision+".");
});
el("export").onclick=()=>run(async()=>{
  const reviews=[];let after=0;
  while(true){const page=await api("/reviews?limit=200&after="+after);reviews.push(...page.reviews);
    if(page.reviews.length<200)break;after=page.reviews.at(-1).seq;}
  const blob=new Blob([JSON.stringify({grid:[56,56],encoding:"row-major start,length",reviews},null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="inspection-reviews.json";a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);message("Exported "+reviews.length+" review revisions.");
});
window.addEventListener("beforeunload",e=>{if(dirty){e.preventDefault();e.returnValue="";}});
