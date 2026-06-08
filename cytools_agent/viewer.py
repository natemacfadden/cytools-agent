# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  A zero-dependency viewer for an orchestrator session, in two
#               modes sharing one look (_STYLE) and one renderer (_RENDER_JS):
#                 - live server: polls the two JSONL files a session writes
#                   (scratch/session.jsonl, scratch/evidence.jsonl) and renders
#                   the conversation + progress as they grow.
#                       python -m cytools_agent.viewer [port]
#                 - static export: bakes a saved log into ONE self-contained
#                   .html (data + figures embedded), so anyone can open it in a
#                   browser -- no server, no Python.
#                       python -m cytools_agent.viewer export [stamp]
#               All functions here are human-read.
# -----------------------------------------------------------------------------

# external imports
import base64
import glob
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# local imports
from cytools_agent.orchestrator import (EVIDENCE_PATH, LOG_DIR, SESSION_PATH,
                                        export_script)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRATCH = os.path.join(_REPO, "scratch")
_REPLAY_DIR = os.path.join(_SCRATCH, "replays")


# replay support -- a saved session re-runs in an ISOLATED subprocess (its own
# figure dir, no LLM, no writes to the live logs), so it is safe alongside a
# live run.
# human-read
def _logs():
    """The saved sessions, newest first: stamp, question, answer."""
    out = []
    for p in sorted(glob.glob(os.path.join(LOG_DIR, "session_*.json")),
                    reverse=True):
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        stamp = os.path.basename(p)[len("session_"):-len(".json")]
        out.append({"stamp": stamp, "question": (d.get("question") or "")[:90],
                    "answer": (d.get("answer") or "")[:140]})
    return out


# human-read
def _start_replay(stamp):
    """Launch the replay of session `stamp` as a detached subprocess with its
    own figure dir scratch/replays/<stamp>/, so it cannot touch the live run's
    figures. Returns True if started."""
    logpath = os.path.join(LOG_DIR, f"session_{stamp}.json")
    if not os.path.exists(logpath):
        return False
    d = json.load(open(logpath))
    rdir = os.path.join(_REPLAY_DIR, stamp)
    os.makedirs(rdir, exist_ok=True)
    for f in glob.glob(os.path.join(rdir, "fig_*.png")):
        os.remove(f)
    script = os.path.join(rdir, "replay.py")
    export_script(d.get("evidence", []), script)
    env = dict(os.environ, CYTOOLS_AGENT_FIG_DIR=rdir)
    with open(os.path.join(rdir, "replay.log"), "w") as logf:
        subprocess.Popen([sys.executable, script], cwd=_REPO, env=env,
                         stdout=logf, stderr=subprocess.STDOUT)
    return True


# human-read
def _replay_figs(stamp):
    """Figures produced so far by a replay, as repo-relative paths."""
    rdir = os.path.join(_REPLAY_DIR, stamp)
    figs = sorted(glob.glob(os.path.join(rdir, "fig_*.png")))
    return [os.path.relpath(f, _REPO) for f in figs]


# the shared look -- used by BOTH the live page and the static export
_STYLE = """<style>
 body{margin:0;font:14px -apple-system,Segoe UI,sans-serif;background:#0f1115;
   color:#d7dae0}
 header{padding:10px 16px;background:#161922;border-bottom:1px solid #262b36;
   font-weight:600}
 #wrap{display:flex;height:calc(100vh - 120px)}
 #actor{padding:9px 16px;background:#11151d;border-bottom:1px solid #262b36;
   font-weight:600}
 #replaybar{padding:7px 16px;background:#13161d;border-bottom:1px solid #262b36;
   font-size:12px;color:#aeb6c4}
 #replaybar select{background:#1b2230;color:#d7dae0;border:1px solid #2a3140;
   border-radius:4px;padding:2px;max-width:55%}
 #replaybar button{font-size:11px;background:#2a3140;color:#cfe;border:0;
   border-radius:4px;padding:2px 9px;cursor:pointer}
 #replayimg img{max-width:520px;margin-top:8px;border:1px solid #262b36;
   border-radius:6px;display:block}
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;
   margin-right:8px;vertical-align:middle}
 .dot.pm{background:#5b8def}.dot.eng{background:#46b17b}
 .dot.none{background:#6b7280}
 .live{animation:pulse 1.1s infinite}
 @keyframes pulse{0%{opacity:1}50%{opacity:.25}100%{opacity:1}}
 .col{overflow:auto;padding:12px 16px}
 #left{width:38%;border-right:1px solid #262b36}
 #right{flex:1}
 h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#7f8794;
   margin:4px 0 10px}
 .ev{border-left:3px solid #3a4151;padding:6px 10px;margin:8px 0;
   background:#161922;border-radius:0 6px 6px 0}
 .ev .k{font-size:11px;text-transform:uppercase;letter-spacing:.05em;
   color:#8b93a3}
 .ev{position:relative}
 .ev:not(:last-child)::after{content:'';position:absolute;left:-2px;top:100%;
   height:8px;width:2px;background:#3a4151}
 .seq{color:#5b8def;font-weight:700;font-variant-numeric:tabular-nums}
 .when{color:#6b7280;font-size:10px;float:right}
 .pm{border-left-color:#5b8def}.eng{border-left-color:#46b17b}
 .ask{border-left-color:#c9883a}.done{border-left-color:#46b17b;
   background:#13241b}
 .obs{border:1px solid #262b36;border-radius:8px;margin:10px 0;overflow:hidden}
 .obs .hd{padding:6px 10px;background:#161922;display:flex;gap:8px;
   align-items:center}
 .obs .rnd{font-size:11px;color:#8b93a3}
 .owhen{color:#6b7280;font-size:10px}
 .rounddiv{font-size:11px;text-transform:uppercase;letter-spacing:.05em;
   color:#46b17b;margin:16px 0 4px;border-bottom:1px solid #25323a;
   padding-bottom:3px}
 .field{padding:6px 10px;border-top:1px solid #1d222c}
 .field.gt{border-left:3px solid #46b17b}
 .field.claim{border-left:3px solid #5a6270}
 .lbl{font-size:11px;text-transform:uppercase;color:#7f8794;margin-bottom:3px}
 .tag{font-size:9px;text-transform:uppercase;letter-spacing:.04em;
   padding:1px 6px;border-radius:8px;margin-left:6px;vertical-align:middle}
 .tag.gt{background:#173026;color:#5fd39a}
 .tag.claim{background:#262b36;color:#9aa3b2}
 .legend{font-size:11px;color:#7f8794;margin:-4px 0 10px}
 .legend b.gt{color:#5fd39a}.legend b.claim{color:#9aa3b2}
 pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12px
   ui-monospace,Menlo,monospace}
 .truth{background:#1b2230}.claim{color:#9aa3b2;font-style:italic}
 .badge{font-size:10px;padding:1px 6px;border-radius:10px;background:#243; }
 .bad{background:#412}
 .obs.val{border-color:#7a5cc0}.badge.val{background:#3a2d5e}
 .val-ev{border-left-color:#7a5cc0}
 .muted{color:#7f8794}
 .cpbtn{float:right;font-size:10px;background:#2a3140;color:#aeb6c4;border:0;
   border-radius:4px;padding:1px 7px;cursor:pointer}
 .cpbtn:hover{background:#39455a;color:#fff}
 button#cpall{font-size:11px;margin-left:10px}
</style>"""

_HEAD = ('<!doctype html><html><head><meta charset="utf-8">\n'
         '<title>orchestrator viewer</title>' + _STYLE + '</head><body>')

# the body skeleton; <!--EXTRA--> is the live replay bar or the baked figures
_BODY = """<header>orchestrator viewer
  <button id="cpall" class="cpbtn" onclick="copyAll()">copy all (md)</button>
  <span id="stat" class="muted"></span></header>
<div id="actor"><span class="dot none"></span>waiting...</div>
<!--EXTRA-->
<div id="wrap">
 <div id="left" class="col"><h2>progress</h2><div id="timeline"></div></div>
 <div id="right" class="col"><h2>evidence (engineer observations)</h2>
   <div class="legend"><b class="gt">ground truth</b> = exact code &amp;
     output, captured by the harness (unfakable).
     <b class="claim">claim</b> = the engineer's own words.</div>
   <div id="evidence"></div></div>
</div>"""

_LIVE_REPLAYBAR = """<div id="replaybar">replay a saved chat:
  <select id="logsel"></select>
  <button onclick="doReplay()">replay</button>
  <span id="replaystatus" class="muted"></span>
  <div id="replayimg"></div>
</div>"""

# the shared renderer -- pure DOM building from LAST.session / LAST.evidence
_RENDER_JS = """
function el(tag, cls, txt){
  const e=document.createElement(tag); if(cls)e.className=cls;
  if(txt!=null)e.textContent=txt; return e;
}
function cp(text){
  const b=el('button','cpbtn','copy');
  b.onclick=()=>navigator.clipboard&&navigator.clipboard.writeText(text);
  return b;
}
function copyAll(){
  const md=[];
  for(const e of LAST.session){
    md.push('## '+e.event);
    md.push(e.text||e.task||e.report||e.message||
            (e.todo?e.todo.join('\\n'):''));
  }
  md.push('\\n# evidence');
  LAST.evidence.forEach((o,i)=>{
    md.push('### #'+(i+1)+' (round '+(o.round||'')+')');
    md.push('- intent: '+o.intent);
    md.push('- ran_code:\\n```python\\n'+o.ran_code+'\\n```');
    md.push('- received_output:\\n```\\n'+o.received_output+'\\n```');
    md.push('- interpretation: '+o.interpretation);
  });
  navigator.clipboard&&navigator.clipboard.writeText(md.join('\\n'));
}
function field(parent, name, text, kind){   // kind: 'gt' or 'claim'
  const f=el('div','field '+kind);
  const lbl=el('div','lbl'); lbl.appendChild(document.createTextNode(name));
  lbl.appendChild(el('span','tag '+kind,
    kind==='gt'?'ground truth':'claim'));
  f.appendChild(lbl);
  const p=el('pre',kind==='gt'?'truth':'claim',
    text==null||text===''?'(none)':String(text));
  f.appendChild(p); parent.appendChild(f);
}
function renderActor(session){
  let a=null;
  for(const e of session) if(e.event==='active') a=e;
  const box=document.getElementById('actor'); box.innerHTML='';
  const who=a?a.who:'none', live=who!=='none';
  const cls=who==='PM'?'pm':who==='engineer'?'eng':'none';
  box.appendChild(el('span','dot '+cls+(live?' live':'')));
  let label;
  if(!a) label='waiting...';
  else if(who==='none') label='idle / done';
  else label=(who==='PM'?'Project manager':'Engineer')+': '+(a.phase||'')
    +(a.round?(' (round '+a.round+')'):'');
  if(live && a && a.t){
    const age=Date.now()/1000 - a.t;
    if(age>240) label+='  (no new step in '+Math.round(age)+'s; stalled?)';
  }
  box.appendChild(document.createTextNode(label));
}
function renderTimeline(events){
  const box=document.getElementById('timeline'); box.innerHTML='';
  const t0=events.length?events[0].t:0;
  let n=0;
  for(const e of events){
    if(e.event==='active') continue;
    n++;
    let cls='ev', k=e.event, body='';
    if(e.event==='question'){cls+=' ask'; k='question'; body=e.text;}
    else if(e.event==='direct_speech'){cls+=' pm'; k='PM direct speech';
      body=e.text;}
    else if(e.event==='plan'){cls+=' pm'; k='PM plan';
      body=(e.todo||[]).map((s,i)=>(i+1)+'. '+s).join('\\n');}
    else if(e.event==='dispatch'){cls+=' pm';
      k='PM dispatch (round '+e.round+')'; body=e.task;}
    else if(e.event==='off_step'){cls+=' ask';
      k='OFF-STEP (round '+e.round+'): answer did not address the step';
      body=e.report||'';}
    else if(e.event==='engineer_report'){cls+=' eng';
      k='engineer report (round '+e.round+', '+e.n_obs+' obs'+
        (e.ok===false?', DID NOT FINISH':'')+')'; body=e.report;}
    else if(e.event==='engineer_timing'){cls+=' eng';
      k='timing (round '+e.round+')';
      body=e.llm_calls+' LLM calls = '+e.llm_s+'s, code = '+e.code_s+'s';}
    else if(e.event==='llm_call'){cls+=' pm';
      k='LLM call: '+e.label+(e.think?' (think)':'');
      body=e.s+'s, prompt '+(e.prompt_chars||0)+' chars';}
    else if(e.event==='step_failed'){cls+=' ask';
      k='STEP FAILED (walk stopped)'; body=e.step;}
    else if(e.event==='respond'){cls+=' done'; k='PM to user'; body=e.message;}
    const when=e.t?('+'+Math.round(e.t-t0)+'s'):'';
    const d=el('div',cls); const kd=el('div','k');
    kd.appendChild(el('span','seq', n+'.'));
    kd.appendChild(document.createTextNode(' '+k+' '));
    kd.appendChild(el('span','when', when));
    kd.appendChild(cp(k+'\\n'+body)); d.appendChild(kd);
    d.appendChild(el('pre',null,body)); box.appendChild(d);
  }
}
function renderEvidence(obs, t0){
  const box=document.getElementById('evidence'); box.innerHTML='';
  let lastRound=null;
  obs.forEach((o,i)=>{
    if(o.round!==lastRound){           // round divider lines up with dispatch
      lastRound=o.round;
      box.appendChild(el('div','rounddiv','round '+o.round+' (engineer)'));
    }
    const c=el('div','obs'+(o.kind==='validation'?' val':''));
    const hd=el('div','hd');
    hd.appendChild(el('span','seq','#'+(i+1)));
    const when=o.t?('+'+Math.round(o.t-t0)+'s'):'';
    hd.appendChild(el('span','owhen', when));
    if(o.kind==='validation')
      hd.appendChild(el('span','badge val','VALIDATION'));
    hd.appendChild(el('span','badge'+(o.valid_python?'':' bad'),
      o.valid_python?'valid python':'INVALID'));
    hd.appendChild(cp('intent: '+o.intent+'\\nran_code:\\n'+o.ran_code+
      '\\nreceived_output:\\n'+o.received_output+
      '\\ninterpretation: '+o.interpretation));
    c.appendChild(hd);
    field(c,'intent',o.intent,'claim');
    field(c,'ran_code',o.ran_code,'gt');
    field(c,'received_output',o.received_output,'gt');
    field(c,'interpretation',o.interpretation,'claim');
    box.appendChild(c);
  });
}
"""

# live bootstrap: poll /data and re-render; offer log replay over the server
_LIVE_TAIL = """
async function tick(){
  try{
    const r=await fetch('/data'); const d=await r.json(); LAST=d;
    const t0=d.session.length?d.session[0].t:0;
    renderActor(d.session); renderTimeline(d.session);
    renderEvidence(d.evidence, t0);
    document.getElementById('stat').textContent=
      ' -- '+d.session.length+' events, '+d.evidence.length+' observations';
  }catch(e){document.getElementById('stat').textContent=' -- waiting...';}
}
async function loadLogs(){
  const r=await fetch('/logs'); const logs=await r.json();
  const sel=document.getElementById('logsel'); sel.innerHTML='';
  for(const L of logs){
    const o=document.createElement('option'); o.value=L.stamp;
    o.textContent=L.stamp+'  '+L.question; sel.appendChild(o);
  }
  if(!logs.length) sel.innerHTML='<option>(no saved logs yet)</option>';
}
let replayTimer=null;
async function doReplay(){
  const stamp=document.getElementById('logsel').value;
  const st=document.getElementById('replaystatus');
  const box=document.getElementById('replayimg');
  if(!stamp||stamp.startsWith('(')){return;}
  st.textContent=' replaying (re-running the saved code, no LLM)...';
  box.innerHTML='';
  await fetch('/replay?stamp='+encodeURIComponent(stamp));
  if(replayTimer)clearInterval(replayTimer);
  let tries=0;
  replayTimer=setInterval(async()=>{
    tries++;
    const r=await fetch('/replay_figs?stamp='+encodeURIComponent(stamp));
    const d=await r.json();
    if(d.figs && d.figs.length){
      clearInterval(replayTimer); st.textContent=' done';
      box.innerHTML='';
      for(const f of d.figs){
        const img=document.createElement('img');
        img.src='/file?p='+encodeURIComponent(f)+'&t='+Date.now();
        box.appendChild(img);
      }
    } else if(tries>150){ clearInterval(replayTimer);
      st.textContent=' (no figure after ~5 min -- see replay.log)'; }
  }, 2000);
}
loadLogs();
setInterval(tick,1500); tick();
"""

# static bootstrap: data is baked in, render once (no polling, no server)
_STATIC_TAIL = """
renderActor(LAST.session);
renderTimeline(LAST.session);
var t0=LAST.session.length?LAST.session[0].t:0;
renderEvidence(LAST.evidence, t0);
document.getElementById('stat').textContent=
  ' -- '+LAST.session.length+' events, '+LAST.evidence.length+' observations';
"""

_PAGE = (_HEAD + _BODY.replace("<!--EXTRA-->", _LIVE_REPLAYBAR)
         + "<script>\nvar LAST={session:[],evidence:[]};\n"
         + _RENDER_JS + _LIVE_TAIL + "</script></body></html>")


# static export -- one self-contained file anyone can open
# --------------------------------------------------------
# human-read
def export_html(stamp, out_path=None):
    """Bake a saved session into ONE self-contained .html (data + figures
    embedded as base64), so anyone can open it in a browser with no server and
    no Python. Same look/renderer as the live viewer; the replay bar is
    replaced by the run's saved figures. `stamp` is a session stamp or a path
    to a session_*.json. Returns the written path."""
    logpath = stamp if str(stamp).endswith(".json") else \
        os.path.join(LOG_DIR, f"session_{stamp}.json")
    d = json.load(open(logpath))
    data = {"session": d.get("session", []), "evidence": d.get("evidence", [])}
    imgs = []
    for f in d.get("figures", []):
        p = f if os.path.isabs(f) else os.path.join(_REPO, f)
        if os.path.isfile(p):
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            imgs.append('<img src="data:image/png;base64,%s">' % b64)
    figs = ('<div id="replaybar">saved figures:<div id="replayimg">'
            + "".join(imgs) + "</div></div>") if imgs else ""
    # escape </ so embedded code/output can never close the <script> early
    blob = json.dumps(data).replace("</", "<\\/")
    page = (_HEAD + _BODY.replace("<!--EXTRA-->", figs)
            + "<script>\nvar LAST=" + blob + ";\n"
            + _RENDER_JS + _STATIC_TAIL + "</script></body></html>")
    base = os.path.basename(logpath)[:-len(".json")]
    out_path = out_path or os.path.join(LOG_DIR, base + ".html")
    with open(out_path, "w") as fh:
        fh.write(page)
    return out_path


# human-read
def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:   # a half-written final line
                pass
    return out


# human-read
class _Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype):
        body = body if isinstance(body, bytes) else body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(json.dumps(obj), "application/json")

    def _file(self, relpath):
        full = os.path.abspath(os.path.join(_REPO, relpath))
        if not full.startswith(_SCRATCH) or not os.path.isfile(full):
            self.send_error(404)         # only serve files under scratch/
            return
        ctype = "image/png" if full.endswith(".png") else "text/plain"
        with open(full, "rb") as f:
            self._send(f.read(), ctype)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        stamp = q.get("stamp", [""])[0]
        if u.path == "/data":
            self._json({"session": _read_jsonl(SESSION_PATH),
                        "evidence": _read_jsonl(EVIDENCE_PATH)})
        elif u.path == "/logs":
            self._json(_logs())
        elif u.path == "/replay":
            self._json({"started": _start_replay(stamp)})
        elif u.path == "/replay_figs":
            self._json({"figs": _replay_figs(stamp)})
        elif u.path == "/file":
            self._file(q.get("p", [""])[0])
        else:
            self._send(_PAGE, "text/html")

    def log_message(self, *_):   # silence per-request logging
        pass


# human-read
def main():
    args = sys.argv[1:]
    if args and args[0] == "export":          # static export, not the server
        stamp = args[1] if len(args) > 1 else (
            _logs()[0]["stamp"] if _logs() else None)
        if not stamp:
            print("no saved logs to export")
            return
        print("wrote", export_html(stamp))
        return
    port = int(args[0]) if args else 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    print(f"orchestrator viewer at http://127.0.0.1:{port}  (Ctrl-C to stop)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
