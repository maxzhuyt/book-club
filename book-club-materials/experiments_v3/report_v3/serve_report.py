"""Local viewer for REPORT_V3_INTERMEDIATE.md with a text-selection annotation layer.

Run:  python serve_report.py            # picks a free port, binds 127.0.0.1
      python serve_report.py --port 8910

Open the printed URL. Select any text in the report to attach a comment; comments
persist to  report_v3/annotations.json  (one JSON list), which the agent reads back
to act on your feedback. Highlights and the sidebar reload from that file.

Pure stdlib + the `markdown` package. Localhost-only (not exposed to the network).
"""
from __future__ import annotations
import argparse, html, json, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import markdown  # pip install markdown

HERE = Path(__file__).resolve().parent
REPORT_MD = HERE / "REPORT_V3_INTERMEDIATE.md"
ANNOT_FILE = HERE / "annotations.json"
FIG_DIR = HERE / "figures_intermediate"

_lock = threading.Lock()


# ----------------------------- markdown -> html -----------------------------

def render_markdown() -> str:
    text = REPORT_MD.read_text()
    # base python-markdown has no ~~strikethrough~~; convert the (few) spans to <del>.
    text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", text, flags=re.DOTALL)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists",
                                       "toc", "attr_list"])
    return md.convert(text)


# ----------------------------- annotation store -----------------------------

def load_annotations() -> list:
    if not ANNOT_FILE.exists():
        return []
    try:
        return json.loads(ANNOT_FILE.read_text())
    except Exception:
        return []


def save_annotations(items: list) -> None:
    ANNOT_FILE.write_text(json.dumps(items, indent=2, ensure_ascii=False))


def add_annotation(payload: dict) -> dict:
    with _lock:
        items = load_annotations()
        n = max([a.get("n", 0) for a in items], default=0) + 1
        ann = {
            "id": "a%d_%d" % (int(time.time() * 1000), n),
            "n": n,
            "start": int(payload.get("start", 0)),
            "end": int(payload.get("end", 0)),
            "quote": str(payload.get("quote", ""))[:4000],
            "prefix": str(payload.get("prefix", ""))[:80],
            "comment": str(payload.get("comment", ""))[:8000],
            "status": "open",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        items.append(ann)
        save_annotations(items)
        return ann


def update_annotation(aid: str, fields: dict) -> bool:
    with _lock:
        items = load_annotations()
        hit = False
        for a in items:
            if a["id"] == aid:
                if "comment" in fields:
                    a["comment"] = str(fields["comment"])[:8000]
                if "status" in fields and fields["status"] in ("open", "resolved"):
                    a["status"] = fields["status"]
                hit = True
        if hit:
            save_annotations(items)
        return hit


def delete_annotation(aid: str) -> bool:
    with _lock:
        items = load_annotations()
        new = [a for a in items if a["id"] != aid]
        if len(new) != len(items):
            save_annotations(new)
            return True
        return False


# ----------------------------- page template -----------------------------
# No f-strings/format: the CSS+JS are full of braces. We str.replace tokens.

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Book Club V3 — Intermediate Report (annotatable)</title>
<style>
  :root { --ink:#1a1a1a; --muted:#6b7280; --line:#e5e7eb; --hl:#fff3b0; --hl-resolved:#d7f0d7;
          --accent:#2563eb; --bg:#ffffff; --sidebar:#fafafa; }
  * { box-sizing: border-box; }
  body { margin:0; color:var(--ink); background:var(--bg);
         font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
  #wrap { display:grid; grid-template-columns: minmax(0,1fr) 360px; gap:0; align-items:start; }
  #main { padding:32px 44px 120px; max-width:900px; margin:0 auto; }
  #side { position:sticky; top:0; height:100vh; overflow:auto; border-left:1px solid var(--line);
          background:var(--sidebar); padding:16px 16px 80px; }
  @media (max-width: 980px){ #wrap{grid-template-columns:1fr;} #side{position:static;height:auto;border-left:none;border-top:1px solid var(--line);} }
  #content h1{font-size:1.9rem;line-height:1.2;margin:.2em 0 .5em;}
  #content h2{font-size:1.45rem;margin:1.6em 0 .4em;padding-bottom:.2em;border-bottom:1px solid var(--line);}
  #content h3{font-size:1.18rem;margin:1.4em 0 .3em;}
  #content table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92rem;display:block;overflow-x:auto;}
  #content th,#content td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top;}
  #content th{background:#f3f4f6;}
  #content tr:nth-child(even) td{background:#fafafa;}
  #content code{background:#f3f4f6;padding:.1em .35em;border-radius:4px;font-size:.88em;}
  #content pre{background:#f6f8fa;padding:12px;border-radius:8px;overflow:auto;}
  #content img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff;}
  /* figure captions: the italic paragraph immediately after an image paragraph */
  #content p:has(> img){text-align:center;margin-bottom:4px;}
  #content p:has(> img) + p{font-size:0.88rem;color:#555;line-height:1.55;
        max-width:780px;margin:0 auto 1.5em;padding:6px 10px 0;border-top:1px solid var(--line);}
  #content blockquote{border-left:3px solid var(--line);margin:1em 0;padding:.2em 1em;color:var(--muted);}
  #content del{color:var(--muted);}
  mark.annot{background:var(--hl);border-radius:2px;padding:0 .02em;cursor:pointer;
             box-shadow: inset 0 -2px 0 rgba(0,0,0,.06);}
  mark.annot.resolved{background:var(--hl-resolved);}
  mark.annot.focus{outline:2px solid var(--accent);}
  /* floating add button + composer */
  #float{position:absolute;z-index:50;display:none;}
  #float button, .btn{font:inherit;font-size:.85rem;border:1px solid var(--line);background:#fff;
        border-radius:6px;padding:5px 10px;cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,.12);}
  #float button:hover,.btn:hover{border-color:var(--accent);color:var(--accent);}
  #composer{position:absolute;z-index:51;display:none;width:320px;background:#fff;border:1px solid var(--line);
        border-radius:10px;box-shadow:0 6px 24px rgba(0,0,0,.18);padding:10px;}
  #composer .q{font-size:.8rem;color:var(--muted);max-height:64px;overflow:auto;margin-bottom:6px;
        border-left:3px solid var(--hl);padding-left:8px;}
  #composer textarea{width:100%;height:80px;border:1px solid var(--line);border-radius:6px;padding:8px;font:inherit;font-size:.9rem;resize:vertical;}
  #composer .row{display:flex;justify-content:flex-end;gap:8px;margin-top:8px;}
  .btn.primary{background:var(--accent);color:#fff;border-color:var(--accent);}
  /* sidebar */
  #side h2{font-size:1rem;margin:.2em 0 .8em;display:flex;align-items:center;gap:8px;}
  #side .count{background:var(--accent);color:#fff;border-radius:10px;padding:0 8px;font-size:.78rem;}
  .acard{border:1px solid var(--line);border-radius:8px;background:#fff;padding:10px;margin-bottom:10px;}
  .acard.resolved{opacity:.62;}
  .acard .n{font-weight:600;color:var(--accent);font-size:.8rem;}
  .acard .quote{font-size:.82rem;color:#374151;border-left:3px solid var(--hl);padding-left:8px;margin:6px 0;
        max-height:80px;overflow:auto;white-space:pre-wrap;}
  .acard .cmt{font-size:.9rem;white-space:pre-wrap;}
  .acard .meta{font-size:.72rem;color:var(--muted);margin-top:6px;display:flex;gap:10px;justify-content:space-between;align-items:center;}
  .acard .acts{display:flex;gap:6px;}
  .acard .acts a{cursor:pointer;color:var(--muted);text-decoration:underline;}
  .acard .acts a:hover{color:var(--accent);}
  #toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;}
  #empty{color:var(--muted);font-size:.9rem;}
  .hint{position:fixed;bottom:14px;left:14px;background:#111;color:#fff;font-size:.8rem;padding:8px 12px;
        border-radius:8px;opacity:.85;z-index:60;}
</style>
</head>
<body>
<div id="wrap">
  <div id="main"><div id="content">__CONTENT__</div></div>
  <div id="side">
    <h2>Annotations <span class="count" id="count">0</span></h2>
    <div id="toolbar">
      <button class="btn" id="filterBtn">Show: all</button>
      <a class="btn" id="exportBtn" href="/api/annotations" download="annotations.json">Export JSON</a>
    </div>
    <div id="list"></div>
    <div id="empty">No annotations yet. Select text in the report to add one.</div>
  </div>
</div>

<div id="float"><button id="addBtn">💬 Add note</button></div>
<div id="composer">
  <div class="q" id="composerQuote"></div>
  <textarea id="composerText" placeholder="Your comment / thing for the agent to review…"></textarea>
  <div class="row"><button class="btn" id="cancelBtn">Cancel</button><button class="btn primary" id="saveBtn">Save</button></div>
</div>
<div class="hint" id="hint">Select text → “Add note”. Saved to annotations.json</div>

<script>
const content = document.getElementById('content');
const floatEl = document.getElementById('float');
const composer = document.getElementById('composer');
const composerText = document.getElementById('composerText');
const composerQuote = document.getElementById('composerQuote');
let pending = null;       // {start,end,quote,prefix}
let annotations = [];
let filter = 'all';

// ---- text offset helpers (offsets are into content.textContent; stable across <mark> inserts) ----
function preTextLen(node, offset){
  const r = document.createRange();
  r.selectNodeContents(content);
  r.setEnd(node, offset);
  return r.toString().length;
}
function nodeAtOffset(target){
  const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT);
  let acc = 0, node;
  while ((node = walker.nextNode())){
    const len = node.nodeValue.length;
    if (acc + len >= target) return {node, offset: target - acc};
    acc += len;
  }
  return null;
}
function rangeFromOffsets(start, end){
  const a = nodeAtOffset(start), b = nodeAtOffset(end);
  if (!a || !b) return null;
  const r = document.createRange();
  try { r.setStart(a.node, a.offset); r.setEnd(b.node, b.offset); } catch(e){ return null; }
  return r;
}
function plainText(){ return content.textContent; }

// ---- highlighting ----
function wrapRange(range, ann){
  const nodes = [];
  // root at an element: a TreeWalker rooted at a Text node yields nothing,
  // which would drop the (very common) single-text-node selection.
  let root = range.commonAncestorContainer;
  if (root.nodeType === Node.TEXT_NODE) root = root.parentNode;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walker.nextNode())){ if (range.intersectsNode(n)) nodes.push(n); }
  nodes.forEach(node => {
    let s = 0, e = node.nodeValue.length;
    if (node === range.startContainer) s = range.startOffset;
    if (node === range.endContainer) e = range.endOffset;
    if (s >= e) return;
    const r = document.createRange();
    try { r.setStart(node, s); r.setEnd(node, e); } catch(err){ return; }
    const m = document.createElement('mark');
    m.className = 'annot' + (ann.status === 'resolved' ? ' resolved' : '');
    m.dataset.id = ann.id;
    m.title = '#' + ann.n + ': ' + (ann.comment || '');
    try { r.surroundContents(m); } catch(err){ /* skip pathological spans */ }
  });
}
function clearHighlights(){
  content.querySelectorAll('mark.annot').forEach(m => {
    const parent = m.parentNode;
    while (m.firstChild) parent.insertBefore(m.firstChild, m);
    parent.removeChild(m);
    parent.normalize();
  });
}
function renderHighlights(){
  clearHighlights();
  // apply in document order; offsets are stable because marks add no text
  [...annotations].sort((a,b)=>a.start-b.start).forEach(ann => {
    if (filter==='open' && ann.status!=='open') return;
    let r = rangeFromOffsets(ann.start, ann.end);
    // resilience: if the quote drifted, re-find it by text search
    if ((!r || r.toString().trim() !== (ann.quote||'').trim()) && ann.quote){
      const idx = plainText().indexOf(ann.quote, Math.max(0, ann.start - 60));
      const idx2 = idx >= 0 ? idx : plainText().indexOf(ann.quote);
      if (idx2 >= 0) r = rangeFromOffsets(idx2, idx2 + ann.quote.length);
    }
    if (r) wrapRange(r, ann);
  });
}

// ---- sidebar ----
function renderSidebar(){
  const list = document.getElementById('list');
  const empty = document.getElementById('empty');
  const visible = annotations.filter(a => filter==='all' || a.status==='open');
  document.getElementById('count').textContent = annotations.length;
  list.innerHTML = '';
  empty.style.display = annotations.length ? 'none' : 'block';
  visible.sort((a,b)=>a.start-b.start).forEach(ann => {
    const card = document.createElement('div');
    card.className = 'acard' + (ann.status==='resolved'?' resolved':'');
    card.dataset.id = ann.id;
    card.innerHTML =
      '<div class="n">#'+ann.n+'</div>'+
      '<div class="quote">'+escapeHtml(ann.quote||'')+'</div>'+
      '<div class="cmt">'+escapeHtml(ann.comment||'')+'</div>'+
      '<div class="meta"><span>'+ann.created+'</span>'+
        '<span class="acts">'+
          '<a data-act="goto">locate</a>'+
          '<a data-act="toggle">'+(ann.status==='resolved'?'reopen':'resolve')+'</a>'+
          '<a data-act="del">delete</a>'+
        '</span></div>';
    card.querySelector('[data-act=goto]').onclick = ()=>gotoAnn(ann.id);
    card.querySelector('[data-act=toggle]').onclick = ()=>toggleAnn(ann);
    card.querySelector('[data-act=del]').onclick = ()=>delAnn(ann.id);
    list.appendChild(card);
  });
}
function escapeHtml(s){ return s.replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

function gotoAnn(id){
  const m = content.querySelector('mark.annot[data-id="'+id+'"]');
  if (m){ m.scrollIntoView({behavior:'smooth', block:'center'});
    content.querySelectorAll('mark.focus').forEach(x=>x.classList.remove('focus'));
    content.querySelectorAll('mark.annot[data-id="'+id+'"]').forEach(x=>x.classList.add('focus'));
    setTimeout(()=>content.querySelectorAll('mark.focus').forEach(x=>x.classList.remove('focus')), 2000);
  }
}

// ---- API ----
async function loadAnnotations(){
  const r = await fetch('/api/annotations'); annotations = await r.json();
  renderHighlights(); renderSidebar();
}
async function saveAnnotation(payload){
  const r = await fetch('/api/annotations', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  return r.json();
}
async function toggleAnn(ann){
  await fetch('/api/annotations/'+ann.id, {method:'PATCH', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({status: ann.status==='resolved'?'open':'resolved'})});
  await loadAnnotations();
}
async function delAnn(id){
  if (!confirm('Delete this annotation?')) return;
  await fetch('/api/annotations/'+id, {method:'DELETE'});
  await loadAnnotations();
}

// ---- selection -> floating button -> composer ----
content.addEventListener('mouseup', (e)=>{
  setTimeout(()=>{
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount===0){ floatEl.style.display='none'; return; }
    const range = sel.getRangeAt(0);
    if (!content.contains(range.commonAncestorContainer)){ floatEl.style.display='none'; return; }
    const text = sel.toString();
    if (!text.trim()){ floatEl.style.display='none'; return; }
    const start = preTextLen(range.startContainer, range.startOffset);
    const end = preTextLen(range.endContainer, range.endOffset);
    pending = {start:Math.min(start,end), end:Math.max(start,end), quote:text,
               prefix: plainText().slice(Math.max(0,Math.min(start,end)-40), Math.min(start,end))};
    const rect = range.getBoundingClientRect();
    floatEl.style.left = (window.scrollX + rect.left) + 'px';
    floatEl.style.top  = (window.scrollY + rect.bottom + 6) + 'px';
    floatEl.style.display = 'block';
  }, 1);
});
document.getElementById('addBtn').onclick = ()=>{
  if (!pending) return;
  floatEl.style.display='none';
  composerQuote.textContent = pending.quote.length>240 ? pending.quote.slice(0,240)+'…' : pending.quote;
  composerText.value='';
  composer.style.left = floatEl.style.left;
  composer.style.top = floatEl.style.top;
  composer.style.display='block';
  composerText.focus();
};
document.getElementById('cancelBtn').onclick = ()=>{ composer.style.display='none'; pending=null; };
document.getElementById('saveBtn').onclick = async ()=>{
  if (!pending) return;
  const comment = composerText.value.trim();
  if (!comment){ composerText.focus(); return; }
  await saveAnnotation({...pending, comment});
  composer.style.display='none'; pending=null;
  window.getSelection().removeAllRanges();
  await loadAnnotations();
};
composerText.addEventListener('keydown', (e)=>{
  if ((e.metaKey||e.ctrlKey) && e.key==='Enter') document.getElementById('saveBtn').click();
  if (e.key==='Escape') document.getElementById('cancelBtn').click();
});
document.addEventListener('mousedown', (e)=>{
  if (!floatEl.contains(e.target)) floatEl.style.display='none';
});
content.addEventListener('click', (e)=>{
  const m = e.target.closest('mark.annot');
  if (m) gotoCard(m.dataset.id);
});
function gotoCard(id){
  const card = document.querySelector('.acard[data-id="'+id+'"]');
  if (card){ card.scrollIntoView({behavior:'smooth',block:'center'}); card.style.outline='2px solid var(--accent)';
    setTimeout(()=>card.style.outline='', 1500); }
}
document.getElementById('filterBtn').onclick = (e)=>{
  filter = filter==='all' ? 'open' : 'all';
  e.target.textContent = 'Show: ' + filter;
  renderHighlights(); renderSidebar();
};
setTimeout(()=>{document.getElementById('hint').style.display='none';}, 6000);
loadAnnotations();
</script>
</body>
</html>"""


def build_page() -> bytes:
    return PAGE.replace("__CONTENT__", render_markdown()).encode("utf-8")


# ----------------------------- http handler -----------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body=b"", ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, build_page())
        elif self.path == "/api/annotations":
            self._json(200, load_annotations())
        elif self.path.startswith("/figures_intermediate/"):
            name = self.path.split("/figures_intermediate/", 1)[1].split("?")[0]
            f = (FIG_DIR / name).resolve()
            if f.parent == FIG_DIR and f.exists():
                self._send(200, f.read_bytes(), "image/png")
            else:
                self._send(404, b"not found")
        else:
            self._send(404, b"not found")

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_POST(self):
        if self.path == "/api/annotations":
            try:
                ann = add_annotation(self._read_body())
                self._json(200, ann)
            except Exception as e:
                self._json(400, {"error": str(e)})
        else:
            self._send(404, b"not found")

    def do_PATCH(self):
        m = re.match(r"^/api/annotations/([^/]+)$", self.path)
        if m:
            ok = update_annotation(m.group(1), self._read_body())
            self._json(200 if ok else 404, {"ok": ok})
        else:
            self._send(404, b"not found")

    def do_DELETE(self):
        m = re.match(r"^/api/annotations/([^/]+)$", self.path)
        if m:
            ok = delete_annotation(m.group(1))
            self._json(200 if ok else 404, {"ok": ok})
        else:
            self._send(404, b"not found")


def main():
    global REPORT_MD, ANNOT_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0,
                    help="port (default: first free among 8910,8000,8765,5005,0)")
    ap.add_argument("--report", default=str(REPORT_MD),
                    help="markdown report to serve (default: REPORT_V3_INTERMEDIATE.md)")
    ap.add_argument("--annotations", default=None,
                    help="annotations JSON path (default: per-report)")
    args = ap.parse_args()

    REPORT_MD = Path(args.report).resolve()
    if args.annotations:
        ANNOT_FILE = Path(args.annotations).resolve()
    elif REPORT_MD.name == "REPORT_V3_INTERMEDIATE.md":
        ANNOT_FILE = HERE / "annotations.json"          # preserve the existing file
    else:
        ANNOT_FILE = REPORT_MD.with_suffix(".annotations.json")

    candidates = [args.port] if args.port else [8910, 8000, 8765, 5005, 0]
    httpd = None
    for p in candidates:
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit("could not bind any candidate port")
    port = httpd.server_address[1]
    print("Serving report at:  http://127.0.0.1:%d/" % port, flush=True)
    print("Annotations file:   %s" % ANNOT_FILE, flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
