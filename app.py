import math
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline as hf_pipeline
from captum.attr import LayerIntegratedGradients

# ── Models ────────────────────────────────────────────────────────────────────
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "Izziemirg/medsentinel-adr-deberta"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model     = model.to(DEVICE).eval()

def _fwd(input_ids, attention_mask):
    return model(input_ids=input_ids,
                 attention_mask=attention_mask).logits[:, 1]

lig = LayerIntegratedGradients(_fwd, model.deberta.embeddings.word_embeddings)

ner_pipe = hf_pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1,
)

# ── Core helpers ──────────────────────────────────────────────────────────────
def encode(text):
    enc = tokenizer(text, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=256)
    return {k: v.to(DEVICE) for k, v in enc.items()}

def get_prediction(text):
    enc = encode(text)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred  = int(np.argmax(probs))
    return pred, float(probs[0]), float(probs[1])

def get_ig_scores(text, n_steps=20):
    enc            = encode(text)
    input_ids      = enc["input_ids"].cpu()
    attention_mask = enc["attention_mask"].cpu()
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
    baseline       = torch.full_like(input_ids, tokenizer.pad_token_id or 0)

    model.cpu()
    
    attributions, _ = lig.attribute(
        inputs=input_ids, baselines=baseline,
        additional_forward_args=(attention_mask),
        n_steps=n_steps, return_convergence_delta=True,
        internal_batch_size=4,
    )

    model.to(DEVICE)
    
    scores  = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    seq_len = int(attention_mask[0].sum().item())
    tokens  = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())[:seq_len]
    return tokens, scores[:seq_len]

def merge_subword_tokens(tokens, scores):
    SPECIAL = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
    mt, ms  = [], []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in SPECIAL:
            i += 1
            continue
        word  = tok.lstrip("▁")
        score = float(scores[i])
        while (i + 1 < len(tokens)
               and tokens[i + 1] not in SPECIAL
               and not tokens[i + 1].startswith("▁")):
            i += 1
            word  += tokens[i]
            score += float(scores[i])
        mt.append(word)
        ms.append(score)
        i += 1
    return mt, np.array(ms)

# ── Logo SVG ──────────────────────────────────────────────────────────────────
LOGO_SVG = """
<svg width='52' height='60' viewBox='0 0 38 44' fill='none' xmlns='http://www.w3.org/2000/svg'>
  <path d='M19 2 L35 9.5 L35 25.5 C35 35.5 27.5 41.5 19 43.5 C10.5 41.5 3 35.5 3 25.5 L3 9.5 Z'
        stroke='#2563eb' stroke-width='2.2' fill='none' stroke-linejoin='round'/>
  <line x1='7'  y1='22' x2='15' y2='22' stroke='#2563eb' stroke-width='2.2' stroke-linecap='round'/>
  <line x1='23' y1='22' x2='31' y2='22' stroke='#2563eb' stroke-width='2.2' stroke-linecap='round'/>
  <line x1='19' y1='10' x2='19' y2='18' stroke='#2563eb' stroke-width='2.2' stroke-linecap='round'/>
  <line x1='19' y1='26' x2='19' y2='34' stroke='#2563eb' stroke-width='2.2' stroke-linecap='round'/>
  <circle cx='19' cy='22' r='3.5' fill='#22c55e'/>
</svg>"""

# ── Gauge ─────────────────────────────────────────────────────────────────────
def build_gauge_html(prob_severe, prob_mild):
    cx, cy, r, sw = 100, 92, 68, 18

    def pt(deg):
        rad = math.radians(deg)
        return cx + r * math.cos(rad), cy - r * math.sin(rad)

    x0,  y0  = pt(180)
    x04, y04 = pt(108)
    x07, y07 = pt(54)
    x1,  y1  = pt(0)
    x50, y50 = pt(90)

    ir, or_ = r - sw // 2, r + sw // 2
    def tick(deg):
        rad = math.radians(deg)
        return (cx + ir * math.cos(rad), cy - ir * math.sin(rad),
                cx + or_ * math.cos(rad), cy - or_ * math.sin(rad))
    t04, t07 = tick(108), tick(54)

    nr   = r - sw // 2 - 3
    ndeg = 180 - prob_severe * 180
    nx   = cx + nr * math.cos(math.radians(ndeg))
    ny   = cy - nr * math.sin(math.radians(ndeg))

    if prob_severe >= 0.70:
        sc, bb, bd, sl = "#dc2626", "#fef2f2", "#fecaca", "SEVERE"
    elif prob_severe >= 0.40:
        sc, bb, bd, sl = "#d97706", "#fffbeb", "#fde68a", "UNCERTAIN"
    else:
        sc, bb, bd, sl = "#16a34a", "#f0fdf4", "#bbf7d0", "MILD"

    gap = abs(prob_severe - prob_mild)
    cl  = "High"    if gap > 0.60 else ("Medium" if gap > 0.25 else "Low")
    cc  = "#16a34a" if gap > 0.60 else ("#d97706" if gap > 0.25 else "#dc2626")

    margin = max(0.03, min(0.12, 0.04 + 0.10 * (1.0 - abs(prob_severe - 0.5) * 2)))
    ci_lo  = round(max(0.01, prob_severe - margin), 2)
    ci_hi  = round(min(0.99, prob_severe + margin), 2)

    return f"""
<div style='font-family:system-ui,sans-serif;background:white;
            border:1px solid #bbf7d0;border-top:3px solid #22c55e;
            border-radius:14px;padding:20px 22px;box-shadow:0 2px 12px rgba(0,0,0,0.07);
            box-sizing:border-box'>
  <div style='margin-bottom:14px'>
    <div style='font-size:10px;font-weight:700;text-transform:uppercase;
                letter-spacing:0.1em;color:#1e3a5f;margin-bottom:6px'>Severity Assessment</div>
    <div style='display:inline-block;background:{bb};border:1.5px solid {bd};
                border-radius:7px;padding:4px 12px'>
      <span style='font-size:1.2rem;font-weight:800;color:{sc}'>{sl}</span>
    </div>
  </div>
  <svg viewBox='0 0 200 108' xmlns='http://www.w3.org/2000/svg'
       style='width:100%;max-width:260px;display:block;margin:0 auto'>
    <path d='M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {x50:.1f} {y50:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f}'
          fill='none' stroke='#f1f5f9' stroke-width='{sw}' stroke-linecap='butt'/>
    <path d='M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {x04:.1f} {y04:.1f}'
          fill='none' stroke='#22c55e' stroke-width='{sw}' stroke-linecap='butt'/>
    <path d='M {x04:.1f} {y04:.1f} A {r} {r} 0 0 1 {x07:.1f} {y07:.1f}'
          fill='none' stroke='#f59e0b' stroke-width='{sw}' stroke-linecap='butt'/>
    <path d='M {x07:.1f} {y07:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f}'
          fill='none' stroke='#ef4444' stroke-width='{sw}' stroke-linecap='butt'/>
    <line x1='{t04[0]:.1f}' y1='{t04[1]:.1f}' x2='{t04[2]:.1f}' y2='{t04[3]:.1f}'
          stroke='white' stroke-width='2.5'/>
    <line x1='{t07[0]:.1f}' y1='{t07[1]:.1f}' x2='{t07[2]:.1f}' y2='{t07[3]:.1f}'
          stroke='white' stroke-width='2.5'/>
    <line x1='{cx}' y1='{cy}' x2='{nx:.1f}' y2='{ny:.1f}'
          stroke='#94a3b8' stroke-width='5' stroke-linecap='round' opacity='0.25'/>
    <line x1='{cx}' y1='{cy}' x2='{nx:.1f}' y2='{ny:.1f}'
          stroke='#1e293b' stroke-width='2.5' stroke-linecap='round'/>
    <circle cx='{cx}' cy='{cy}' r='6' fill='#1e293b'/>
    <circle cx='{cx}' cy='{cy}' r='2.5' fill='white'/>
    <text x='{x0:.0f}' y='105' font-size='8.5' fill='#94a3b8' text-anchor='middle'
          font-family='system-ui,sans-serif'>0</text>
    <text x='{x1:.0f}' y='105' font-size='8.5' fill='#94a3b8' text-anchor='middle'
          font-family='system-ui,sans-serif'>1.0</text>
  </svg>
  <div style='display:flex;justify-content:space-around;margin-top:12px;
              padding-top:12px;border-top:1px solid #f1f5f9'>
    <div style='text-align:center'>
      <div style='font-size:9px;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.08em;color:#1e3a5f'>Severity Prob.</div>
      <div style='font-weight:800;color:#dc2626;font-size:1.05rem;margin-top:2px'>{prob_severe:.3f}</div>
    </div>
    <div style='text-align:center'>
      <div style='font-size:9px;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.08em;color:#1e3a5f'>Mild Prob.</div>
      <div style='font-weight:800;color:#16a34a;font-size:1.05rem;margin-top:2px'>{prob_mild:.3f}</div>
    </div>
    <div style='text-align:center'>
      <div style='font-size:9px;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.08em;color:#1e3a5f'>Est. CI</div>
      <div style='font-weight:600;color:#475569;font-size:0.88rem;margin-top:2px'>[{ci_lo:.2f}, {ci_hi:.2f}]</div>
    </div>
  </div>
  <div style='text-align:center;margin-top:8px;font-size:9px;color:#cbd5e1;font-style:italic'>
    CI estimated for prototype display · not statistically rigorous
  </div>
</div>"""


# ── Model Confidence Tile ─────────────────────────────────────────────────────
def build_confidence_html(prob_severe, prob_mild):
    gap = abs(prob_severe - prob_mild)
    cl  = "High"    if gap > 0.60 else ("Medium" if gap > 0.25 else "Low")
    cc  = "#16a34a" if gap > 0.60 else ("#d97706" if gap > 0.25 else "#dc2626")

    # Pointer position: gap 0→1 maps to 0→100%, clamped to keep inside bar
    pct = round(min(max(gap * 100, 4), 96), 1)

    low_w    = "#1e3a5f" if cl == "Low"    else "#94a3b8"
    medium_w = "#1e3a5f" if cl == "Medium" else "#94a3b8"
    high_w   = "#1e3a5f" if cl == "High"   else "#94a3b8"
    low_sz   = "12px" if cl == "Low"    else "10px"
    medium_sz= "12px" if cl == "Medium" else "10px"
    high_sz  = "12px" if cl == "High"   else "10px"

    return f"""
<div style='font-family:system-ui,sans-serif;background:white;
            border:1px solid #bbf7d0;border-top:3px solid #22c55e;
            border-radius:14px;padding:16px 22px 18px;
            box-shadow:0 2px 12px rgba(0,0,0,0.07);box-sizing:border-box;margin-top:12px'>
  <div style='font-size:10px;font-weight:700;text-transform:uppercase;
              letter-spacing:0.1em;color:#1e3a5f;margin-bottom:14px'>Model Confidence</div>
  <div style='position:relative;padding-top:18px;margin:0 6px'>
    <!-- Triangle pointer above bar -->
    <div style='position:absolute;top:0;left:{pct}%;transform:translateX(-50%)'>
      <div style='width:0;height:0;
                  border-left:7px solid transparent;
                  border-right:7px solid transparent;
                  border-top:10px solid {cc}'></div>
    </div>
    <!-- Gradient bar -->
    <div style='height:12px;border-radius:6px;
                background:linear-gradient(to right,#ef4444 0%,#f59e0b 60%,#22c55e 100%)'></div>
    <!-- Dot on bar at pointer position -->
    <div style='position:absolute;top:18px;left:{pct}%;
                transform:translate(-50%,0);margin-top:0px'>
      <div style='width:14px;height:14px;border-radius:50%;background:white;
                  border:2.5px solid {cc};box-shadow:0 1px 5px rgba(0,0,0,0.18);
                  margin-top:-13px'></div>
    </div>
  </div>
  <!-- Labels -->
  <div style='display:flex;justify-content:space-between;margin-top:10px;padding:0 2px'>
    <span style='font-size:{low_sz};font-weight:700;color:{low_w};
                 font-family:system-ui,sans-serif;text-transform:uppercase;
                 letter-spacing:0.06em'>Low</span>
    <span style='font-size:{medium_sz};font-weight:700;color:{medium_w};
                 font-family:system-ui,sans-serif;text-transform:uppercase;
                 letter-spacing:0.06em'>Medium</span>
    <span style='font-size:{high_sz};font-weight:700;color:{high_w};
                 font-family:system-ui,sans-serif;text-transform:uppercase;
                 letter-spacing:0.06em'>High</span>
  </div>
</div>"""

# ── Heatmap ───────────────────────────────────────────────────────────────────
def build_heatmap_html(tokens, scores):
    mt, ms  = merge_subword_tokens(tokens, scores)
    abs_max = max(abs(ms).max(), 1e-9)
    norm    = ms / abs_max
    parts   = []
    for tok, val in zip(mt, norm):
        alpha = max(0.12, round(min(abs(float(val)), 1.0), 2))
        if val > 0:
            bg   = f"rgba(220,50,50,{alpha})"
            bord = f"rgba(180,30,30,{min(alpha + 0.15, 1):.2f})"
        else:
            bg   = f"rgba(37,99,235,{alpha})"
            bord = f"rgba(29,78,216,{min(alpha + 0.15, 1):.2f})"
        tc = "white" if alpha > 0.45 else "#1e293b"
        parts.append(
            f'<span style="background:{bg};border:1px solid {bord};color:{tc};'
            f'border-radius:5px;padding:4px 9px;margin:3px 2px;font-family:monospace;'
            f'font-size:13.5px;display:inline-block;font-weight:500">{tok}</span>'
        )
    return (
        '<div style="font-family:system-ui,sans-serif;background:white;'
        'border:1px solid #bfdbfe;border-top:3px solid #2563eb;'
        'border-radius:14px;padding:16px 18px;box-shadow:0 2px 12px rgba(0,0,0,0.07);'
        'box-sizing:border-box">'
        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;'
        'color:#1e3a5f;margin-bottom:10px">Clinical Evidence Highlighting</div>'
        f'<div style="line-height:2.6">{" ".join(parts)}</div>'
        '<div style="display:flex;gap:14px;margin-top:12px;padding-top:10px;'
        'border-top:1px solid #f1f5f9;font-size:11px;color:#64748b;flex-wrap:wrap;align-items:center">'
        '<div style="display:flex;align-items:center;gap:6px">'
        '<div style="width:22px;height:11px;background:rgba(220,50,50,0.72);border-radius:3px;'
        'border:1px solid rgba(180,30,30,0.5)"></div>'
        '<span>→ <strong>Severe</strong></span></div>'
        '<div style="display:flex;align-items:center;gap:6px">'
        '<div style="width:22px;height:11px;background:rgba(37,99,235,0.72);border-radius:3px;'
        'border:1px solid rgba(29,78,216,0.5)"></div>'
        '<span>→ <strong>Mild</strong></span></div>'
        '<div style="color:#94a3b8;font-style:italic;font-size:10px;margin-left:auto">'
        'Opacity = magnitude</div></div></div>'
    )

# ── NER ───────────────────────────────────────────────────────────────────────
def build_ner_html(text, entities):
    COLORS = {
        "Sign_symptom":         ("#fef2f2", "#dc2626",  "#fca5a5"),
        "Disease_disorder":     ("#fff7ed", "#ea580c",  "#fed7aa"),
        "Medication":           ("#eff6ff", "#2563eb",  "#bfdbfe"),
        "Clinical_event":       ("#f5f3ff", "#7c3aed",  "#ddd6fe"),
        "Biological_structure": ("#f0fdf4", "#16a34a",  "#bbf7d0"),
        "Date":                 ("#f8fafc", "#64748b",  "#e2e8f0"),
        "Age":                  ("#f8fafc", "#64748b",  "#e2e8f0"),
    }
    DEFAULT = ("#fafafa", "#374151", "#e5e7eb")

    if not entities:
        return (
            '<div style="font-family:system-ui,sans-serif;background:white;'
            'border:1px solid #bfdbfe;border-top:3px solid #2563eb;'
            'border-radius:14px;padding:16px 18px;box-shadow:0 2px 12px rgba(0,0,0,0.07);'
            'box-sizing:border-box;'
            'color:#94a3b8;font-style:italic;font-size:0.88rem">'
            'No biomedical entities detected.</div>'
        )

    result = ""
    cursor = 0
    for ent in sorted(entities, key=lambda e: e["start"]):
        start, end = ent["start"], ent["end"]
        label = ent["entity_group"]
        score = ent["score"]
        bg, fg, border = COLORS.get(label, DEFAULT)
        result += text[cursor:start]
        result += (
            f'<span style="background:{bg};color:{fg};border:1px solid {border};'
            f'border-radius:4px;padding:2px 6px;margin:0 1px;font-weight:600;'
            f'font-size:13px;display:inline-block;white-space:nowrap" '
            f'title="{label} (confidence: {score:.2f})">'
            f'{text[start:end]}'
            f'<sup style="font-size:9px;margin-left:3px;opacity:0.75">'
            f'{label.replace("_", " ")}</sup>'
            f'</span>'
        )
        cursor = end
    result += text[cursor:]

    seen = {e["entity_group"] for e in entities}
    legend_items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'margin-right:10px;margin-bottom:4px">'
        f'<span style="background:{COLORS.get(k, DEFAULT)[0]};color:{COLORS.get(k, DEFAULT)[1]};'
        f'border:1px solid {COLORS.get(k, DEFAULT)[2]};border-radius:3px;padding:1px 6px;'
        f'font-size:10px;font-weight:600">{k.replace("_", " ")}</span></span>'
        for k in COLORS if k in seen
    )

    return (
        '<div style="font-family:system-ui,sans-serif;background:white;'
        'border:1px solid #bfdbfe;border-top:3px solid #2563eb;'
        'border-radius:14px;padding:16px 18px;box-shadow:0 2px 12px rgba(0,0,0,0.07);'
        'box-sizing:border-box">'
        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;'
        'color:#1e3a5f;margin-bottom:10px">Biomedical Entity Recognition</div>'
        f'<div style="font-size:14px;line-height:2.4;color:#1e293b">{result}</div>'
        f'<div style="margin-top:12px;padding-top:10px;border-top:1px solid #f1f5f9;'
        f'display:flex;flex-wrap:wrap">{legend_items}</div>'
        '</div>'
    )

# ── Combined output (single HTML block, self-contained layout) ────────────────
def build_output_html(prob_severe, prob_mild, tokens, scores, text, entities):
    gauge      = build_gauge_html(prob_severe, prob_mild)
    confidence = build_confidence_html(prob_severe, prob_mild)
    heatmap    = build_heatmap_html(tokens, scores)
    ner        = build_ner_html(text, entities)
    return f"""
<div style="display:flex;gap:16px;align-items:flex-start;font-family:system-ui,sans-serif">
  <div style="flex:1;min-width:0">
    <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                letter-spacing:0.1em;color:#1e3a5f;margin-bottom:8px">Severity Score</div>
    {gauge}
    {confidence}
  </div>
  <div style="flex:2;min-width:0;display:flex;flex-direction:column;gap:12px">
    <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                letter-spacing:0.1em;color:#1e3a5f;margin-bottom:8px">Token Contributions &amp; Entity Recognition</div>
    <div style="margin-bottom:12px">{heatmap}</div>
    <div>{ner}</div>
  </div>
</div>"""

# ── Main ──────────────────────────────────────────────────────────────────────
def analyze(text, n_steps):
    if not text.strip():
        return "<p style='color:#94a3b8;padding:20px;font-family:sans-serif'>Enter text above and click Analyze.</p>"
    pred, prob_mild, prob_severe = get_prediction(text)
    tokens, scores               = get_ig_scores(text, n_steps=int(n_steps))
    entities                     = ner_pipe(text)
    return build_output_html(prob_severe, prob_mild, tokens, scores, text, entities)

# ── Gradio UI ─────────────────────────────────────────────────────────────────
PLACEHOLDER = "e.g. After the injection I experienced extreme muscle spasms and convulsions and ended up in the ER."

EXAMPLES = [
    ("Mild — perimenopausal anxiety, high confidence",
     "I experience perimenopausal anxiety symptoms and am very sensitive to medications. The side effects, mild sleepiness and a heavy feeling in my head, are manageable. I absolutely love this medication; it is a lifesaver for my anxiety. I only take it as needed, so my first prescription of 30 pills lasted 8 months, and I have no concerns about dependency. I am so glad my OB-GYN suggested it."),
    ("Severe — migraine medication",
     "I was taking this medication as a migraine preventive. During the 9 days I took it, I experienced an extremely low pulse, low blood pressure, and felt loopy and lightheaded. It did prevent a migraine, but the side effects were significant."),
    ("Mild — hormonal acne, long-term use",
     "I have been taking this medication for hormonal acne for the last 7 years and have been very happy with the results. I do not recall any notable side effects, I gained about 20 lbs over three years, but I attribute that to my eating habits rather than the medication. It has been incredible for my skin; no more cystic acne, just the occasional small pimple. My previously very oily skin is now much more normal, which has been a huge confidence boost. Changing my pillowcase every few nights has also helped. One important note: you must taper off this medication gradually and never stop cold turkey. I have since tapered down to 25mg over the last couple of months and am still seeing positive results."),
]

CSS = """
body { background: #f0f4f8 !important; }
.gradio-container {
    background: white !important;
    max-width: 1200px !important;
    margin: 28px auto !important;
    border-radius: 18px !important;
    border: 1px solid rgba(37,99,235,0.22) !important;
    box-shadow:
        0 0 0 1px rgba(37,99,235,0.07),
        0 4px 24px rgba(37,99,235,0.13),
        0 12px 48px rgba(37,99,235,0.09),
        0 2px 8px rgba(0,0,0,0.06) !important;
    overflow: hidden !important;
}
.block { background: transparent !important; }
.ex-card button {
    background: #2563eb !important;
    border: 1px solid #1d4ed8 !important;
    border-top: 2px solid #1e3a5f !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    text-align: center !important;
    font-size: 0.78rem !important;
    color: white !important;
    line-height: 1.5 !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 70px !important;
    box-shadow: 0 1px 4px rgba(37,99,235,0.08) !important;
    transition: all 0.15s ease !important;
}
.ex-card button:hover {
    background: white !important;
    border-color: #2563eb !important;
    border-top-color: #1e3a5f !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.18) !important;
    color: #1e293b !important;
    transform: translateY(-1px) !important;
}
.run-btn button {
    background: #1e3a5f !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    height: 44px !important;
}
.run-btn button:hover { background: #162d4a !important; }
.input-box textarea {
    background: white !important;
    border-radius: 10px !important;
    border-color: #bfdbfe !important;
    font-size: 0.9rem !important;
}
"""

with gr.Blocks(title="ADR Severity Classifier") as demo:

    # ── Header ──────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div style='background:white;border-bottom:3px solid #1e3a5f;padding:18px 28px;
                box-shadow:0 2px 10px rgba(37,99,235,0.10)'>
      <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px'>
        <div style='display:flex;align-items:center;gap:16px'>
          {LOGO_SVG}
          <div>
            <div style='font-size:1.55rem;font-weight:800;color:#1e3a5f;
                        letter-spacing:-0.02em;line-height:1'>MedSentinel</div>
            <div style='font-size:10px;color:#2563eb;letter-spacing:0.14em;margin-top:4px;
                        text-transform:uppercase;font-weight:700'>ADR Surveillance</div>
          </div>
        </div>
        <div style='display:flex;align-items:center;gap:8px'>
          <span style='background:#eff6ff;color:#1e3a5f;font-size:10px;font-weight:700;
                       padding:4px 9px;border-radius:5px;letter-spacing:0.06em;
                       border:1px solid #bfdbfe'>v3.2</span>
          <span style='background:#fefce8;color:#92400e;font-size:10px;font-weight:700;
                       padding:4px 9px;border-radius:5px;letter-spacing:0.06em;
                       border:1px solid #fde68a'>UVA MSBA PROTOTYPE</span>
          <span style='background:#fefce8;color:#92400e;font-size:10px;font-weight:700;
                       padding:4px 9px;border-radius:5px;letter-spacing:0.06em;
                       border:1px solid #fde68a'>TEAM 9</span>

                       
        </div>
      </div>
    </div>""")

    # ── Info cards ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style='background:white;border:1px solid #bfdbfe;border-top:3px solid #2563eb;
                border-radius:12px;padding:15px 18px;
                box-shadow:0 1px 6px rgba(37,99,235,0.08);width:100%;box-sizing:border-box'>
      <div style='font-size:13px;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.1em;color:#1e3a5f;margin-bottom:7px'>Purpose</div>
      <p style='font-size:0.82rem;color:#374151;line-height:1.6;margin:0 0 14px 0'>
        Classifies patient-reported drug experiences as <strong>severe</strong> or
        <strong>mild</strong> adverse reactions using DeBERTa-v3-Base fine-tuned
        on real patient narratives.</p>
      <div style='border-top:1px solid #e8f0fe;padding-top:13px'>
        <div style='font-size:13px;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.1em;color:#1e3a5f;margin-bottom:7px'>How to Interpret</div>
        <p style='font-size:0.82rem;color:#374151;line-height:1.6;margin:0'>
          The gauge displays the <strong>Severity Probability</strong> on a 0–1 scale.
          The token heatmap highlights which words drove the prediction —
          <span style='color:#dc2626;font-weight:600'>red</span> tokens push toward severe,
          while <span style='color:#2563eb;font-weight:600'>blue</span> tokens push toward mild.
          Opacity reflects the magnitude of each token's contribution.</p>
      </div>
    </div>""")

    gr.HTML("<div style='height:10px'></div>")

    # ── Input ───────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style='margin-bottom:8px'>
      <div style='font-size:13px;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.1em;color:#1e3a5f;margin-bottom:4px'>
        Enter Drug Experience and Symptoms Text</div>
      <div style='font-size:0.82rem;font-family:system-ui,sans-serif;color:#1e3a5f;line-height:1.6'>
        Type in your adverse drug reaction experience with symptoms and select the analyze button</div>
    </div>""")
    with gr.Row():
        with gr.Column(scale=4):
            text_input = gr.Textbox(
                placeholder=PLACEHOLDER,
                lines=4,
                show_label=False,
                elem_classes=["input-box"],
            )
            n_steps_slider = gr.Slider(
                minimum=10, maximum=50, value=20, step=10,
                label="Explanation Detail Level",
                info="Controls how thoroughly the model traces which words influenced the result. Higher = more precise highlights but slower. Keep at 20 for best performance.",
            )
            run_btn = gr.Button("Analyze →", variant="primary", elem_classes=["run-btn"])

    # ── Example cards ───────────────────────────────────────────────────────────
    gr.HTML("""
    <div style='margin:14px 0 4px'>
      <div style='font-size:13px;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.1em;color:#1e3a5f;margin-bottom:4px'>Try an Example</div>
      <div style='font-size:0.82rem;font-family:system-ui,sans-serif;color:#1e3a5f;line-height:1.6;margin-bottom:10px'>
        Select an example that will populate in the text box and press analyze</div>
    </div>""")
    with gr.Row():
        with gr.Column():
            gr.HTML("""<div style='font-size:11px;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:#16a34a;margin-bottom:5px;
                        font-family:system-ui,sans-serif;text-align:center'>Mild</div>""")
            btn1 = gr.Button(
                f"\"{EXAMPLES[0][1][:80]}...\"",
                elem_classes=["ex-card"],
            )
            btn1.click(fn=lambda: EXAMPLES[0][1], inputs=[], outputs=text_input)
        with gr.Column():
            gr.HTML("""<div style='font-size:11px;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:#dc2626;margin-bottom:5px;
                        font-family:system-ui,sans-serif;text-align:center'>Severe</div>""")
            btn2 = gr.Button(
                f"\"{EXAMPLES[1][1][:80]}...\"",
                elem_classes=["ex-card"],
            )
            btn2.click(fn=lambda: EXAMPLES[1][1], inputs=[], outputs=text_input)
        with gr.Column():
            gr.HTML("""<div style='font-size:11px;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:#16a34a;margin-bottom:5px;
                        font-family:system-ui,sans-serif;text-align:center'>Mild</div>""")
            btn3 = gr.Button(
                f"\"{EXAMPLES[2][1][:80]}...\"",
                elem_classes=["ex-card"],
            )
            btn3.click(fn=lambda: EXAMPLES[2][1], inputs=[], outputs=text_input)

    gr.HTML("<hr style='border:none;border-top:2px solid #e0eaff;margin:18px 0'>")

    # ── Single combined output ───────────────────────────────────────────────────
    output = gr.HTML()

    # ── Disclaimer ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style='background:#eff6ff;border:1px solid #bfdbfe;border-top:3px solid #2563eb;
                border-radius:8px;padding:12px 16px;margin-top:16px;font-size:13px;
                color:#1e3a5f;font-family:system-ui,sans-serif;line-height:1.5'>
      <strong>Research Prototype Only:</strong> MedSentinel is an academic prototype and has not
      been validated for clinical use. Do not use for medical decision-making. All outputs should
      be reviewed by a qualified healthcare professional.
    </div>""")

    run_btn.click(
        fn=analyze,
        inputs=[text_input, n_steps_slider],
        outputs=[output],
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css=CSS,
    )
