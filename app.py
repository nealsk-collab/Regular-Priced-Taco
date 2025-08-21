# streamlit run app.py
# --- Renewals Dashboard (Streamlit) ---
# Author: ChatGPT for Neal Khan @ Prophix
# Rebuilt: 2025-08-21
# Generated at (America/Toronto): 2025-08-21 13:05:00 EDT
#
# How to run
# 1) pip install streamlit pandas numpy plotly openpyxl python-dateutil
#    (optional for heatmap clicking) pip install streamlit-plotly-events
# 2) streamlit run app.py

import io
from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def fmt_m(x: float) -> str:
    try:
        return f"${x/1_000_000:.2f}M"
    except Exception:
        return "—"


def fmt_dollars(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return "—"

# Optional component to capture plotly click events for drilldown
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

# ---------------------------------------------------------------------
# Page config & minimal styling
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Renewals Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .metric-card {background: white; border-radius: 16px; padding: 16px; box-shadow: 0 4px 18px rgba(0,0,0,0.06);} 
      .pill {display:inline-block; padding: 2px 10px; border-radius: 999px; background:#f1f5f9; font-size:12px; margin-left:8px;}
      .help-text {color:#64748b; font-size: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Display a generated-at timestamp (local to America/Toronto)
GENERATED_AT = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y-%m-%d %H:%M:%S %Z")

# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def list_excel_sheets(file_bytes: bytes) -> list:
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        return xl.sheet_names
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, file_type: str, sheet: str | None) -> pd.DataFrame:
    if file_type == "csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)

@st.cache_data(show_spinner=False)
def load_optional_tab(file_bytes: bytes, tab_name: str) -> pd.DataFrame | None:
    try:
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=tab_name)
    except Exception:
        return None

# ---------------------------------------------------------------------
# Column mapping helpers
# ---------------------------------------------------------------------

CANONICAL_FIELDS = {
    "account": ["account", "account name", "customer", "customer name", "company", "client"],
    "renewal_date": ["renewal date", "contract end", "end date", "expiry", "exp date", "close date", "renew date"],
    "arr": ["arr", "arr usd", "mrr", "acv", "contract value", "revenue", "amount", "acv ($cad)", "acv cad"],
    "status": ["status", "renewal status", "outcome", "stage", "result", "state"],
    "owner": ["owner", "account owner", "ae", "csm", "rep"],
    "segment": ["segment", "tier", "size", "customer segment"],
    "region": ["region", "geo", "territory"],
    "product": ["product", "sku", "solution", "package"],
    "probability": ["probability", "%", "win prob", "renewal prob"],
    "term_months": ["term", "term months", "months", "subscription term"],
    "churn_reason": ["churn reason", "lost reason", "reason", "cancellation reason"],
    "start_date": ["start date", "contract start", "effective date"],
    # Optional extras used in app
    "cz_churn_risk": ["cz churn risk", "churn risk", "risk score", "cz risk"],
    "status_v2": ["status v2", "status2", "renewal status v2"],
    "cy_stage": ["cy contract stage", "cy contract status", "contract stage"],
    "file_updated_date": ["file updated date", "file update date", "dataset updated", "as of date", "snapshot date"],
}


def normalize(s: str) -> str:
    return str(s).strip().lower().replace("_", " ")


@st.cache_data(show_spinner=False)
def auto_map_columns(df: pd.DataFrame) -> dict:
    mapping: dict[str, str | None] = {k: None for k in CANONICAL_FIELDS}
    for canon, synonyms in CANONICAL_FIELDS.items():
        for c in df.columns:
            n = normalize(c)
            if any(syn in n or n in syn for syn in synonyms):
                mapping[canon] = c
                break
    return mapping

# ---------------------------------------------------------------------
# Status bucketing (mutually exclusive, incl. Upcoming)
# ---------------------------------------------------------------------

STATUS_V2_ORDER = ["Cancelled", "Upcoming Renewal", "Pending Payment", "Renewed"]
STATUS_V2_STACK_BOTTOM_TO_TOP = ["Renewed", "Pending Payment", "Upcoming Renewal", "Cancelled"]
STATUS_V2_COLOR = {
    "Cancelled": "#e74c3c",        # red
    "Upcoming Renewal": "#f39c12",  # orange
    "Pending Payment": "#f1c40f",   # yellow
    "Renewed": "#2ecc71",          # green
}


def normalize_status_v2(s: pd.Series) -> pd.Series:
    n = s.astype(str).str.strip().str.lower()
    is_upcoming = n.eq("upcoming") | n.eq("upcoming renewal") | n.eq("upcoming-renewal")
    is_pending_pay = n.str.contains("pending") | n.str.contains("payment") | n.str.contains("invoice")
    is_renewed = n.str.startswith("renewed")
    is_cancel = n.str.contains("cancel") | n.str.contains("churn") | n.eq("lost")
    out = np.where(is_upcoming, "Upcoming Renewal",
          np.where(is_pending_pay, "Pending Payment",
          np.where(is_renewed, "Renewed",
          np.where(is_cancel, "Cancelled", n.str.title()))))
    return pd.Series(out, index=s.index)


def derive_status_bucket(raw: pd.Series) -> pd.Series:
    n = raw.astype(str).str.strip().str.lower()
    is_up = n.isin(["upcoming", "upcoming renewal", "upcoming-renewal"])  # exact forms
    b = np.where(is_up, "Upcoming Renewal",
         np.where(n.str.contains("pending|payment|invoice"), "Pending Payment",
         np.where(n.str.startswith("renewed"), "Renewed",
         np.where(n.str.contains("cancel|churn") | n.eq("lost"), "Cancelled", "Other"))))
    return pd.Series(b, index=raw.index)


def summarize_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({"status_v2": STATUS_V2_STACK_BOTTOM_TO_TOP, "acv": [0.0, 0.0, 0.0, 0.0]})
    bucket = derive_status_bucket(df["status_v2_raw"]) if "status_v2_raw" in df.columns else pd.Series([], dtype=str)
    ac = pd.to_numeric(df["acv"], errors="coerce").fillna(0)
    return pd.DataFrame({
        "status_v2": STATUS_V2_STACK_BOTTOM_TO_TOP,
        "acv": [
            float(ac[bucket.eq("Renewed")].sum()),
            float(ac[bucket.eq("Pending Payment")].sum()),
            float(ac[bucket.eq("Upcoming Renewal")].sum()),
            float(ac[bucket.eq("Cancelled")].sum()),
        ],
    })

# ---------------------------------------------------------------------
# Helpers for file-updated date + safe date-range defaults
# ---------------------------------------------------------------------

def find_file_updated_date(df: pd.DataFrame, mapped_name: str | None = None):
    """Return python date if a 'file updated date' column exists; else None."""
    cand = None
    if mapped_name and mapped_name in df.columns:
        s = pd.to_datetime(df[mapped_name], errors="coerce")
        if s.notna().any():
            cand = s.max()
    if cand is None:
        names = [c for c in df.columns if "file" in str(c).lower() and "update" in str(c).lower() and "date" in str(c).lower()]
        for c in names:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                cand = s.max(); break
    return cand.date() if cand is not None and pd.notna(cand) else None


def ensure_date_range_defaults(file_updated, min_d, max_d):
    """Initialize st.session_state['date_range'] once and return (start, end) dates.
    start = first of next month after file_updated; end = Dec 31 same year.
    """
    if isinstance(file_updated, pd.Timestamp):
        fud = file_updated
    elif isinstance(file_updated, date):
        fud = pd.Timestamp(file_updated)
    else:
        fud = pd.Timestamp.today()
    default_start = (fud + pd.offsets.MonthBegin(1)).normalize()
    default_end = pd.Timestamp(year=fud.year, month=12, day=31)

    if "date_range" not in st.session_state:
        s = default_start.date() if pd.notna(default_start) else (min_d.date() if pd.notna(min_d) else date.today())
        e = default_end.date() if pd.notna(default_end) else (max_d.date() if pd.notna(max_d) else date.today())
        st.session_state["date_range"] = (s, e)
    # Create the widget without passing value again (avoid Streamlit warning)
    dr = st.sidebar.date_input("Renewal date range", key="date_range")
    # Normalize shapes
    if isinstance(dr, (list, tuple)):
        if len(dr) == 2:
            return dr[0], dr[1]
        elif len(dr) == 1:
            return dr[0], dr[0]
        else:
            return st.session_state["date_range"][0], st.session_state["date_range"][1]
    else:
        return dr, dr

# ---------------------------------------------------------------------
# Sidebar — upload & settings
# ---------------------------------------------------------------------

st.sidebar.header("1) Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel (.xlsx)", type=["csv", "xlsx"], accept_multiple_files=False)

file_type = None
sheet_names: list[str] = []
selected_sheet = None
raw_bytes: bytes | None = None

if uploaded is not None:
    raw_bytes = uploaded.read()
    file_type = "csv" if uploaded.name.lower().endswith(".csv") else "xlsx"
    if file_type == "xlsx":
        sheet_names = list_excel_sheets(raw_bytes)
        if sheet_names:
            default_sheet = next((s for s in sheet_names if s.strip().lower() == "customer data details - act"), sheet_names[0])
            selected_sheet = st.sidebar.selectbox("Choose Excel tab", options=sheet_names, index=sheet_names.index(default_sheet))

    # Optional tabs
    fd = load_optional_tab(raw_bytes, "field descriptions") if file_type == "xlsx" else None
    addl = load_optional_tab(raw_bytes, "additional details") if file_type == "xlsx" else None
    if fd is not None and not fd.empty:
        with st.sidebar.expander("Field descriptions (from workbook"):
            st.dataframe(fd, use_container_width=True, height=200)
    if addl is not None and not addl.empty:
        with st.sidebar.expander("Additional details / requirements"):
            st.dataframe(addl, use_container_width=True, height=160)

st.sidebar.header("2) Settings")
fy_start_month = st.sidebar.selectbox("Fiscal year starts in…", options=list(range(1,13)), index=0, format_func=lambda m: datetime(2000,m,1).strftime("%B"))

st.sidebar.header("3) Targets & Defaults")
begin_acv = st.sidebar.number_input("Beginning ACV ($CAD)", min_value=0.0, step=1000.0, value=164_330_000.0, key="begin_acv")
grr_target = st.sidebar.number_input("GRR% Target", min_value=0.0, max_value=100.0, step=0.5, value=92.5, key="grr_target")
file_updated_date = st.sidebar.date_input("File updated date", value=date.today(), key="file_updated_date")

# Derived target: Annual Max Churn Value
annual_max_churn_value = ((100.0 - grr_target) / 100.0) * begin_acv if begin_acv and grr_target is not None else 0.0

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

st.title("📈 Renewals Dashboard")
st.caption(f"Code generated at: {GENERATED_AT}")
st.caption("Upload your dataset, map fields, then explore KPIs, charts, and drilldowns.")

if uploaded is None:
    st.info("👋 Upload a CSV or Excel workbook in the left sidebar to begin.")
    st.stop()

data = load_data(raw_bytes, file_type or "xlsx", selected_sheet)
if data is None or data.empty:
    st.warning("The selected sheet/file appears empty.")
    st.stop()

# Column mapping UI
st.subheader("Column Mapping")
suggested = auto_map_columns(data)
cols = list(data.columns)
col_map = {}
left, right = st.columns(2)
with left:
    col_map["account"] = st.selectbox("Account/Customer name", options=[None]+cols, index=(cols.index(suggested.get("account")) + 1) if suggested.get("account") in cols else 0)
    col_map["renewal_date"] = st.selectbox("Renewal / End Date", options=[None]+cols, index=(cols.index(suggested.get("renewal_date")) + 1) if suggested.get("renewal_date") in cols else 0)
    col_map["arr"] = st.selectbox("ACV / ARR / Contract Value", options=[None]+cols, index=(cols.index(suggested.get("arr")) + 1) if suggested.get("arr") in cols else 0)
    col_map["status_v2"] = st.selectbox("Status v2", options=[None]+cols, index=(cols.index(suggested.get("status_v2")) + 1) if suggested.get("status_v2") in cols else 0)
    col_map["probability"] = st.selectbox("Probability (optional)", options=[None]+cols, index=(cols.index(suggested.get("probability")) + 1) if suggested.get("probability") in cols else 0)
with right:
    col_map["cy_stage"] = st.selectbox("CY Contract Stage/Status", options=[None]+cols, index=(cols.index(suggested.get("cy_stage")) + 1) if suggested.get("cy_stage") in cols else 0)
    col_map["owner"] = st.selectbox("Owner (optional)", options=[None]+cols, index=(cols.index(suggested.get("owner")) + 1) if suggested.get("owner") in cols else 0)
    col_map["segment"] = st.selectbox("Segment (optional)", options=[None]+cols, index=(cols.index(suggested.get("segment")) + 1) if suggested.get("segment") in cols else 0)
    col_map["region"] = st.selectbox("Region (optional)", options=[None]+cols, index=(cols.index(suggested.get("region")) + 1) if suggested.get("region") in cols else 0)
    col_map["cz_churn_risk"] = st.selectbox("CZ churn risk (optional)", options=[None]+cols, index=(cols.index(suggested.get("cz_churn_risk")) + 1) if suggested.get("cz_churn_risk") in cols else 0)
    col_map["file_updated_date"] = st.selectbox("File updated date (optional)", options=[None]+cols, index=(cols.index(suggested.get("file_updated_date")) + 1) if suggested.get("file_updated_date") in cols else 0)

required_missing = [k for k in ["account","renewal_date","arr","status_v2","cy_stage"] if not col_map.get(k)]
if required_missing:
    st.warning("Please map Account, Renewal Date, ACV/ARR, Status v2, and CY Contract Stage/Status to continue.")
    st.stop()

# Working frame
work = pd.DataFrame({
    "account": data[col_map["account"]],
    "renewal_date": pd.to_datetime(data[col_map["renewal_date"]], errors="coerce"),
    "acv": pd.to_numeric(data[col_map["arr"]].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.replace(" ", "", regex=False), errors="coerce"),
    "status_v2_raw": data[col_map["status_v2"]].astype(str),
    "status_v2": normalize_status_v2(data[col_map["status_v2"]]),
    "cy_stage": data[col_map["cy_stage"]].astype(str),
})
for opt in ["owner","segment","region","product","probability","term_months","cz_churn_risk"]:
    c = col_map.get(opt)
    if c:
        if opt in ["probability","term_months"]:
            work[opt] = pd.to_numeric(data[c], errors="coerce")
        else:
            work[opt] = data[c].astype(str)

# File updated date from file (latest value)
try:
    fud_from_file = None
    if col_map.get("file_updated_date"):
        fud_from_file = pd.to_datetime(data[col_map["file_updated_date"]], errors="coerce").max()
    if pd.notna(fud_from_file):
        st.session_state["file_updated_date"] = fud_from_file.date()
except Exception:
    pass

file_updated_date = pd.to_datetime(st.session_state.get("file_updated_date", file_updated_date))

# Build date-range defaults & widget (safe)
min_date = pd.to_datetime(work["renewal_date"]).min()
max_date = pd.to_datetime(work["renewal_date"]).max()
start_d, end_d = ensure_date_range_defaults(file_updated_date, min_date, max_date)

# Normalize datetime bounds
filter_start_dt = pd.to_datetime(start_d)
filter_end_dt = pd.to_datetime(end_d).replace(hour=23, minute=59, second=59)
filter_start = filter_start_dt.date() if pd.notna(filter_start_dt) else None
filter_end = filter_end_dt.date() if pd.notna(filter_end_dt) else None

# Filters
work["month"] = work["renewal_date"].dt.to_period("M").dt.to_timestamp()
with st.expander("Filters", expanded=True):
    c1, c2, c3, c4, c5 = st.columns(5)
    stage_f = c1.multiselect("CY Contract Stage", sorted([s for s in work.get("cy_stage", pd.Series([])).dropna().astype(str).str.strip().unique()]))
    status_f = c2.multiselect("Status v2", STATUS_V2_STACK_BOTTOM_TO_TOP)
    reg = c3.multiselect("Region", sorted([s for s in work.get("region", pd.Series([])).dropna().unique()]))
    own = c4.multiselect("Owner", sorted([s for s in work.get("owner", pd.Series([])).dropna().unique()]))
    risk = c5.multiselect("CZ churn risk", sorted([s for s in work.get("cz_churn_risk", pd.Series([])).dropna().unique()]))

mask = pd.Series(True, index=work.index)
mask &= work["renewal_date"].between(filter_start_dt, filter_end_dt)
if stage_f:
    mask &= work.get("cy_stage").astype(str).str.strip().isin(stage_f)
if status_f:
    mask &= work.get("status_v2").isin(status_f)
if reg:
    mask &= work.get("region").isin(reg)
if own:
    mask &= work.get("owner").isin(own)
if risk:
    mask &= work.get("cz_churn_risk").isin(risk)

flt = work[mask].copy()

# ---------------------------------------------------------------------
# Calendar-year windows (ORG-LEVEL, full year — now aligned to Renewal Year for Cancelled to match GRR)
# ---------------------------------------------------------------------

if isinstance(file_updated_date, (pd.Timestamp, datetime, date)):
    file_updated_date = pd.to_datetime(file_updated_date)
cy_start = pd.Timestamp(year=file_updated_date.year, month=1, day=1)
cy_end = pd.Timestamp(year=file_updated_date.year, month=12, day=31, hour=23, minute=59, second=59)

_work_stage = work["cy_stage"].astype(str).str.strip().str.lower()
_work_status = normalize_status_v2(work["status_v2"])  # canonical labels

# ACV for Renewal (CY as of Jan 1): renewal-year renewals within the calendar year
mask_cy_renewals = _work_stage.eq("renewal year") & work["renewal_date"].between(cy_start, cy_end)
acv_for_renewal_cy = pd.to_numeric(work.loc[mask_cy_renewals, "acv"], errors="coerce").fillna(0).sum()

# ACV Renewed CY — Renewal Year stage, full calendar year
mask_renewed_cy = work["renewal_date"].between(cy_start, cy_end) & _work_status.eq("Renewed") & _work_stage.eq("renewal year")
acv_renewed_cy = pd.to_numeric(work.loc[mask_renewed_cy, "acv"], errors="coerce").fillna(0).sum()

# Cancelled CY — **Renewal Year** only (to match GRR waterfall)  ← aligns to ~9.8M in your data
mask_churn_cy = work["renewal_date"].between(cy_start, cy_end) & _work_status.eq("Cancelled") & _work_stage.eq("renewal year")
cy_existing_churn = pd.to_numeric(work.loc[mask_churn_cy, "acv"], errors="coerce").fillna(0).sum()

annual_max_churn_value = ((100.0 - grr_target) / 100.0) * begin_acv if begin_acv and grr_target is not None else 0.0
remaining_churn_allowance = max(0.0, annual_max_churn_value - cy_existing_churn)

# KPI cards (CY FULL YEAR)
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("ACV for Renewal (CY as of Jan 1)", fmt_m(acv_for_renewal_cy))
    st.markdown("</div>", unsafe_allow_html=True)
with k2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("ACV Renewed CY", fmt_m(acv_renewed_cy))
    st.markdown("</div>", unsafe_allow_html=True)
with k3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("ACV Cancelled CY", fmt_m(cy_existing_churn))
    st.markdown("</div>", unsafe_allow_html=True)
with k4:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Annual Max Churn Value", fmt_m(annual_max_churn_value))
    st.markdown("</div>", unsafe_allow_html=True)
with k5:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Remaining Churn Allowance", fmt_m(remaining_churn_allowance))
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# GRR Waterfall — Calendar Year (true waterfall; Cancelled goes UP)
#  - Implemented with go.Waterfall using only supported properties in your Plotly build
#  - All steps are positive ("relative") so they WATERFALL UP; Ending ACV is a "total"
#  - Note: Plotly Waterfall in your environment doesn't support per-bar colors; 'increasing' applies to all positives.
# ---------------------------------------------------------------------

st.subheader("GRR Waterfall — Calendar Year (CY totals)")
wrk_stage_l = work["cy_stage"].astype(str).str.strip().str.lower()

# Renewal Year + CY window for components
sub_wf_year = work.loc[(wrk_stage_l.eq("renewal year")) & (work["renewal_date"].between(cy_start, cy_end)), ["status_v2_raw","acv"]].copy()
sub_wf_year["status_v2_raw"] = sub_wf_year["status_v2_raw"].astype(str).str.strip().str.lower()
wf_buckets_cy = summarize_buckets(sub_wf_year)

R = float(wf_buckets_cy.loc[wf_buckets_cy["status_v2"].eq("Renewed"), "acv"].sum())
P = float(wf_buckets_cy.loc[wf_buckets_cy["status_v2"].eq("Pending Payment"), "acv"].sum())
U = float(wf_buckets_cy.loc[wf_buckets_cy["status_v2"].eq("Upcoming Renewal"), "acv"].sum())
C = float(wf_buckets_cy.loc[wf_buckets_cy["status_v2"].eq("Cancelled"), "acv"].sum())

ENDING = max(0.0, R + P + U - C)

steps_x = ["Renewed CY", "Pending Payment (CY)", "Upcoming (CY)", "Cancelled CY", "Ending ACV"]
measures = ["relative", "relative", "relative", "relative", "total"]
ys = [R, P, U, C, ENDING]

fig_wf = go.Figure(go.Waterfall(
    name="GRR CY",
    orientation="v",
    x=steps_x,
    measure=measures,
    y=ys,
    # Waterfall in this Plotly build doesn't accept trace-level 'marker'; use supported buckets only
    increasing={"marker": {"color": "#f39c12"}},  # single color for all positive steps
    decreasing={"marker": {"color": "#e74c3c"}},  # not used (no negatives here)
    totals={"marker": {"color": "#3498db"}},
    connector={"line": {"color": "rgba(0,0,0,0.2)"}}
))
fig_wf.update_layout(margin=dict(t=40, r=20, l=20, b=10))
fig_wf.update_yaxes(title_text="ACV ($CAD)")
# Horizontal reference line at (ACV for Renewal CY − ACV Cancelled CY) = max possible ending ACV
max_possible_end = max(0.0, acv_for_renewal_cy - cy_existing_churn)
fig_wf.add_hline(y=max_possible_end, line_dash="dot", line_color="#e74c3c", 
                 annotation_text=f"Max possible ending ACV: {fmt_dollars(max_possible_end)}",
                 annotation_position="top left")
st.plotly_chart(fig_wf, use_container_width=True)

# ---------------------------------------------------------------------
# Stacked bars: Up For Renewal & In-Commitment (ACV), guideline = Minimum to Renew
# (These remain driven by the sidebar date range)
# ---------------------------------------------------------------------

st.subheader("Status v2 — Up For Renewal (ACV $CAD)")
stage_norm = flt["cy_stage"].astype(str).str.strip().str.lower()
up_mask = stage_norm.eq("renewal year")
sub_up = flt.loc[up_mask, ["status_v2_raw","acv"]].copy()
sub_up["status_v2_raw"] = sub_up["status_v2_raw"].astype(str).str.strip().str.lower()
up_agg = summarize_buckets(sub_up)
up_agg["x"] = "Up For Renewal"

fig_up = px.bar(
    up_agg,
    x="x",
    y="acv",
    color="status_v2",
    category_orders={"status_v2": STATUS_V2_STACK_BOTTOM_TO_TOP},
    color_discrete_map=STATUS_V2_COLOR,
    barmode="stack",
    title=f"Up For Renewal — ACV ($CAD) by Status v2 ({filter_start} → {filter_end})",
)
fig_up.update_yaxes(title_text="ACV ($CAD)")
fig_up.update_xaxes(title_text="")
_up_total = float(up_agg["acv"].sum()) if len(up_agg) else 0.0
_min_to_renew = max(0.0, _up_total - remaining_churn_allowance)
if _min_to_renew > 0:
    fig_up.add_hline(y=_min_to_renew, line_dash="dot", line_color="#8e44ad", annotation_text=f"Minimum Renewal to hit GRR ({fmt_dollars(_min_to_renew)})", annotation_position="top left")
fig_up.add_annotation(x="Up For Renewal", y=_up_total, text=f"Total ACV: {fmt_dollars(_up_total)}", showarrow=False, yshift=10)

st.plotly_chart(fig_up, use_container_width=True)

st.markdown(f"**Total Up For Renewal (period)**: {fmt_m(_up_total)}  ")
st.markdown(f"**Minimum to Renew to hit GRR target**: {fmt_m(_min_to_renew)} (Remaining churn allowance: {fmt_m(remaining_churn_allowance)})")

st.subheader("Status v2 — In-Commitment (ACV $CAD)")
in_mask = stage_norm.eq("in-commitment")
sub_in = flt.loc[in_mask, ["status_v2_raw","acv"]].copy()
sub_in["status_v2_raw"] = sub_in["status_v2_raw"].astype(str).str.strip().str.lower()
in_agg = summarize_buckets(sub_in)
in_agg["x"] = "In-Commitment"

fig_in = px.bar(
    in_agg,
    x="x",
    y="acv",
    color="status_v2",
    category_orders={"status_v2": STATUS_V2_STACK_BOTTOM_TO_TOP},
    color_discrete_map=STATUS_V2_COLOR,
    barmode="stack",
    title=f"In-Commitment — ACV ($CAD) by Status v2 ({filter_start} → {filter_end})",
)
fig_in.update_yaxes(title_text="ACV ($CAD)")
fig_in.update_xaxes(title_text="")
_in_total = float(in_agg["acv"].sum()) if len(in_agg) else 0.0
fig_in.add_annotation(x="In-Commitment", y=_in_total, text=f"Total ACV: {fmt_dollars(_in_total)}", showarrow=False, yshift=10)

st.plotly_chart(fig_in, use_container_width=True)

# ---------------------------------------------------------------------
# Debug: show how raw status maps to buckets (period & stage)
# ---------------------------------------------------------------------

with st.expander("Debug: Upcoming calculation — mapping & sums"):
    def debug_table(stage_exact: str) -> pd.DataFrame:
        d = work.copy()
        stage_l = d["cy_stage"].astype(str).str.strip().str.lower()
        m = d["renewal_date"].between(filter_start_dt, filter_end_dt) & stage_l.eq(stage_exact)
        d = d.loc[m, ["status_v2_raw","acv"]].copy()
        d["bucket"] = derive_status_bucket(d["status_v2_raw"])  # same logic as charts
        g = d.groupby(["status_v2_raw","bucket"], as_index=False)["acv"].sum().sort_values("acv", ascending=False)
        return g
    c_u, c_c = st.columns(2)
    with c_u:
        st.markdown("**Up For Renewal (stage == 'Renewal Year') — raw status → bucket**")
        st.dataframe(debug_table("renewal year"), use_container_width=True, height=280)
    with c_c:
        st.markdown("**In-Commitment (stage == 'In-Commitment') — raw status → bucket**")
        st.dataframe(debug_table("in-commitment"), use_container_width=True, height=280)

# ---------------------------------------------------------------------
# Cohort Heatmap (ACV gradient, clickable)
# ---------------------------------------------------------------------

st.subheader("Cohort Heatmap (Month × cz churn risk) — ACV")
risk_src = flt.copy()
if "cz_churn_risk" in risk_src.columns:
    risk_src["cz_churn_risk"] = risk_src["cz_churn_risk"].fillna("Unassigned")
    risk_src["month"] = risk_src["renewal_date"].dt.to_period("M").dt.to_timestamp()
    risk_grp = risk_src.groupby(["month","cz_churn_risk"], as_index=False)["acv"].sum()

    if not risk_grp.empty:
        pt = risk_grp.pivot_table(index="cz_churn_risk", columns="month", values="acv", aggfunc="sum", fill_value=0)
        risk_labels = pt.index.tolist(); months = pt.columns
        z_vals = pt.values.astype(float)
        text_labels = np.vectorize(lambda v: f"{int(round(v/1000.0)):,}K")(z_vals)

        fig_heat = go.Figure(go.Heatmap(
            z=z_vals,
            x=months.strftime("%Y-%m"),
            y=[str(x) for x in risk_labels],
            text=text_labels, texttemplate="%{text}", textfont={"size":12},
            hovertext=[[f"{risk_labels[i]} • {months[j].strftime('%Y-%m')}<br>ACV: {fmt_dollars(z_vals[i,j])}" for j in range(len(months))] for i in range(len(risk_labels))],
            hoverinfo="text",
            colorscale="Blues",
            showscale=True,
            xgap=3, ygap=3,
        ))
        fig_heat.update_layout(margin=dict(t=20,r=20,l=20,b=10), plot_bgcolor="white", paper_bgcolor="white")
        fig_heat.update_xaxes(title_text="Month", tickangle=-30, tickfont={"size":11}, showgrid=True, gridcolor="#cbd5e1")
        fig_heat.update_yaxes(title_text="cz churn risk", tickfont={"size":11}, showgrid=True, gridcolor="#cbd5e1")

        if "cohort_selections" not in st.session_state:
            st.session_state["cohort_selections"] = []
        if HAS_PLOTLY_EVENTS:
            clicks = plotly_events(fig_heat, click_event=True, select_event=True, override_height=460, override_width="100%", key="cohort_risk")
            if clicks:
                m = clicks[0].get("x"); r = clicks[0].get("y")
                if m and r:
                    pair = (str(m), str(r))
                    if pair not in st.session_state["cohort_selections"]:
                        st.session_state["cohort_selections"].append(pair)
        else:
            st.plotly_chart(fig_heat, use_container_width=True)
            cell_labels = [f"{months[j].strftime('%Y-%m')} × {risk_labels[i]}" for i in range(len(risk_labels)) for j in range(len(months)) if pt.values[i,j] > 0]
            label_to_pair = {lab: (lab.split(" × ")[0], lab.split(" × ")[1]) for lab in cell_labels}
            manual = st.multiselect("Add/remove heatmap cells for drilldown:", options=cell_labels, key="cohort_manual_select")
            for lab in manual:
                p = label_to_pair.get(lab)
                if p and p not in st.session_state["cohort_selections"]:
                    st.session_state["cohort_selections"].append(p)

        st.session_state["cohort_combined_pairs"] = st.session_state.get("cohort_selections", [])
        if st.session_state["cohort_combined_pairs"]:
            cA, cB = st.columns([3,1])
            with cA:
                st.markdown("**Selected cells for drilldown:** " + ", ".join([f"{m} × {r}" for m,r in st.session_state["cohort_combined_pairs"]]))
            with cB:
                if st.button("Clear selections"):
                    st.session_state["cohort_selections"] = []
                    st.session_state["cohort_manual_select"] = []
    else:
        st.info("No data in the current time range to build the heatmap.")
else:
    st.info("cz churn risk column not provided.")

# ---------------------------------------------------------------------
# ACV by Owner (uses normalized filter_start/filter_end in title)
# ---------------------------------------------------------------------

st.subheader("ACV by Owner — current filters")
if "owner" in flt.columns and flt["owner"].notna().any():
    by_owner = flt.groupby("owner", as_index=False)["acv"].sum().sort_values("acv", ascending=False).head(30)
    if not by_owner.empty:
        fig_owner = px.bar(by_owner, x="owner", y="acv", title=f"Owners — ACV ($CAD) ({filter_start} → {filter_end})")
        fig_owner.update_layout(margin=dict(t=70,r=20,l=20,b=120), height=480)
        fig_owner.update_xaxes(tickangle=-30, automargin=True)
        fig_owner.update_yaxes(title_text="ACV ($CAD)")
        st.plotly_chart(fig_owner, use_container_width=True)
    else:
        st.info("No owner data in current filters.")
else:
    st.info("Owner column not provided or empty.")

# ---------------------------------------------------------------------
# Drilldown table (optionally filtered by cohort cell clicks)
# ---------------------------------------------------------------------

st.subheader("Drilldown: Accounts")
drill = flt.copy()
# selections: list of ("YYYY-MM", risk_label)
selections = st.session_state.get("cohort_combined_pairs", st.session_state.get("cohort_selections", []))
if selections:
    if "month" not in drill.columns:
        drill["month"] = drill["renewal_date"].dt.to_period("M").dt.to_timestamp()
    mask_sel = pd.Series(False, index=drill.index)
    descs = []
    for m_str, r_lbl in selections:
        try:
            m_ts = pd.to_datetime(m_str)
            mask_sel |= ((drill["month"] == m_ts) & (drill.get("cz_churn_risk") == r_lbl))
            descs.append(f"{m_ts.strftime('%Y-%m')} × {r_lbl}")
        except Exception:
            pass
    drill = drill[mask_sel]
    if descs:
        st.caption("Filtered by selections: " + ", ".join(descs))

show_cols = [c for c in ["account","renewal_date","acv","status_v2","cy_stage","owner","segment","region","product","probability","term_months","cz_churn_risk"] if c in drill.columns]
st.dataframe(drill[show_cols].sort_values(["renewal_date","acv"], ascending=[True, False]), use_container_width=True, height=360)

csv = drill[show_cols].to_csv(index=False)
st.download_button("Download filtered accounts (CSV)", data=csv, file_name="renewals_drilldown.csv", mime="text/csv")

# ---------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------

with st.expander("Notes & Best Practices"):
    st.markdown(
        f"""
        - **Generated at**: {GENERATED_AT}
        - **ACV charts**: Separate stacked bars show **Up For Renewal** and **In-Commitment** in **ACV ($CAD)**, stacked bottom→top as **Renewed → Pending Payment → Upcoming Renewal → Cancelled**.
        - The guide line on *Up For Renewal* shows **Minimum Renewal to hit GRR** for the selected period (= `Total Up For Renewal` − `Remaining Churn Allowance`).
        - **KPIs**: **Calendar-year totals**. *ACV for Renewal (CY)* & *Renewed CY* use **Stage = Renewal Year**. *Cancelled CY* now also uses **Stage = Renewal Year** to match the GRR chart.
        - **Cohort heatmap**: ACV gradient (Blues). Click cells to drive drilldown (requires `streamlit-plotly-events`). Fallback multiselect is shown if the component is missing.
        - Use the **Debug** expander to verify raw → bucket mapping used by the charts.
        """
    )
