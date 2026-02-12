# pages/home.py
from dash import html, Input, Output

GREEN = "#bfeda6"
YELLOW = "#ffd966"
RED = "#f6546a"

DECISION_THRESHOLD = 6  # days


def layout():
    return html.Div(
        children=[
            html.H3("Maintenance timeline", style={"marginTop": 0}),
            html.Div(
                "Buckets are based on action time = (rul_pred − leadtime).",
                style={"opacity": 0.7, "marginTop": "-6px"},
            ),
            html.Div(id="home_week_tiles", style={"marginTop": "12px"}),
        ]
    )


def _bucket_label(action_days: float) -> str:
    if action_days <= 0:
        return "Overdue / Immediate"
    if action_days <= 3:
        return "0–3 days"
    if action_days <= 7:
        return "3–7 days"
    if action_days <= 14:
        return "1–2 weeks"
    if action_days <= 28:
        return "2–4 weeks"
    if action_days <= 56:
        return "1–2 months"
    return "2+ months"


def _bucket_color(action_days: float) -> str:
    if action_days <= 0:
        return RED
    if action_days <= DECISION_THRESHOLD:
        return YELLOW
    return GREEN


def _make_tiles(items):
    buckets = [
        "Overdue / Immediate",
        "0–3 days",
        "3–7 days",
        "1–2 weeks",
        "2–4 weeks",
        "1–2 months",
        "2+ months",
    ]
    by_bucket = {b: [] for b in buckets}

    for it in items:
        action_days = float(it["pred"] - it["leadtime"])
        by_bucket[_bucket_label(action_days)].append({**it, "action_days": action_days})

    tiles = []

    for b in buckets:
        group = sorted(by_bucket[b], key=lambda x: x["pred"])
        count = len(group)

        # tile color: worst (most urgent) action_days
        if group:
            worst_action = min(g["action_days"] for g in group)
            bg = _bucket_color(worst_action)
        else:
            bg = GREEN

        # inside tile: show a few machine IDs
        sample = []
        for g in group[:6]:
            sample.append(f"{int(g['id']):03d}")
        inside_ids = "  ".join(sample) if sample else "—"

        # tooltip text aligned
        if group:
            lines = []
            for g in group:
                id_str = f"{int(g['id']):03d}"           # 3-digit id
                part_str = f"{str(g['product']):<6}"     # tire/brake padded
                rul_str = f"{float(g['pred']):>5.1f}"    # aligned number
                lines.append(f"{id_str}  {part_str}  {rul_str} d")

            header = "ID   Part    RUL"
            separator = "-" * 18
            tooltip = "\n".join([header, separator] + lines[:25])
            if len(lines) > 25:
                tooltip += f"\n... +{len(lines)-25} more"
        else:
            tooltip = "No machines"

        tiles.append(
            html.Div(
                title=tooltip,  # ✅ native tooltip (always works)
                style={
                    "position": "relative",
                    "flex": "1 1 calc(25% - 12px)",
                    "minWidth": "200px",
                    "height": "125px",
                    "borderRadius": "16px",
                    "backgroundColor": bg,
                    "padding": "14px",
                    "display": "flex",
                    "flexDirection": "column",
                    "justifyContent": "center",
                    "alignItems": "center",
                    "fontWeight": "700",
                    "boxShadow": "0 3px 10px rgba(0,0,0,0.08)",
                    "cursor": "default",
                },
                children=[
                    html.Div(b, style={"fontSize": "13px", "opacity": 0.85, "textAlign": "center"}),
                    html.Div(str(count), style={"fontSize": "34px", "marginTop": "6px"}),
                    html.Div("machines", style={"fontSize": "12px", "opacity": 0.7}),
                    html.Div(
                        inside_ids,
                        style={
                            "marginTop": "6px",
                            "fontSize": "12px",
                            "opacity": 0.85,
                            "fontFamily": "monospace",
                        },
                    ),
                ],
            )
        )

    return html.Div(
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "14px",
        },
        children=tiles,
    )


def register_callbacks(app):
    @app.callback(
        Output("home_week_tiles", "children"),
        Input("bench_data_store", "data"),
        Input("bench_out_store", "data"),
    )
    def _update_home_tiles(data_rows, out_rows):
        data_rows = data_rows or []
        out_rows = out_rows or []

        data_by_id = {int(r["id"]): r for r in data_rows}

        items = []
        for o in out_rows:
            mid = int(o["id"])
            d = data_by_id.get(mid)
            if not d:
                continue

            items.append(
                dict(
                    id=mid,
                    product=str(d.get("product", "unknown")),
                    leadtime=int(d.get("leadtime", 7)),
                    pred=float(o.get("pred_cost_sensitive", 0.0)),
                )
            )

        return _make_tiles(items)
