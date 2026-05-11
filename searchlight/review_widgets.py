from io import BytesIO
import math
from pathlib import Path
from urllib.parse import quote

import ipywidgets as widgets
import pandas
from matplotlib import pyplot

from .models.simulens.v1 import (
    DEFAULT_SPLIT_SIZES,
    _segment_lightcurve,
    _segment_windows_for_split_size,
)
from searchlight.public_candidate_exports import plot_candidate_plot_with_tpf


def make_zooniverse_review_item(row, lc, zoom_window_days=None, zoom_window_jd=None):
    path = row["path"]
    score = row.get("lensing")
    subject_id = row.get("subject_id")
    most_likely_answer = row.get("most_likely_answer")
    num_votes = row.get("data.num_votes")
    agreement = row.get("data.agreement")

    header_items = [
        widgets.HTML(f"<strong>{Path(path).name}</strong>"),
        widgets.HTML(f"<code>subject_id={subject_id}</code>"),
    ]
    if pandas.notna(most_likely_answer):
        header_items.append(
            widgets.HTML(f"<code>most_likely={most_likely_answer}</code>")
        )
    header = widgets.HBox(header_items)

    ticid = lc.meta.get("TICID") or lc.meta.get("TARGETID")
    sector = lc.meta.get("SECTOR")
    ra = lc.meta.get("RA")
    dec = lc.meta.get("DEC")

    tpf_fig = plot_candidate_plot_with_tpf(
        path,
        zoom_window_days=zoom_window_days,
        zoom_window_jd=zoom_window_jd,
    )
    zoom_window = getattr(tpf_fig, "searchlight_zoom_window", None)

    lines = [
        f"<div><strong>File:</strong> <code>{path}</code></div>",
        f"<div><strong>Subject ID:</strong> <code>{subject_id}</code></div>",
    ]
    if zoom_window is not None:
        lines.append(
            "<div><strong>Zoom window:</strong> "
            f"<code>{zoom_window['start_jd']:.6f}</code> to "
            f"<code>{zoom_window['end_jd']:.6f}</code> JD "
            f"(<code>{zoom_window['start_day']:.6f}</code> to "
            f"<code>{zoom_window['end_day']:.6f}</code> days, "
            f"{zoom_window['source']})</div>"
        )
    if pandas.notna(most_likely_answer):
        lines.append(
            f"<div><strong>Most likely answer:</strong> {most_likely_answer}</div>"
        )
    if pandas.notna(num_votes):
        lines.append(f"<div><strong>Votes:</strong> {num_votes}</div>")
    if pandas.notna(agreement):
        lines.append(f"<div><strong>Agreement:</strong> {agreement}</div>")
    if pandas.notna(score):
        lines.append(f"<div><strong>Simulens score:</strong> {score}</div>")
    if ticid is not None:
        lines.append(f"<div><strong>TIC:</strong> {ticid}</div>")
    if sector is not None:
        lines.append(f"<div><strong>Sector:</strong> {sector}</div>")
    if ra is not None and dec is not None:
        coord = f"{ra} {dec}"
        simbad_url = "https://simbad.cds.unistra.fr/simbad/sim-coo?Coord=" + quote(
            coord
        )
        lines.append(
            f'<div><strong>SIMBAD:</strong> <a href="{simbad_url}" target="_blank">{simbad_url}</a></div>'
        )
    details = widgets.HTML("".join(lines))

    tpf_buffer = BytesIO()
    tpf_fig.savefig(tpf_buffer, format="png", bbox_inches="tight")
    pyplot.close(tpf_fig)
    split_children = [widgets.Image(value=tpf_buffer.getvalue(), format="png")]
    open_index = 0

    def split_label(split_size_days):
        if math.isinf(split_size_days):
            return "all data"
        return f"{split_size_days:g} days"

    for split_size in DEFAULT_SPLIT_SIZES:
        if math.isinf(split_size):
            continue
        split_size_days = float(split_size)
        windows = _segment_windows_for_split_size(lc, split_size_days)
        segments = [_segment_lightcurve(lc, start, end) for start, end, _ in windows]
        segments = [seg for seg in segments if len(seg) > 0]

        if not segments:
            split_children.append(
                widgets.HTML(
                    f"<div>No non-empty segments for <code>{split_label(split_size_days)}</code>.</div>"
                )
            )
            # if math.isinf(split_size_days):
            #     open_index = len(split_children) - 1
            continue

        fig, axes = pyplot.subplots(
            len(segments),
            1,
            figsize=(12, max(2.5, 2.5 * len(segments))),
            squeeze=False,
        )
        axes = axes.flatten()
        fig.suptitle(
            f"{Path(path).name} | split_size={split_size_days} days",
            fontsize=12,
        )

        for i, seg in enumerate(segments):
            ax = axes[i]
            ax.scatter(seg["time"].jd, seg["flux"], s=2, alpha=0.6)
            ax.set_ylabel("flux")
            ax.set_title(f"segment {i + 1} ({len(seg)} points)", fontsize=10)

        axes[-1].set_xlabel("time (JD)")
        pyplot.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        pyplot.close(fig)
        split_children.append(widgets.Image(value=buffer.getvalue(), format="png"))
        # if math.isinf(split_size_days):
        #     open_index = len(split_children) - 1

    plots = widgets.Accordion(children=split_children)
    for index, split_size in enumerate(DEFAULT_SPLIT_SIZES):
        plots.set_title(index, split_label(float(split_size)))
    plots.selected_index = open_index

    card = widgets.VBox([header, details, plots])
    card.layout.border = "1px solid #ddd"
    card.layout.padding = "8px"
    card.layout.margin = "0 0 16px 0"
    return card
