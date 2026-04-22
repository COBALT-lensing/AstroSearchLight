import uuid
import hashlib
import logging
import re
import time
from pathlib import Path

import numpy
from matplotlib import pyplot
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from astropy import units
from astropy.coordinates import SkyCoord
from lightkurve import read, search_targetpixelfile, search_tesscut

from .lightcurves import load

ZOONIVERSE_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "black-hole-hunters.org")
TPF_REFERENCE_LOOKBACK_DAYS = 2.0
logger = logging.getLogger(__name__)


def flux_to_numpy(flux):
    if hasattr(flux, "value"):
        flux = flux.value
    return numpy.asarray(flux, dtype=float)


def clear_corrupt_lightkurve_download(exc):
    message = str(exc)
    path_match = None
    if "buffer is too small for requested array" in message:
        path_match = re.search(r"(\S+_tp\.fits)", message)
    elif "File is not a zip file" in message:
        path_match = re.search(r"(\S+\.zip)", message)

    if path_match is None:
        return False

    corrupt_path = Path(path_match.group(1))
    if not corrupt_path.exists():
        return False

    corrupt_path.unlink(missing_ok=True)
    logger.warning("Removed corrupt cached pixel download at %s", corrupt_path)
    return True


def build_tpf_cache_path(download_dir, ticid, sector=None, ra=None, dec=None):
    cache_root = Path(download_dir) / "cached_tpfs"
    cache_key = repr(
        {
            "ticid": ticid,
            "sector": sector,
            "ra": None if ra is None else round(float(ra), 8),
            "dec": None if dec is None else round(float(dec), 8),
        }
    )
    digest = hashlib.md5(cache_key.encode()).hexdigest()
    return cache_root / digest[:2] / digest[2:4] / f"{digest}.fits"


def search_tesscut_pixel_data(sector=None, ra=None, dec=None):
    if ra is None or dec is None:
        return None

    target = SkyCoord(float(ra) * units.deg, float(dec) * units.deg, frame="icrs")
    search_result = search_tesscut(target, sector=sector)
    if len(search_result) == 0:
        return None
    return search_result


def search_spoc_tpf_data(ticid, sector=None):
    if ticid is None:
        return None

    search_result = search_targetpixelfile(
        f"TIC {int(ticid)}",
        mission="TESS",
        author="SPOC",
        sector=sector,
        limit=1,
    )
    if len(search_result) == 0:
        return None
    return search_result


def build_candidate_id(path):
    return str(uuid.uuid5(ZOONIVERSE_UUID_NAMESPACE, str(path)))


def build_output_name(path):
    candidate_id = build_candidate_id(path)
    return candidate_id, f"{candidate_id}.png"


def build_output_relative_path(filename):
    stem = filename.removesuffix(".png").replace("-", "")
    return Path(stem[:2]) / stem[2:4] / filename


def render_candidate_plot(path, output_path):
    lc = load(path)
    time_days = numpy.asarray(lc["time"].jd, dtype=float)
    time_days = time_days - time_days[0]
    brightness = flux_to_numpy(lc["flux"])

    finite = numpy.isfinite(time_days) & numpy.isfinite(brightness)
    time_days = time_days[finite]
    brightness = brightness[finite]

    if len(time_days) == 0:
        raise ValueError(f"No finite data points found in {path}")

    median_brightness = numpy.nanmedian(brightness)
    if numpy.isfinite(median_brightness) and median_brightness != 0:
        brightness = brightness / median_brightness

    fig, ax = pyplot.subplots(figsize=(10, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#f7f4ea")
    ax.set_facecolor("#fffdf8")

    ax.scatter(
        time_days,
        brightness,
        s=9,
        color="#24445c",
        alpha=0.75,
        edgecolors="none",
        zorder=2,
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9aa8b6")
    ax.spines["bottom"].set_color("#9aa8b6")
    ax.grid(axis="y", color="#d8dfdf", linewidth=0.8, alpha=0.8)

    ax.set_xlabel("Time (days)", color="#23313d")
    ax.set_ylabel("Brightness", color="#23313d")
    ax.tick_params(colors="#4c5b67")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    pyplot.close(fig)


def render_candidate_task(task):
    render_candidate_plot(task["path"], Path(task["output_path"]))
    return task["output_path"]


def extract_tess_tpf_metadata(path):
    lc = load(path)
    ticid = lc.meta.get("TICID") or lc.meta.get("TARGETID")
    sector = lc.meta.get("SECTOR")
    ra = lc.meta.get("RA")
    dec = lc.meta.get("DEC")
    return ticid, sector, ra, dec


def search_for_tess_pixel_data(ticid, sector=None, ra=None, dec=None):
    if ra is not None and dec is not None:
        try:
            search_result = search_tesscut_pixel_data(sector=sector, ra=ra, dec=dec)
            if search_result is not None:
                return ("tesscut", search_result)
        except Exception as exc:
            logger.warning(
                "TESScut search by coordinates failed for sector %s; trying SPOC TPF search: %s",
                sector,
                exc,
            )

    if ticid is not None:
        try:
            search_result = search_spoc_tpf_data(ticid=ticid, sector=sector)
            if search_result is not None:
                return ("spoc", search_result)
        except Exception as exc:
            logger.warning(
                "TPF search by TIC failed for sector %s after TESScut fallback path: %s",
                sector,
                exc,
            )

    raise FileNotFoundError("No TESS pixel data found via TESScut or SPOC TPF search")


def download_tess_tpf(
    ticid,
    sector=None,
    ra=None,
    dec=None,
    download_dir=None,
    max_attempts=4,
):
    if ticid is None and (ra is None or dec is None):
        raise ValueError(
            "Could not determine a TIC ID or sky position for pixel download"
        )

    cache_path = build_tpf_cache_path(
        download_dir=download_dir,
        ticid=ticid,
        sector=sector,
        ra=ra,
        dec=dec,
    )
    if cache_path.exists():
        try:
            return read(cache_path)
        except Exception as exc:
            cache_path.unlink(missing_ok=True)
            logger.warning("Removed unreadable cached TPF at %s: %s", cache_path, exc)

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            source, search_result = search_for_tess_pixel_data(
                ticid=ticid,
                sector=sector,
                ra=ra,
                dec=dec,
            )
            if source == "tesscut":
                tesscut_errors = []
                for cutout_size in ((11, 11), (9, 9), (7, 7), (5, 5)):
                    try:
                        tpf = search_result.download(
                            download_dir=download_dir,
                            cutout_size=cutout_size,
                        )
                        if tpf is None:
                            raise FileNotFoundError("TPF download returned no file")
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        tpf.hdu.writeto(cache_path, overwrite=True)
                        return tpf
                    except Exception as exc:
                        tesscut_errors.append(exc)
                        if clear_corrupt_lightkurve_download(exc):
                            continue
                        if "too near the edge of the FFI" in str(exc):
                            logger.warning(
                                "TESScut cutout size %s failed near FFI edge for sector %s; trying a smaller cutout",
                                cutout_size,
                                sector,
                            )
                            continue
                        if "File is not a zip file" in str(exc):
                            logger.warning(
                                "TESScut cutout size %s hit a corrupt zip for sector %s; retrying with a fresh download",
                                cutout_size,
                                sector,
                            )
                            continue
                        raise

                spoc_result = search_spoc_tpf_data(ticid=ticid, sector=sector)
                if spoc_result is not None:
                    tpf = spoc_result.download(download_dir=download_dir)
                    if tpf is None:
                        raise FileNotFoundError("TPF download returned no file")
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    tpf.hdu.writeto(cache_path, overwrite=True)
                    return tpf

                if tesscut_errors:
                    raise tesscut_errors[-1]
                raise FileNotFoundError("TESScut returned no usable cutout")

            tpf = search_result.download(download_dir=download_dir)
            if tpf is None:
                raise FileNotFoundError("TPF download returned no file")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tpf.hdu.writeto(cache_path, overwrite=True)
            return tpf
        except Exception as exc:
            last_error = exc
            cleared_corrupt_file = clear_corrupt_lightkurve_download(exc)
            cache_path.unlink(missing_ok=True)
            if attempt == max_attempts:
                break
            if cleared_corrupt_file:
                logger.warning(
                    "Corrupt cached pixel data detected for sector %s on attempt %d/%d; retrying download",
                    sector,
                    attempt,
                    max_attempts,
                )
            else:
                logger.warning(
                    "Pixel-data download failed for TIC %s in sector %s on attempt %d/%d: %s",
                    ticid,
                    sector,
                    attempt,
                    max_attempts,
                    exc,
                )
            time.sleep(attempt)

    raise RuntimeError(
        f"Failed to download TESS target pixel file after {max_attempts} attempts"
    ) from last_error


def nanmean_frame(frames):
    frame_stack = numpy.stack(frames, axis=0)
    finite = numpy.isfinite(frame_stack)
    counts = finite.sum(axis=0)
    totals = numpy.where(finite, frame_stack, 0.0).sum(axis=0)
    return numpy.divide(
        totals,
        counts,
        out=numpy.full(frame_stack.shape[1:], numpy.nan, dtype=float),
        where=counts > 0,
    )


def summarize_tpf_frames(
    tpf,
    window_start_day,
    window_end_day,
    time_zero_jd,
    peak_day=None,
    reference_lookback_days=TPF_REFERENCE_LOOKBACK_DAYS,
):
    flux_cube = flux_to_numpy(tpf.flux)
    if flux_cube.ndim != 3 or len(flux_cube) == 0:
        raise ValueError("Unexpected target pixel file shape")

    tpf_time_jd = numpy.asarray(tpf.time.jd, dtype=float)
    finite_frames = numpy.flatnonzero(numpy.isfinite(tpf_time_jd))
    if len(finite_frames) == 0:
        raise ValueError("No finite target pixel frames available")

    window_start_jd = time_zero_jd + window_start_day
    window_end_jd = time_zero_jd + window_end_day
    reference_start_jd = window_start_jd - reference_lookback_days
    reference_frames = finite_frames[
        (tpf_time_jd[finite_frames] >= reference_start_jd)
        & (tpf_time_jd[finite_frames] < window_start_jd)
    ]
    window_frames = finite_frames[
        (tpf_time_jd[finite_frames] >= window_start_jd)
        & (tpf_time_jd[finite_frames] <= window_end_jd)
    ]
    if len(window_frames) == 0:
        window_frames = finite_frames
    if len(reference_frames) == 0:
        reference_frames = finite_frames[tpf_time_jd[finite_frames] < window_start_jd]
    if len(reference_frames) == 0:
        logger.warning(
            "No target pixel frames were available before the plotted window; "
            "using the first plotted-window frame as the reference"
        )
        reference_frames = window_frames[:1]

    reference_frame = nanmean_frame(
        [
            numpy.array(flux_cube[index], dtype=float, copy=True)
            for index in reference_frames
        ]
    )

    sample_times_jd = numpy.linspace(window_start_jd, window_end_jd, 5)
    peak_sample_index = None
    if peak_day is not None and numpy.isfinite(peak_day):
        peak_sample_index = 2
        peak_jd = numpy.clip(time_zero_jd + peak_day, window_start_jd, window_end_jd)
        sample_times_jd = numpy.array(
            [
                window_start_jd,
                (window_start_jd + peak_jd) / 2,
                peak_jd,
                (peak_jd + window_end_jd) / 2,
                window_end_jd,
            ],
            dtype=float,
        )
    frame_indices = [None] * 5
    used_frame_indices = set()
    for sample_position, sample_time_jd in enumerate(sample_times_jd):
        nearest = int(
            window_frames[
                numpy.argmin(numpy.abs(tpf_time_jd[window_frames] - sample_time_jd))
            ]
        )
        if nearest not in used_frame_indices or sample_position == peak_sample_index:
            frame_indices[sample_position] = nearest
            used_frame_indices.add(nearest)

    if any(index is None for index in frame_indices):
        for candidate in window_frames:
            candidate = int(candidate)
            if candidate in used_frame_indices:
                continue
            missing_position = next(
                index
                for index, frame_index in enumerate(frame_indices)
                if frame_index is None
            )
            frame_indices[missing_position] = candidate
            used_frame_indices.add(candidate)
            if all(index is not None for index in frame_indices):
                break

    frames = []
    for sample_number, index in enumerate(frame_indices, start=1):
        if index is None:
            continue
        frame = flux_cube[index]
        if numpy.isfinite(frame).any():
            relative_day = tpf_time_jd[index] - time_zero_jd
            frames.append((f"{sample_number}", relative_day, frame))

    if not frames:
        raise ValueError("No finite target pixel heatmaps available")
    if not numpy.isfinite(reference_frame).any():
        raise ValueError("No finite target pixel reference heatmap available")

    return frames, reference_frame


def add_tpf_samples_subplot(
    ax,
    frames,
    reference_frame,
    x_left,
    x_right,
    lightcurve_median_flux,
):
    if not numpy.isfinite(lightcurve_median_flux) or lightcurve_median_flux == 0:
        raise ValueError("Light-curve median flux must be finite and non-zero")

    rendered_frames = []
    difference_values = []

    for _, relative_day, frame in frames[:5]:
        frame_array = numpy.array(frame, dtype=float, copy=True)
        difference_frame = (frame_array - reference_frame) / lightcurve_median_flux
        rendered_frames.append((relative_day, difference_frame))
        finite_difference = difference_frame[numpy.isfinite(difference_frame)]
        if len(finite_difference) > 0:
            difference_values.append(finite_difference)

    if len(difference_values) == 0:
        raise ValueError("No finite target pixel values available")

    valid_difference_values = numpy.concatenate(difference_values)
    abs_limit = numpy.nanpercentile(numpy.abs(valid_difference_values), 90)
    if not numpy.isfinite(abs_limit) or abs_limit == 0:
        abs_limit = numpy.nanmax(numpy.abs(valid_difference_values))
    if not numpy.isfinite(abs_limit) or abs_limit == 0:
        abs_limit = 1.0
    abs_limit = max(abs_limit, 0.005)
    diff_vmin = -abs_limit
    diff_vmax = abs_limit
    norm = Normalize(vmin=diff_vmin, vmax=diff_vmax)

    ax.set_facecolor("#fffdf8")
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors="#4c5b67", labelsize=8, length=3, pad=2)
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9aa8b6")

    sample_width = min(
        (x_right - x_left) * 0.207,
        max((x_right - x_left) * 0.138, 0.207),
    )
    sample_height = 0.94
    sample_bottom = 0.06

    slot_centers = numpy.linspace(x_left, x_right, len(rendered_frames))
    for slot_center, (relative_day, rendered_frame) in zip(
        slot_centers,
        rendered_frames,
    ):
        left = (slot_center - x_left) / (x_right - x_left)
        width = sample_width / (x_right - x_left)
        left = numpy.clip(left - width / 2, 0.0, max(0.0, 1.0 - width))
        inset = ax.inset_axes([left, sample_bottom, width, sample_height])
        inset.set_facecolor("#fffdf8")
        frame_to_plot = numpy.array(rendered_frame, copy=True)
        frame_to_plot[~numpy.isfinite(frame_to_plot)] = numpy.nan
        inset.imshow(
            frame_to_plot,
            origin="lower",
            cmap="RdBu_r",
            norm=norm,
        )
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_xlabel(
            f"{relative_day:.1f} d", fontsize=7, color="#4c5b67", labelpad=2
        )
        for spine in inset.spines.values():
            spine.set_color("#d6cfc2")
            spine.set_linewidth(1.0)

        ax.annotate(
            "",
            xy=(left + width / 2, sample_bottom),
            xycoords=ax.transAxes,
            xytext=(relative_day, 0.0),
            textcoords=ax.transData,
            arrowprops={
                "arrowstyle": "-",
                "color": "#c7b9a5",
                "linewidth": 1.0,
                "alpha": 0.9,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=False,
            zorder=1,
        )

    tick_values = sorted({round(relative_day, 3) for _, relative_day, _ in frames[:5]})
    ax.set_xticks(tick_values)
    ax.set_xlabel("Time (days)", color="#23313d", labelpad=6)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.text(
        0.0,
        1.02,
        "TESS pixel data",
        transform=ax.transAxes,
        fontsize=9,
        color="#23313d",
        ha="left",
        va="bottom",
    )

    colorbar_mappable = cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    colorbar = pyplot.colorbar(
        colorbar_mappable,
        ax=ax,
        orientation="vertical",
        fraction=0.035,
        pad=0.02,
    )
    colorbar.set_label(
        "Brightness difference",
        color="#23313d",
        fontsize=8,
    )
    colorbar.ax.tick_params(colors="#4c5b67", labelsize=7)
    colorbar.outline.set_edgecolor("#9aa8b6")


def plot_full_lightcurve_overview(ax, time_days, brightness, x_left, x_right):
    finite = numpy.isfinite(time_days) & numpy.isfinite(brightness)
    if not finite.any():
        return

    ax.set_facecolor("#fffdf8")
    ax.scatter(
        time_days[finite],
        brightness[finite],
        s=4,
        color="#24445c",
        alpha=0.5,
        edgecolors="none",
        zorder=2,
    )
    ax.axvspan(
        x_left,
        x_right,
        color="#d7c5a5",
        alpha=0.42,
        linewidth=0,
        zorder=1,
    )
    ax.set_xlim(
        float(numpy.nanmin(time_days[finite])),
        float(numpy.nanmax(time_days[finite])),
    )
    y_min = float(numpy.nanmin(brightness[finite]))
    y_max = float(numpy.nanmax(brightness[finite]))
    if y_min < y_max:
        y_padding = max((y_max - y_min) * 0.08, 0.005)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9aa8b6")
    ax.spines["bottom"].set_color("#9aa8b6")
    ax.tick_params(colors="#4c5b67", labelsize=8)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel("Time (days)", color="#23313d")
    ax.set_ylabel("Brightness", color="#23313d")
    ax.set_title(
        "Full light curve",
        fontsize=9,
        color="#23313d",
        loc="left",
        pad=6,
    )
    ax.grid(axis="y", color="#d8dfdf", linewidth=0.8, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_color("#d6cfc2")


def smooth_series(values, window=11):
    if len(values) == 0:
        return values
    window = max(3, min(int(window), len(values)))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values
    kernel = numpy.ones(window, dtype=float) / window
    padded = numpy.pad(values, (window // 2,), mode="edge")
    return numpy.convolve(padded, kernel, mode="valid")


def choose_peak_window(time_days, brightness):
    median_level = 1.0
    smoothed = smooth_series(brightness, window=min(21, max(5, len(brightness) // 40)))
    deviation = numpy.abs(smoothed - median_level)
    peak_index = int(numpy.nanargmax(deviation))

    local_scatter = numpy.nanmedian(numpy.abs(brightness - smoothed))
    if not numpy.isfinite(local_scatter) or local_scatter == 0:
        local_scatter = numpy.nanmedian(numpy.abs(brightness - median_level))
    if not numpy.isfinite(local_scatter) or local_scatter == 0:
        local_scatter = 0.01
    return_tolerance = max(local_scatter * 2.5, 0.01)

    left_return = None
    for idx in range(peak_index, -1, -1):
        if abs(smoothed[idx] - median_level) <= return_tolerance:
            left_return = idx
            break

    right_return = None
    for idx in range(peak_index, len(smoothed)):
        if abs(smoothed[idx] - median_level) <= return_tolerance:
            right_return = idx
            break

    if left_return is None:
        left_span = time_days[peak_index] - time_days[0]
    else:
        left_span = time_days[peak_index] - time_days[left_return]

    if right_return is None:
        right_span = time_days[-1] - time_days[peak_index]
    else:
        right_span = time_days[right_return] - time_days[peak_index]

    half_width = max(left_span, right_span)
    total_span = max(time_days[-1] - time_days[0], 0.0)
    minimum_half_width = max(total_span * 0.03, 0.15)
    half_width = max(half_width * 1.15, minimum_half_width)

    center = time_days[peak_index]
    left_limit = max(time_days[0], center - half_width)
    right_limit = min(time_days[-1], center + half_width)

    if left_limit == time_days[0] and center - left_limit < half_width:
        right_limit = min(time_days[-1], left_limit + 2 * half_width)
    if right_limit == time_days[-1] and right_limit - center < half_width:
        left_limit = max(time_days[0], right_limit - 2 * half_width)

    if left_limit == right_limit:
        left_limit = time_days[0]
        right_limit = time_days[-1]

    return left_limit, right_limit, center


def render_candidate_plot_with_tpf(path, output_path, tpf_download_dir=None):
    lc = load(path)
    absolute_time_days = numpy.asarray(lc["time"].jd, dtype=float)
    time_zero_jd = absolute_time_days[0]
    time_days = absolute_time_days - time_zero_jd
    brightness = flux_to_numpy(lc["flux"])

    finite = numpy.isfinite(time_days) & numpy.isfinite(brightness)
    time_days = time_days[finite]
    brightness = brightness[finite]

    if len(time_days) == 0:
        raise ValueError(f"No finite data points found in {path}")

    median_brightness = numpy.nanmedian(brightness)
    if numpy.isfinite(median_brightness) and median_brightness != 0:
        brightness = brightness / median_brightness

    x_left, x_right, peak_day = choose_peak_window(time_days, brightness)

    fig, (ax_tpf, ax, ax_overview) = pyplot.subplots(
        3,
        1,
        figsize=(10, 7.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.0]},
    )
    fig.patch.set_facecolor("#f7f4ea")
    ax_tpf.set_facecolor("#fffdf8")
    ax.set_facecolor("#fffdf8")
    ax_overview.set_facecolor("#fffdf8")

    ax.scatter(
        time_days,
        brightness,
        s=9,
        color="#24445c",
        alpha=0.75,
        edgecolors="none",
        zorder=2,
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9aa8b6")
    ax.spines["bottom"].set_color("#9aa8b6")
    ax.grid(axis="y", color="#d8dfdf", linewidth=0.8, alpha=0.8)

    ticid, sector, ra, dec = extract_tess_tpf_metadata(path)
    tpf = download_tess_tpf(
        ticid=ticid,
        sector=sector,
        ra=ra,
        dec=dec,
        download_dir=tpf_download_dir,
    )
    frames, reference_frame = summarize_tpf_frames(
        tpf,
        window_start_day=x_left,
        window_end_day=x_right,
        time_zero_jd=time_zero_jd,
        peak_day=peak_day,
    )

    ax.set_xlabel("Time (days)", color="#23313d")
    ax.set_ylabel("Brightness", color="#23313d")
    ax.tick_params(colors="#4c5b67")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlim(x_left, x_right)
    ax.set_title(
        "Zoomed light curve",
        fontsize=9,
        color="#23313d",
        loc="left",
        pad=8,
    )
    plot_full_lightcurve_overview(
        ax_overview,
        time_days,
        brightness,
        x_left,
        x_right,
    )

    add_tpf_samples_subplot(
        ax_tpf,
        frames,
        reference_frame,
        x_left=x_left,
        x_right=x_right,
        lightcurve_median_flux=median_brightness,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    pyplot.close(fig)


def render_candidate_tpf_task(task):
    render_candidate_plot_with_tpf(
        task["path"],
        Path(task["output_path"]),
        tpf_download_dir=task.get("tpf_download_dir"),
    )
    return task["output_path"]
