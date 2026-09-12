"""stage_36: generate an English, submission-oriented CEUS figure suite."""
from __future__ import annotations

import base64
import io
import json
import os
import struct
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "streetscape" / "logs" / ".matplotlib_cache"))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import pandas as pd
from PIL import Image

OUT = ROOT / "author_workspace" / "publication" / "ceus" / "figures"
SOURCE = ROOT / "author_workspace" / "publication" / "ceus" / "source_data"
WIDTH_IN = 183 / 25.4

COLORS = {
    "blue": "#315F9B", "teal": "#176B55", "orange": "#D56A3A",
    "purple": "#7A4E87", "grey": "#AAB2B9", "dark": "#263238",
    "light": "#EEF3F5", "sky": "#5DADE2", "green": "#4E8B67",
}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": .7, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def save(fig: plt.Figure, number: int, slug: str) -> list[str]:
    directory = OUT / f"Fig{number}_{slug}"
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"Fig{number}_{slug}"
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [str(base.with_suffix(ext).relative_to(ROOT)) for ext in (".png", ".pdf", ".svg", ".tiff")]


def label(ax, value: str) -> None:
    ax.text(-.06, 1.04, value, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="right", va="bottom")


def box(ax, xy, wh, title, subtitle, color, title_size=6.7):
    x, y = xy; w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.012,rounding_size=.018",
                                facecolor="white", edgecolor=color, linewidth=1.2))
    ax.text(x + w/2, y + h*.61, title, ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=COLORS["dark"])
    ax.text(x + w/2, y + h*.29, subtitle, ha="center", va="center", fontsize=5.4,
            color="#5B6770", linespacing=1.2)


def arrow(ax, start, end, color="#718096"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1, color=color))


def soft_panel(ax, xy, wh, color, title, panel_letter):
    """Rounded pastel panel used by the Fig. 1 systems schematic."""
    x, y = xy; w, h = wh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=.010,rounding_size=.024",
        facecolor=color, edgecolor="none", zorder=0,
    ))
    ax.text(x + .018*w, y + h - .055*h, panel_letter, fontsize=9,
            fontweight="bold", ha="left", va="top", color="#111111")
    ax.text(x + w/2, y + h - .075*h, title, fontsize=7.3,
            fontweight="bold", ha="center", va="top", color="#111111")


def tiny_box(ax, xy, wh, title, subtitle="", face="white", edge="#53606A"):
    x, y = xy; w, h = wh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=.004,rounding_size=.008",
        facecolor=face, edgecolor=edge, linewidth=.65, zorder=2,
    ))
    ax.text(x+w/2, y+h*(.60 if subtitle else .50), title, ha="center", va="center",
            fontsize=5.8, fontweight="bold", color=COLORS["dark"], zorder=3)
    if subtitle:
        ax.text(x+w/2, y+h*.25, subtitle, ha="center", va="center",
                fontsize=5.0, color="#5B6770", linespacing=1.05, zorder=3)


def tiny_arrow(ax, start, end, color="#46525C", lw=.75):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=color, zorder=5))


def draw_view_rosette(ax, center, radius):
    cx, cy = center
    for index, theta in enumerate(np.linspace(0, 2*np.pi, 12, endpoint=False)):
        ex, ey = cx + radius*np.cos(theta), cy + radius*np.sin(theta)
        ax.plot([cx, ex], [cy, ey], color="#51606A", lw=.45, zorder=2)
        ax.add_patch(Circle((ex, ey), radius*.12, facecolor="#DCEAF7",
                            edgecolor=COLORS["blue"], linewidth=.55, zorder=3))
    ax.add_patch(Circle((cx, cy), radius*.15, facecolor=COLORS["orange"],
                        edgecolor="white", linewidth=.5, zorder=4))
    ax.text(cx, cy+radius*1.42, "N", ha="center", va="center", fontsize=5.5,
            fontweight="bold", color=COLORS["blue"])


def draw_panorama_icon(ax, xy, wh):
    x, y = xy; w, h = wh
    clip = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=.006",
                          facecolor="#D8EEF8", edgecolor="#48606D", linewidth=.6, zorder=2)
    ax.add_patch(clip)
    ax.add_patch(Rectangle((x, y), w, h*.36, facecolor="#D8D0BE", edgecolor="none", zorder=2.1))
    for offset, color in ((.08, "#699768"), (.28, "#4E8B67"), (.72, "#5A9165"), (.90, "#789E68")):
        ax.add_patch(Circle((x+w*offset, y+h*.55), h*.22, facecolor=color,
                            edgecolor="white", linewidth=.35, zorder=3))
        ax.plot([x+w*offset, x+w*offset], [y+h*.20, y+h*.48], color="#77624A", lw=.6, zorder=3)
    ax.add_patch(Polygon([(x+w*.40,y+h*.36),(x+w*.46,y+h*.70),(x+w*.61,y+h*.70),
                          (x+w*.66,y+h*.36)], closed=True, facecolor="#C7B4A4",
                         edgecolor="white", linewidth=.35, zorder=2.5))
    ax.plot([x+w*.50, x+w*.50], [y+h*.02, y+h*.34], color="white", lw=.55, ls="--", zorder=3)
    ax.text(x+w*.02, y+h*.85, "N", fontsize=5.0, fontweight="bold", color="#163F6C", zorder=4)


def draw_route_graph(ax, origin, scale, route_colors=None):
    ox, oy = origin
    pts=np.array([[0,.35],[.18,.58],[.36,.38],[.54,.67],[.72,.44],[.94,.68],
                  [.16,.10],[.42,.12],[.66,.12],[.92,.20]])
    pts[:,0]=ox+pts[:,0]*scale;pts[:,1]=oy+pts[:,1]*scale*.72
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(0,6),(6,7),(7,8),(8,9),(2,7),(4,8)]
    for i,j in edges: ax.plot(pts[[i,j],0],pts[[i,j],1],color="#A7AFB5",lw=.7,zorder=2)
    if route_colors:
        for path,color,shift in route_colors:
            q=pts[path].copy();q[:,1]+=shift
            ax.plot(q[:,0],q[:,1],color=color,lw=1.25,zorder=4,solid_capstyle="round")
    for x,y in pts: ax.add_patch(Circle((x,y),scale*.018,facecolor="#E9F2D7",edgecolor="#1E2A31",lw=.45,zorder=5))
    return pts


def figure1() -> list[str]:
    # Contract: show how geographic orientation is retained through learning and
    # becomes two distinct decision services. Style-only adaptation of the user's
    # modular reference: original geometry, icons, colors and panel logic.
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 6.15), facecolor="white")
    fig.subplots_adjust(left=.018, right=.987, top=.985, bottom=.018)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

    blush="#FAF6F6"; pale_blue="#EFF5FD"; pale_orange="#FFF1E6"; pale_yellow="#FFF8DA"; pale_green="#EFF8EF"
    soft_panel(ax,(.01,.765),(.98,.225),blush,"Geographically aligned multimodal data foundation","a")
    soft_panel(ax,(.01,.405),(.485,.325),pale_blue,"Direction-aware thermal prediction model","b")
    soft_panel(ax,(.515,.405),(.225,.325),pale_orange,"UTCI post-processing","c")
    soft_panel(ax,(.76,.405),(.23,.325),pale_blue,"Spatiotemporal validation","d")
    soft_panel(ax,(.01,.02),(.48,.345),pale_yellow,"User-facing shade and thermal-comfort routing","e")
    soft_panel(ax,(.51,.02),(.48,.345),pale_green,"Planner-facing streetscape shade decision","f")

    # a | 12 directional images -> true azimuth -> ghost-reduced panorama -> evidence/table.
    draw_view_rosette(ax,(.095,.862),.047)
    ax.text(.095,.795,"12 views per site",ha="center",fontsize=5.6,fontweight="bold")
    ax.text(.095,.782,"heading = camera azimuth",ha="center",fontsize=5.0,color="#5B6770")
    tiny_arrow(ax,(.155,.852),(.205,.852))
    ax.add_patch(Circle((.245,.852),.044,facecolor="white",edgecolor=COLORS["blue"],lw=.7))
    for text,dx,dy in (("N",0,.031),("E",.031,0),("S",0,-.031),("W",-.031,0)):
        ax.text(.245+dx,.852+dy,text,ha="center",va="center",fontsize=5.0,fontweight="bold")
    ax.plot([.245,.245],[.852,.886],color=COLORS["orange"],lw=1.25)
    ax.text(.245,.795,"Absolute direction",ha="center",fontsize=5.6,fontweight="bold")
    ax.text(.245,.782,"pitch = 0° · FOV = 90°",ha="center",fontsize=5.0,color="#5B6770")
    tiny_arrow(ax,(.29,.852),(.335,.852))
    draw_panorama_icon(ax,(.345,.815),(.155,.075))
    ax.text(.422,.795,"North-aligned 360° panorama",ha="center",fontsize=5.6,fontweight="bold")
    ax.text(.422,.782,"GPU single-source optimal seams",ha="center",fontsize=5.0,color="#5B6770")
    tiny_arrow(ax,(.505,.852),(.55,.852))
    for i,(name,color) in enumerate((("Semantics",COLORS["green"]),("DINOv2",COLORS["blue"]),("Depth",COLORS["orange"]))):
        tiny_box(ax,(.558,.817+i*.027),(.105,.022),name,face="white",edge=color)
    ax.text(.610,.795,"Visual evidence",ha="center",fontsize=5.6,fontweight="bold")
    tiny_arrow(ax,(.67,.852),(.71,.852))
    tiny_box(ax,(.716,.812),(.105,.083),"Weather + sun","ERA5 · solar geometry\nSOLWEIG-GPU labels",face="white",edge=COLORS["orange"])
    tiny_arrow(ax,(.825,.852),(.858,.852))
    tiny_box(ax,(.865,.812),(.105,.083),"Space × date × hour","8,975 sites\n14 hot-weather dates",face="white",edge=COLORS["purple"])

    # b | directional tokens, context queries and attention-based multitask model.
    center=(.265,.555); ring=.085
    directions=["N","NE","E","SE","S","SW","W","NW"]
    token_points=[]
    for i,theta in enumerate(np.linspace(np.pi/2,np.pi/2-2*np.pi,8,endpoint=False)):
        px,py=center[0]+ring*np.cos(theta),center[1]+ring*np.sin(theta)
        token_points.append((px,py));ax.add_patch(Circle((px,py),.018,facecolor="#DCEAF7",edgecolor=COLORS["blue"],lw=.6,zorder=3))
        ax.text(px,py,directions[i],ha="center",va="center",fontsize=5.0,fontweight="bold",zorder=4)
        tiny_arrow(ax,(px+(center[0]-px)*.20,py+(center[1]-py)*.20),
                   (center[0]+(px-center[0])*.22,center[1]+(py-center[1])*.22),color="#7D91A4",lw=.55)
    ax.add_patch(Circle(center,.031,facecolor="#E9F4EE",edgecolor=COLORS["teal"],lw=1,zorder=4))
    ax.text(*center,"Cross-\nattention",ha="center",va="center",fontsize=5.0,fontweight="bold",zorder=5)
    tiny_box(ax,(.035,.605),(.115,.062),"Semantic structure","21 classes · SVF/GVI",edge=COLORS["green"])
    tiny_box(ax,(.035,.485),(.115,.062),"Dynamic query","weather · solar",edge=COLORS["orange"])
    tiny_box(ax,(.035,.425),(.115,.045),"Azimuth encoding","true direction",edge=COLORS["purple"])
    for sy in (.636,.516,.447): tiny_arrow(ax,(.15,sy),(.185,.545+(sy-.53)*.15))
    tiny_arrow(ax,(.352,.555),(.385,.59));tiny_arrow(ax,(.352,.555),(.385,.49))
    tiny_box(ax,(.39,.558),(.085,.065),"Shade head","hourly fraction",edge=COLORS["teal"])
    tiny_box(ax,(.39,.458),(.085,.065),"Tmrt head","shade-conditioned",edge=COLORS["orange"])
    ax.text(.265,.424,"Global DINOv2 + 8 geographic directional tokens",ha="center",fontsize=5.0,color="#5B6770")

    # c | deterministic UTCI calculation and three output quantities.
    tiny_box(ax,(.535,.585),(.075,.060),"Tmrt","predicted",edge=COLORS["orange"])
    tiny_box(ax,(.535,.485),(.075,.060),"Ta · RH · v","weather",edge=COLORS["blue"])
    tiny_arrow(ax,(.612,.615),(.646,.565));tiny_arrow(ax,(.612,.515),(.646,.555))
    tiny_box(ax,(.650,.525),(.070,.072),"UTCI","deterministic",face="#FFF9E8",edge=COLORS["purple"])
    ax.text(.627,.449,"Three hourly outputs",ha="center",fontsize=5.4,fontweight="bold")
    ax.text(.627,.431,"Shade · Tmrt · UTCI",ha="center",fontsize=5.0,color="#5B6770")

    # d | explicit space/weather isolation and ensemble.
    for i,(title,sub,color) in enumerate((("Train","places + dates",COLORS["green"]),("Validation","unseen place/date",COLORS["orange"]),("Test","3 independent views",COLORS["purple"]))):
        tiny_box(ax,(.785,.610-i*.078),(.18,.058),title,sub,face="white",edge=color)
        if i<2: tiny_arrow(ax,(.875,.607-i*.078),(.875,.592-i*.078),lw=.55)
    ax.text(.875,.433,"5-seed ensemble · no Test tuning",ha="center",fontsize=5.0,color="#5B6770")

    # e | thermal costs on a maintained road graph and three route objectives.
    pts=draw_route_graph(ax,(.038,.090),.250,[([0,1,2,3,4,5],COLORS["blue"],.004),
                                              ([0,6,7,2,3,4,5],COLORS["green"],0),
                                              ([0,6,7,8,4,5],COLORS["orange"],-.004)])
    ax.scatter([pts[0,0]],[pts[0,1]],s=16,c="#111111",zorder=6)
    ax.scatter([pts[5,0]],[pts[5,1]],s=28,c="#F1A12A",marker="*",edgecolor="#111",lw=.4,zorder=6)
    tiny_arrow(ax,(.292,.174),(.322,.174))
    tiny_box(ax,(.330,.233),(.135,.055),"Shortest","distance baseline",edge=COLORS["blue"])
    tiny_box(ax,(.330,.157),(.135,.055),"Shade priority","lower Tmrt exposure",edge=COLORS["green"])
    tiny_box(ax,(.330,.081),(.135,.055),"UTCI priority","thermal comfort",edge=COLORS["orange"])
    ax.text(.250,.045,"Latest OSM graph · topology repair · grade-separated turns · route reliability",ha="center",fontsize=5.0,color="#5B6770")

    # f | conservative visual evidence cascade ending in an actionable or abstained state.
    draw_panorama_icon(ax,(.532,.221),(.128,.060))
    ax.text(.596,.205,"Street scene",ha="center",fontsize=5.0,fontweight="bold")
    tiny_arrow(ax,(.664,.251),(.693,.274))
    gates=[("Public walk/cycle",COLORS["blue"]),("Shade need (SVF)",COLORS["purple"]),("Existing trees",COLORS["green"]),("Depth + siting",COLORS["orange"])]
    for i,(name,color) in enumerate(gates):
        yy=.259-i*.052;tiny_box(ax,(.700,yy),(.118,.037),name,edge=color)
        if i<3: tiny_arrow(ax,(.759,yy-.002),(.759,yy-.013),lw=.5)
    tiny_arrow(ax,(.818,.181),(.832,.181))
    ax.plot([.832,.832],[.099,.278],color="#46525C",lw=.65,zorder=4)
    for branch_y in (.278,.219,.160,.101):
        tiny_arrow(ax,(.832,branch_y),(.846,branch_y),lw=.55)
    tiny_box(ax,(.850,.256),(.115,.044),"Maintain","adequate / existing",face="#F4FAF4",edge=COLORS["green"])
    tiny_box(ax,(.850,.197),(.115,.044),"Tree first","planner review",face="#FFF9E8",edge=COLORS["orange"])
    tiny_box(ax,(.850,.138),(.115,.044),"Abstain","insufficient evidence",face="#F5F2F8",edge=COLORS["purple"])
    tiny_box(ax,(.850,.079),(.115,.044),"Shade facility","supplement only",face="white",edge=COLORS["grey"])
    ax.text(.750,.043,"Evidence overlay and scenario estimate — screening support, not automatic construction",ha="center",fontsize=5.0,color="#5B6770")

    payload = {
        "sites": 8975, "directions": 12, "hot_weather_dates": 14,
        "model_inputs": ["semantic structure", "global DINOv2", "eight directional DINOv2 windows", "azimuth encoding", "weather", "solar geometry"],
        "outputs": ["shade", "Tmrt", "UTCI"],
        "route_objectives": ["shortest", "shade", "UTCI"],
        "planner_states": ["maintain", "tree-first review", "abstain", "shade-facility supplement"],
        "figure_contract": {"archetype": "schematic-led composite", "backend": "Python/matplotlib", "style_adaptation": "modular pastel panels; original drawing"},
    }
    (SOURCE / "Fig1_architecture.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return save(fig, 1, "workflow_and_architecture")


def figure2() -> list[str]:
    path = ROOT / "training/metrics/multidate_evaluation/test_view_overall_metrics.csv"
    data = pd.read_csv(path)
    data = data[data.target.isin(["shade", "tmrt_celsius", "utci_celsius"])].copy()
    data.to_csv(SOURCE / "Fig2_model_generalization.csv", index=False)
    view_order = ["unseen_points_seen_weather", "seen_points_unseen_weather", "unseen_points_unseen_weather"]
    view_labels = ["Unseen places", "Unseen weather", "Unseen places\n+ weather"]
    targets = [("shade", "Shade", "shade_r2", "shade_mae"),
               ("tmrt_celsius", "Tmrt", "tmrt_celsius_r2", "tmrt_celsius_mae"),
               ("utci_celsius", "UTCI", "utci_celsius_r2", "utci_celsius_mae")]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_IN, 3.9), sharex="col")
    fig.subplots_adjust(left=.07, right=.985, top=.91, bottom=.15, wspace=.34, hspace=.18)
    palette = [COLORS["blue"], COLORS["orange"], COLORS["teal"]]
    for col, (target, title, r2_col, mae_col) in enumerate(targets):
        subset = data[data.target.eq(target)].set_index("dataset_split").loc[view_order]
        x = np.arange(3)
        axes[0,col].bar(x, subset[r2_col].astype(float), color=palette, width=.68)
        axes[0,col].set_ylim(0, 1.04); axes[0,col].set_title(title, fontweight="bold")
        axes[0,col].axhline(0, color="#555", lw=.5)
        for i,v in enumerate(subset[r2_col].astype(float)): axes[0,col].text(i,v+.025,f"{v:.3f}",ha="center",fontsize=5.5)
        axes[1,col].bar(x, subset[mae_col].astype(float), color=palette, width=.68)
        for i,v in enumerate(subset[mae_col].astype(float)): axes[1,col].text(i,v+max(subset[mae_col].astype(float))*.035,f"{v:.3f}",ha="center",fontsize=5.5)
        axes[1,col].set_xticks(x, view_labels, rotation=0)
        axes[0,col].grid(axis="y", color="#E6EAED", lw=.6); axes[1,col].grid(axis="y", color="#E6EAED", lw=.6)
        axes[0,col].set_axisbelow(True); axes[1,col].set_axisbelow(True)
    axes[0,0].set_ylabel("Coefficient of determination (R²)")
    axes[1,0].set_ylabel("Mean absolute error")
    axes[1,0].text(-.52, -.42, "Shade: fraction  ·  Tmrt/UTCI: °C", transform=axes[1,0].transAxes, fontsize=5.5, color="#5B6770")
    label(axes[0,0], "a"); label(axes[1,0], "b")
    fig.text(.985,.025,"Strict Test views; ensemble mean. Joint unseen-place/unseen-weather n = 34,996 records.",ha="right",fontsize=5.5,color="#5B6770")
    return save(fig, 2, "model_generalization")


def figure3() -> list[str]:
    path = ROOT / "training/metrics/multidate_ablation_reporting/paired_bootstrap_degradation_ci.csv"
    data = pd.read_csv(path)
    data = data[(data.test_view == "unseen_points_unseen_weather") & (data.metric == "mae")].copy()
    data.to_csv(SOURCE / "Fig3_ablation_bootstrap.csv", index=False)
    order = ["no_global_dinov2", "no_directional_dinov2", "no_semantic_structure",
             "no_solar_geometry", "no_thermodynamic_weather", "no_wind", "no_shortwave_radiation",
             "mean_direction_pooling", "no_azimuth_encoding", "independent_prediction_heads"]
    display = {v:l for v,l in zip(order,["No global DINOv2", "No directional DINOv2", "No semantic structure",
        "No solar geometry", "No air temperature / RH", "No wind", "No shortwave radiation",
        "Mean directional pooling", "No azimuth encoding", "Independent prediction heads"])}
    fig, axes = plt.subplots(1,3,figsize=(WIDTH_IN,4.25),sharey=True)
    fig.subplots_adjust(left=.26,right=.985,top=.91,bottom=.12,wspace=.28)
    for col,(target,title,color) in enumerate([("shade","Shade MAE",COLORS["blue"]),("tmrt_celsius","Tmrt MAE",COLORS["orange"]),("utci_celsius","UTCI MAE",COLORS["teal"])]):
        ax=axes[col]; sub=data[data.target.eq(target)].set_index("variant_id").loc[order].reset_index(); y=np.arange(len(order))[::-1]
        value=sub.degradation_positive_is_worse.astype(float).to_numpy(); lo=sub.ci_lower.astype(float).to_numpy(); hi=sub.ci_upper.astype(float).to_numpy()
        ax.errorbar(value,y,xerr=np.vstack([value-lo,hi-value]),fmt="o",ms=4,color=color,ecolor=color,elinewidth=1,capsize=2)
        ax.axvline(0,color="#68737D",lw=.8,ls="--");ax.grid(axis="x",color="#E6EAED",lw=.6);ax.set_axisbelow(True);ax.set_title(title,fontweight="bold")
        ax.set_xlabel("Increase in MAE after ablation")
        if col==0: ax.set_yticks(y,[display[v] for v in order])
        label(ax, chr(ord('a')+col))
    fig.text(.985,.025,"Points show paired degradation; bars are 95% point-cluster bootstrap intervals (1,000 replicates).",ha="right",fontsize=5.5,color="#5B6770")
    return save(fig,3,"ablation_effects")


def route_case() -> dict:
    payload={"origin":{"crs":"EPSG:4326","longitude":118.7619971,"latitude":32.0446725},
             "destination":{"crs":"EPSG:4326","longitude":118.7785228,"latitude":32.0739944},
             "hour":17,"mode":"walk","objective":"shortest","algorithm":"astar",
             "max_detour_ratio":.40,"uncertainty_weight":0.0}
    request=Request("http://127.0.0.1:8765/api/compare",data=json.dumps(payload).encode(),
                    headers={"Content-Type":"application/json"},method="POST")
    with urlopen(request,timeout=180) as response: result=json.loads(response.read().decode())
    (SOURCE/"Fig4_route_case.json").write_text(json.dumps({"request":payload,"response":result},indent=2),encoding="utf-8")
    rows=[]
    for r in result["routes"]:
        rows.append({k:r[k] for k in ["objective","distance_m","detour_ratio","mean_shade","mean_tmrt","mean_utci","reliability_score","reliability_grade","uncertainty_mean"]})
    pd.DataFrame(rows).to_csv(SOURCE/"Fig4_route_metrics.csv",index=False)
    return result


def figure4() -> list[str]:
    result=route_case(); routes=result["routes"]
    fig=plt.figure(figsize=(WIDTH_IN,4.25));gs=fig.add_gridspec(1,2,width_ratios=[1.45,1],left=.055,right=.985,top=.91,bottom=.12,wspace=.25)
    ax=fig.add_subplot(gs[0]);label(ax,"a");ax.set_title("Central-city route alternatives: Hanzhongmen → Xuanwumen, 17:00",loc="left",pad=5)
    all_xy=np.vstack([np.asarray(r["geometry"]["coordinates"],float) for r in routes]);xmin,ymin=all_xy.min(0);xmax,ymax=all_xy.max(0);pad=max(xmax-xmin,ymax-ymin)*.12
    context=json.loads((ROOT/"routing/app/frontend/assets/local_context.json").read_text(encoding="utf-8"))
    for feature in context["features"]:
        xy=np.asarray(feature["geometry"]["coordinates"],float)
        if xy.ndim!=2 or len(xy)<2: continue
        if xy[:,0].max()<xmin-pad or xy[:,0].min()>xmax+pad or xy[:,1].max()<ymin-pad or xy[:,1].min()>ymax+pad: continue
        ax.plot(xy[:,0],xy[:,1],color="#D9DEE2",lw=.35,zorder=1)
    route_style={"shortest":("Shortest",COLORS["blue"],1.5,"-"),"shade":("Shade priority",COLORS["teal"],1.8,"--"),"utci":("UTCI priority",COLORS["orange"],1.6,"-")}
    for r in routes:
        xy=np.asarray(r["geometry"]["coordinates"],float);name,color,lw,ls=route_style[r["objective"]]
        ax.plot(xy[:,0],xy[:,1],color=color,lw=lw,ls=ls,label=name,zorder=3)
    ax.scatter(all_xy[0,0],all_xy[0,1],s=30,color="black",zorder=5,label="Origin")
    ax.scatter(all_xy[len(routes[0]["geometry"]["coordinates"])-1,0],all_xy[len(routes[0]["geometry"]["coordinates"])-1,1],s=55,marker="*",color="#F4A340",edgecolor="black",lw=.6,zorder=5,label="Destination")
    ax.set_xlim(xmin-pad,xmax+pad);ax.set_ylim(ymin-pad,ymax+pad);ax.set_aspect("equal");ax.set_xlabel("Easting (m), EPSG:32650");ax.set_ylabel("Northing (m), EPSG:32650")
    ax.legend(loc="lower right",ncol=1,frameon=True,framealpha=.92,edgecolor="#CCD3D8")
    ax=fig.add_subplot(gs[1]);ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1);label(ax,"b");ax.set_title("Route-level trade-offs",loc="left",pad=5)
    names=[route_style[r["objective"]][0] for r in routes];colors=[route_style[r["objective"]][1] for r in routes]
    x0=.34; colw=.205; rowh=.105; top=.84
    ax.text(.03,top+.045,"Metric",fontsize=6,fontweight="bold",va="center")
    for j,(name,color) in enumerate(zip(names,colors)):
        ax.add_patch(plt.Rectangle((x0+j*colw,top),colw-.012,.075,facecolor=color,alpha=.16,edgecolor=color,lw=.7))
        ax.text(x0+j*colw+(colw-.012)/2,top+.038,name.replace(" priority","\npriority"),ha="center",va="center",fontsize=5.4,fontweight="bold",color=color)
    metrics=[
        ("Distance (km)",[r["distance_m"]/1000 for r in routes],lambda v:f"{v:.2f}"),
        ("Detour (%)",[r["detour_ratio"]*100 for r in routes],lambda v:f"{v:.1f}"),
        ("Mean shade",[r["mean_shade"] for r in routes],lambda v:f"{v:.3f}"),
        ("Mean Tmrt (°C)",[r["mean_tmrt"] for r in routes],lambda v:f"{v:.2f}"),
        ("Mean UTCI (°C)",[r["mean_utci"] for r in routes],lambda v:f"{v:.2f}"),
        ("Reliability",[r["reliability_score"] for r in routes],lambda v:f"{v:.1f}"),
    ]
    for i,(metric_name,values,formatter) in enumerate(metrics):
        y=top-(i+1)*rowh
        ax.add_patch(plt.Rectangle((.02,y),.96,rowh-.008,facecolor="#F7F9FA" if i%2==0 else "white",edgecolor="#D9E0E4",lw=.45))
        ax.text(.03,y+(rowh-.008)/2,metric_name,fontsize=5.8,fontweight="bold",va="center")
        for j,(value,color) in enumerate(zip(values,colors)):
            ax.text(x0+j*colw+(colw-.012)/2,y+(rowh-.008)/2,formatter(value),ha="center",va="center",fontsize=6,color=color,fontweight="bold")
    shade_gain=(routes[1]["mean_shade"]-routes[0]["mean_shade"])*100
    utci_drop=routes[0]["mean_utci"]-routes[2]["mean_utci"]
    ax.add_patch(FancyBboxPatch((.03,.05),.92,.095,boxstyle="round,pad=.012",facecolor="#EEF6F2",edgecolor=COLORS["teal"],lw=.7))
    ax.text(.49,.098,f"Shade route: +{shade_gain:.1f} percentage points shade   ·   UTCI route: −{utci_drop:.2f} °C",ha="center",va="center",fontsize=5.7,fontweight="bold",color=COLORS["teal"])
    fig.text(.055,.025,"Three objectives share one maintained active-travel graph. Reliability is reported, not optimized as a fourth route.",fontsize=5.5,color="#5B6770")
    return save(fig,4,"route_comparison")


def rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im: return np.asarray(im.convert("RGB")).copy()


def semantic_groups(path: Path) -> np.ndarray:
    archive=np.load(path);labels=archive["class_labels"].astype(str);classes=archive["semantic_class"]
    canvas=np.full((*classes.shape,3),[210,214,220],dtype=np.uint8)
    palette={"Sky":[93,173,226],"Vegetation":[42,157,86],"Terrain":[125,184,106],"Sidewalk":[244,162,97],"Pedestrian Area":[250,190,110],"Bike Lane":[241,196,90],"Crosswalk - Plain":[255,220,150],"Road":[104,112,124],"Service Lane":[125,132,143],"Building":[174,126,91],"Wall":[153,117,92]}
    for index,name in enumerate(labels):
        if name in palette: canvas[classes==index]=palette[name]
    return canvas


def figure5() -> list[str]:
    point="6155";heading=270
    pano=next((ROOT/"streetscape/data/panorama/stage_04_formal/images").glob(f"{point}_*_north_pano.png"))
    semantic=ROOT/f"streetscape/data/semantic/stage_06b_spherical_fusion_100points/{point}__mapillary_semantic_spherical.npz"
    depth_path=ROOT/f"streetscape/data/geometry/stage_10c_geometry_walkable_benchmark_30points/depth_cache/{point}_{heading:03d}_depth_anything_v2.npz"
    req=Request(f"http://127.0.0.1:8765/api/planner/points/{point}/analyze",method="POST")
    with urlopen(req,timeout=180) as response: payload=json.loads(response.read().decode())
    overlay=np.asarray(Image.open(io.BytesIO(base64.b64decode(payload["overlay_png_base64"]))).convert("RGB"))
    terminal=pd.read_csv(ROOT/"streetscape/data/planning/stage_21_terminal_resolution_30points/final_point_planning_decisions_30points.csv",dtype={"point_id":str})
    row=terminal[terminal.point_id.eq(point)].iloc[0];depth=np.load(depth_path)["depth_m"].astype(float)
    pd.DataFrame([{"point_id":point,"SVF_V2":row.SVF_V2,"GVI_V2":row.GVI_V2,"walkable_ratio":row.retained_walkable_ratio_mean,"tree_evidence":row.season_aware_tree_evidence_count}]).to_csv(SOURCE/"Fig5_planner_evidence.csv",index=False)
    fig=plt.figure(figsize=(WIDTH_IN,4.15));gs=fig.add_gridspec(2,6,height_ratios=[1,1.12],left=.035,right=.985,top=.92,bottom=.08,wspace=.55,hspace=.48)
    axes=[fig.add_subplot(gs[0,0:2]),fig.add_subplot(gs[0,2:4]),fig.add_subplot(gs[0,4:6]),fig.add_subplot(gs[1,4:6]),fig.add_subplot(gs[1,0:4])]
    a,b,c,d,e=axes
    for ax in (a,b,c,e):ax.set_xticks([]);ax.set_yticks([]);[s.set_visible(False) for s in ax.spines.values()]
    a.imshow(rgb(pano));a.set_title("North-aligned panorama");label(a,"a")
    b.imshow(semantic_groups(semantic));b.set_title("Semantic evidence");label(b,"b")
    lo,hi=np.nanpercentile(depth,[2,98]);im=c.imshow(depth,cmap="magma_r",vmin=lo,vmax=hi);c.set_title("Monocular depth (270° view)");label(c,"c")
    cb=fig.colorbar(im,ax=c,fraction=.047,pad=.02);cb.set_label("Estimated depth (m)",fontsize=6);cb.ax.tick_params(labelsize=5.5)
    d.axis("off");d.set_title("Shade-need gate");label(d,"d");d.add_patch(FancyBboxPatch((.03,.08),.94,.82,boxstyle="round,pad=.02",facecolor="#F4F7FB",edgecolor="#CBD5E1"))
    for i,(name,value,limit,color) in enumerate([("SVF",float(row.SVF_V2),.35,COLORS["blue"]),("GVI",float(row.GVI_V2),.50,COLORS["green"]),("Visible walkable",float(row.retained_walkable_ratio_mean),.20,"#F0A229")]):
        y=.75-i*.2;d.text(.08,y,name,fontsize=6.2);d.add_patch(plt.Rectangle((.42,y-.025),.43,.055,color="#DEE5EA"));d.add_patch(plt.Rectangle((.42,y-.025),.43*min(value/limit,1),.055,color=color));d.text(.91,y,f"{value:.3f}",ha="right",fontsize=6)
    d.text(.08,.17,"SVF > 0.15  →  localization proceeds",fontsize=6,fontweight="bold",color=COLORS["blue"])
    d.text(.08,.08,"Tree-first corridor review",fontsize=6,fontweight="bold",color=COLORS["teal"])
    e.imshow(overlay);e.set_title("Semi-transparent planning candidates (screening, not construction design)");label(e,"e")
    for ax1,ax2 in ((a,b),(b,c)):
        p=ax1.get_position();q=ax2.get_position();fig.add_artist(FancyArrowPatch((p.x1+.004,(p.y0+p.y1)/2),(q.x0-.004,(q.y0+q.y1)/2),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=8,color="#718096"))
    p=c.get_position();q=d.get_position();fig.add_artist(FancyArrowPatch(((p.x0+p.x1)/2,p.y0-.004),((q.x0+q.x1)/2,q.y1+.004),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=8,color="#718096"))
    p=d.get_position();q=e.get_position();fig.add_artist(FancyArrowPatch((p.x0-.004,(p.y0+p.y1)/2),(q.x1+.004,(q.y0+q.y1)/2),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=8,color="#718096"))
    fig.text(.035,.018,"Semantic, depth and walkable-space outputs are automatic visual proxies; intervention locations require planner review. No local image content was edited.",fontsize=5.3,color="#5B6770")
    return save(fig,5,"planner_evidence_chain")


def boundary_rings(path: Path) -> list[np.ndarray]:
    rings=[]
    with path.open("rb") as stream:
        stream.read(100)
        while True:
            head=stream.read(8)
            if not head:break
            _,words=struct.unpack(">2i",head);content=stream.read(words*2);shape=struct.unpack("<i",content[:4])[0]
            if shape not in {5,15,25}:continue
            n_parts,n_points=struct.unpack("<2i",content[36:44]);parts=list(struct.unpack(f"<{n_parts}i",content[44:44+4*n_parts]));off=44+4*n_parts
            xy=np.frombuffer(content,dtype="<f8",count=n_points*2,offset=off).reshape(n_points,2).copy();bounds=parts+[n_points]
            rings.extend(xy[bounds[i]:bounds[i+1]] for i in range(n_parts) if bounds[i+1]-bounds[i]>=3)
    return rings


def figure6() -> list[str]:
    data=pd.read_csv(ROOT/"streetscape/figures/stage_33_citywide_terminal_planning/source_data_8975.csv.gz")
    data[["point_id","utm_x","utm_y","decision_group","SVF","GVI","walkable_direction_count","tree_evidence_direction_count"]].to_csv(SOURCE/"Fig6_citywide_terminal_decisions.csv.gz",index=False,compression="gzip")
    groups={"excluded_or_abstained":("Excluded / abstained",COLORS["grey"]),"maintain_or_no_intervention":("Maintain / no intervention","#7896A5"),"existing_tree_review":("Existing-tree / phenology review",COLORS["green"]),"optimization_eligible":("Tree-priority candidate",COLORS["orange"])}
    fig=plt.figure(figsize=(WIDTH_IN,4.3));gs=fig.add_gridspec(2,2,width_ratios=[1.55,1],height_ratios=[1,1],left=.06,right=.985,top=.92,bottom=.11,wspace=.27,hspace=.38)
    ax=fig.add_subplot(gs[:,0]);label(ax,"a");ax.set_title("Terminal shade-planning decisions across 8,975 sites",loc="left")
    for ring in boundary_rings(ROOT/"data/raw/Nanjing_center_boundary/Nanjing_center_UTM50N.shp"):ax.plot(ring[:,0],ring[:,1],color="#394B55",lw=.75,zorder=4)
    for key in ["excluded_or_abstained","maintain_or_no_intervention","existing_tree_review"]:
        sub=data[data.decision_group.eq(key)];ax.scatter(sub.utm_x,sub.utm_y,s=2.2,color=groups[key][1],alpha=.35,lw=0,rasterized=True,label=f"{groups[key][0]} ({len(sub):,})")
    sub=data[data.decision_group.eq("optimization_eligible")];ax.scatter(sub.utm_x,sub.utm_y,s=50,marker="*",color=COLORS["orange"],edgecolor="white",lw=.5,zorder=5,label="Tree-priority candidate (5)")
    ax.set_aspect("equal");ax.set_xlabel("Easting (m), EPSG:32650");ax.set_ylabel("Northing (m), EPSG:32650");ax.legend(loc="lower left",frameon=True,framealpha=.92,markerscale=2)
    ax=fig.add_subplot(gs[0,1]);label(ax,"b");ax.set_title("Mutually exclusive outcomes",loc="left")
    order=["existing_tree_review","maintain_or_no_intervention","excluded_or_abstained","optimization_eligible"];counts=[int(data.decision_group.eq(k).sum()) for k in order];y=np.arange(4)[::-1]
    ax.barh(y,counts,color=[groups[k][1] for k in order],height=.58);ax.set_yticks(y,[groups[k][0] for k in order]);ax.set_xlabel("Sites")
    for yy,c in zip(y,counts):ax.text(c+max(counts)*.02,yy,f"{c:,}  ({c/len(data)*100:.1f}%)",va="center",fontsize=5.6)
    ax.set_xlim(0,max(counts)*1.25);ax.grid(axis="x",color="#E6EAED",lw=.6);ax.set_axisbelow(True)
    ax=fig.add_subplot(gs[1,1]);label(ax,"c");ax.set_title("Conservative evidence convergence",loc="left");ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    stages=[("8,975","citywide sites",COLORS["blue"]),("6,014","GPU-localized",COLORS["teal"]),("5,990 / 24","tree evidence\nyes / no",COLORS["green"]),("5","planner-review\ncandidates",COLORS["orange"])]
    for i,(n,t,c) in enumerate(stages):
        x=.01+i*.25;box(ax,(x,.33),(.205,.38),n,t,c,title_size=8 if i!=2 else 7)
        if i<3:arrow(ax,(x+.205,.52),(x+.245,.52))
    ax.text(.01,.10,"Pending 0  ·  failures 0  ·  CPU fallback 0",fontsize=6,fontweight="bold",color="#4B5963")
    fig.text(.06,.02,"Candidates support planner review; they are not automatic planting sites or validated causal cooling effects.",fontsize=5.5,color="#5B6770")
    return save(fig,6,"citywide_planning_outcomes")


def main() -> int:
    OUT.mkdir(parents=True,exist_ok=True);SOURCE.mkdir(parents=True,exist_ok=True)
    outputs={"Fig1":figure1(),"Fig2":figure2(),"Fig3":figure3(),"Fig4":figure4(),"Fig5":figure5(),"Fig6":figure6()}
    summary={"step":"stage_36","status":"PASS","target_journal":"Computers, Environment and Urban Systems","backend":"Python/matplotlib","figure_count":6,"outputs":outputs,"generated_or_ai_artwork":False}
    (ROOT/"author_workspace/publication/ceus/figure_suite_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
