import json, os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "results_exp3_paralelismo.jsonl"
if not os.path.exists(RESULTS_FILE):
    print(f"[ERRO] Arquivo não encontrado: {RESULTS_FILE}"); sys.exit(1)

rows = [json.loads(line) for line in open(RESULTS_FILE)]
df = pd.DataFrame(rows)
print(df[["n_workers","mean_time_s","speedup_real","speedup_ideal","efficiency_pct"]].to_string(index=False))

plt.rcParams.update({"figure.facecolor":"#1a1d27","axes.facecolor":"#12141f",
    "axes.edgecolor":"#2d3150","axes.labelcolor":"#c9d1e9","xtick.color":"#8b91b8",
    "ytick.color":"#8b91b8","text.color":"#e8eaf6","grid.color":"#2d3150",
    "grid.linestyle":"--","grid.alpha":0.5,"legend.facecolor":"#22263a",
    "legend.edgecolor":"#2d3150","font.size":11,"axes.titlesize":13,"axes.titleweight":"bold"})
ACCENT,GREEN,RED,YELLOW,WHITE = "#6c63ff","#00d4aa","#ff6b6b","#ffd93d","#c9d1e9"
x = df["n_workers"]; xl = x.astype(str)
info = f"Cluster: {int(df['cluster_nodes'].iloc[0])} nos, {int(df['cluster_total_cpus'].iloc[0])} CPUs" if "cluster_nodes" in df.columns else ""

# Speedup
fig,ax = plt.subplots(figsize=(9,5))
fig.suptitle(f"Speedup Real vs. Ideal — Ray Data (QP3)\n{info}", fontsize=12)
ax.plot(x,df["speedup_real"],marker="o",color=ACCENT,linewidth=2.5,markersize=9,label="Speedup real",zorder=3)
ax.plot(x,df["speedup_ideal"],marker="^",color=YELLOW,linewidth=1.8,markersize=8,linestyle="--",label="Speedup ideal (linear)",zorder=3)
ax.fill_between(x,df["speedup_real"],df["speedup_ideal"],color=RED,alpha=0.13,label="Gap (overhead)")
for _,r in df.iterrows():
    ax.annotate(f"{r['speedup_real']:.2f}x",xy=(r["n_workers"],r["speedup_real"]),xytext=(0,10),textcoords="offset points",ha="center",color=WHITE,fontsize=9)
ax.set_xlabel("Número de Workers (concurrency)"); ax.set_ylabel("Speedup (x)")
ax.set_xticks(x); ax.legend(); ax.grid(zorder=0)
plt.tight_layout(); plt.savefig("grafico_speedup.png",dpi=150,bbox_inches="tight"); plt.close()
print("✓ grafico_speedup.png")

# Eficiência
fig,ax = plt.subplots(figsize=(9,5))
fig.suptitle(f"Eficiência Paralela (%) — Ray Data (QP3)\n{info}", fontsize=12)
colors=[GREEN if e>=70 else YELLOW if e>=50 else RED for e in df["efficiency_pct"]]
bars=ax.bar(xl,df["efficiency_pct"],color=colors,alpha=0.88,zorder=3,width=0.5)
for b,v in zip(bars,df["efficiency_pct"]):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()-4,f"{v:.1f}%",ha="center",va="top",color="#0f1117",fontsize=10,fontweight="bold")
ax.axhline(100,color=YELLOW,linestyle="--",linewidth=1.2,label="100% (ideal)")
ax.axhline(70,color=GREEN,linestyle=":",linewidth=1.2,label="70% (limiar aceitável)")
ax.set_xlabel("Número de Workers"); ax.set_ylabel("Eficiência (%)"); ax.set_ylim(0,120)
ax.legend(fontsize=9); ax.grid(axis="y",zorder=0)
plt.tight_layout(); plt.savefig("grafico_eficiencia.png",dpi=150,bbox_inches="tight"); plt.close()
print("✓ grafico_eficiencia.png")

# Tempo
fig,ax = plt.subplots(figsize=(9,5))
fig.suptitle(f"Tempo Médio de Execução — Ray Data (QP3)\n{info}", fontsize=12)
ax.bar(xl,df["mean_time_s"],color=ACCENT,alpha=0.85,zorder=3,width=0.5)
ax.errorbar(xl,df["mean_time_s"],yerr=df["std_time_s"],fmt="none",color=WHITE,capsize=6,linewidth=1.8,zorder=4)
for i,r in df.iterrows():
    ax.text(i,r["mean_time_s"]+r["std_time_s"]+0.5,f"{r['mean_time_s']:.1f}s",ha="center",color=WHITE,fontsize=9)
ax.set_xlabel("Número de Workers"); ax.set_ylabel("Tempo médio (s)"); ax.grid(axis="y",zorder=0)
plt.tight_layout(); plt.savefig("grafico_tempo.png",dpi=150,bbox_inches="tight"); plt.close()
print("✓ grafico_tempo.png")