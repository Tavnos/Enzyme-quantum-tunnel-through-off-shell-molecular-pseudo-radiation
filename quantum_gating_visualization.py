"""Pre-reactive quantum gating: illustrative kinetic model, NOT experimental data.

Usage:
  python quantum_gating_visualization.py             # PNG + vector SVG
  python quantum_gating_visualization.py --gif       # also an animated GIF
  python quantum_gating_visualization.py --interactive  # adjustable kinetics

Requires: numpy, matplotlib; Pillow additionally for GIF.
Physical caution: r is a coarse effective charge separation, not an experimentally
validated atomic contact criterion. Only unbound approach is modeled. Tunneling
is a distinct post-binding chemical step; the model predicts no pre-contact reaction.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch, Arc

OUT = Path(__file__).resolve().parent
KB = 1.380649e-23
HBAR = 1.054571817e-34
QE = 1.602176634e-19
M_H = 1.67262192369e-27
M_D = 3.3435837724e-27
INK, BLUE, TEAL, ORANGE, GREY = '#16253a', '#356cbe', '#009e92', '#cc683e', '#6c798d'

@dataclass
class Params:
    l_debye: float = .79   # nm; approx 0.15 M monovalent electrolyte in water
    l_bjerrum: float = .71 # nm; water ~298 K
    dg0: float = 2.0      # kBT; higher-energy ready substate when ligand is distant
    coupling: float = 1.5 # model parameter: differential electrostatic coupling
    tau_fast: float = .25 # ns, illustrative FAST/local response
    tau_slow: float = 20. # ns, illustrative SLOW protein reorganization
    duration: float = 5.0 # ns; idealized directed approach, not Brownian trajectory
    r_initial: float = 3.0 # nm, effective charge separation
    r_final: float = .80  # nm, closest unbound effective charge separation
    barrier_eV: float = .20 # illustrative rectangular chemical barrier height


def screened_energy(r, p):
    """Dimensionless Debye-Hueckel UNIT-pair interaction scale, U/kBT."""
    r = np.maximum(np.asarray(r, dtype=float), .01)
    return (p.l_bjerrum/r)*np.exp(-r/p.l_debye)


def ready_eq(r, p):
    """Two-state equilibrium ready fraction; differential shift is model-assumed."""
    dg = p.dg0 - p.coupling*screened_energy(r, p)
    return 1/(1+np.exp(np.clip(dg, -80, 80)))


def solve(p, n=2501):
    t = np.linspace(0, p.duration, n)
    r = p.r_initial + (p.r_final-p.r_initial)*(t/p.duration)
    eq = ready_eq(r, p)
    baseline = float(1/(1+np.exp(p.dg0)))
    dt = t[1]-t[0]
    def relax(tau):
        x = np.empty(n)
        x[0] = baseline
        # Exact update for frozen equilibrium target over each short timestep
        alpha = -np.expm1(-dt/tau)
        for i in range(1, n): x[i] = x[i-1] + alpha*(eq[i]-x[i-1])
        return x
    return t, r, eq, relax(p.tau_fast), relax(p.tau_slow), baseline


def tunneling(width_angstrom, mass, p):
    """WKB rectangular-barrier transmission, chemically illustrative only."""
    width_m = np.asarray(width_angstrom, dtype=float)*1e-10
    return np.exp(-2*width_m*np.sqrt(2*mass*p.barrier_eV*QE)/HBAR)


def canvas():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#b8c4d2','axes.labelcolor':INK,
                         'text.color':INK,'xtick.color':GREY,'ytick.color':GREY,
                         'grid.alpha':.2, 'savefig.facecolor':'white'})


def scene(ax, frac=1.0, show_labels=True):
    ax.set_aspect('equal'); ax.set_xlim(-.25, 5.9); ax.set_ylim(-1.65, 1.65); ax.axis('off')
    ax.add_patch(Circle((1.05,0),1.15,fc='#eaf1fa',ec=BLUE,lw=2))
    ax.add_patch(Circle((1.05,0),.46,fc='#d2e1f7',ec=BLUE,lw=1.2))
    ax.text(1.05, .06, 'ENZYME',ha='center',va='center',fontsize=11,fontweight='bold',color=INK)
    ax.text(1.05,-.23,'ready / resting',ha='center',va='center',fontsize=8,color=GREY)
    x = 5.25 - 1.94*frac
    ax.add_patch(Circle((x,0),.25,fc=ORANGE,ec='white',lw=1.2))
    ax.add_patch(Arc((x,0),1.15,1.15,theta1=30,theta2=330,ec=ORANGE,lw=1,alpha=.23))
    ax.annotate('',(x-.40,-.78),(2.35,-.78),arrowprops={'arrowstyle':'<->','color':GREY,'lw':1.1})
    if show_labels:
        ax.text(x,.5,'substrate',color=ORANGE,ha='center',fontsize=10,fontweight='bold')
        ax.text((x+2.35)/2,-1.06,'effective approach distance',color=GREY,ha='center',fontsize=8)
        ax.text(1.05,1.42,'enzyme conformational ensemble',ha='center',fontsize=9,color=BLUE)
        ax.text(3.85,-1.44,'no stable binding or reaction',ha='center',fontsize=9,color=GREY)


def static_figure(p):
    t,r,peq,pfast,pslow,base=solve(p)
    f=plt.figure(figsize=(14.8,10.5), facecolor='white')
    gs=f.add_gridspec(2,2,left=.065,right=.965,top=.855,bottom=.12,wspace=.24,hspace=.33,
                      height_ratios=[.85,1.0])
    ax0=f.add_subplot(gs[0,0]); ax1=f.add_subplot(gs[0,1]); ax2=f.add_subplot(gs[1,0]); ax3=f.add_subplot(gs[1,1])
    scene(ax0, .78)
    ax0.set_title('A   Approach geometry (schematic)',loc='left',fontweight='bold',pad=13)
    rr=np.linspace(p.r_final,3.4,300)
    ax1.plot(rr,screened_energy(rr,p),color=BLUE,lw=2.7,label=f'Debye length {p.l_debye:.2f} nm')
    ax1.plot(rr,screened_energy(rr,replace(p,l_debye=2.0)),color=GREY,lw=1.8,ls='--',label='Longer screening length: 2.0 nm')
    ax1.axvline(p.r_final,color=ORANGE,ls=':',lw=1.5)
    ax1.set(xlabel='Effective charge separation, r (nm)',ylabel=r'Unit-pair interaction magnitude, $|U|/k_BT$')
    ax1.set_title('B   Electrostatic interaction decays with distance',loc='left',fontweight='bold',pad=13)
    ax1.legend(frameon=False,fontsize=9); ax1.grid(alpha=.18)
    ax1.text(.97,.81,'Not a direct enzyme substate energy\nwithout differential coupling',transform=ax1.transAxes,
             ha='right',va='top',fontsize=8.5,color=GREY)
    ax2.axhline(100*base,color=INK,ls=':',lw=1.3,label='Distant-substrate baseline')
    ax2.plot(t,100*peq,color=GREY,ls='--',lw=2.0,label='Instantaneous equilibrium (upper reference)')
    ax2.plot(t,100*pfast,color=TEAL,lw=2.7,label=f'Fast local response (τ = {p.tau_fast:g} ns)')
    ax2.plot(t,100*pslow,color=ORANGE,lw=2.7,label=f'Slow response (τ = {p.tau_slow:g} ns)')
    ax2.set(xlabel='Illustrative unbound approach duration (ns)',ylabel='Ready-state population (%)',
            xlim=(0,p.duration))
    ax2.set_title('C   Can the conformational ensemble keep up?',loc='left',fontweight='bold',pad=13)
    ax2.legend(frameon=False,fontsize=9,loc='upper left'); ax2.grid(alpha=.2)
    widths=np.linspace(.25,.9,300)
    th=tunneling(widths,M_H,p); td=tunneling(widths,M_D,p)
    ax3.semilogy(widths,th,color=TEAL,lw=2.7,label='Hydrogen (¹H)')
    ax3.semilogy(widths,td,color=ORANGE,lw=2.7,label='Deuterium (²H)')
    ax3.axvline(.55,color=GREY,ls=':',lw=1.5)
    ax3.set(xlabel='Post-binding rectangular barrier width (Å)',ylabel='WKB transmission (dimensionless)',
            xlim=(widths[0],widths[-1]))
    ax3.set_title('D   Tunneling only AFTER a suitable bound geometry',loc='left',fontweight='bold',pad=13)
    ax3.legend(frameon=False,fontsize=9); ax3.grid(alpha=.2,which='both')
    ax3.text(.03,.05,'Barrier = 0.20 eV; 1D WKB illustration\nNot a fitted enzyme rate or intrinsic KIE',transform=ax3.transAxes,
             ha='left',va='bottom',fontsize=8.5,color=GREY)
    fig_title='Pre-reactive quantum gating | from substrate approach to tunneling-ready geometry'
    f.text(.065,.963,fig_title,ha='left',fontsize=17,fontweight='bold',color=INK)
    f.text(.065,.922,'A causal, two-state TOY MODEL — does not imply tunneling occurs before binding or demonstrate an enzyme-specific effect.',
           fontsize=10.1,color=GREY)
    f.text(.065,.045,'Toy parameters: T ≈ 298 K, water, λD = 0.79 nm, lB = 0.71 nm, ΔG₀ = 2 kBT, differential coupling = 1.5; straight-line approach, not diffusion.',
           fontsize=9,color=GREY)
    f.text(.065,.022,f'End of approach: baseline {100*base:.1f}% → fast {100*pfast[-1]:.1f}% / slow {100*pslow[-1]:.1f}%; all predictions conditional on assumed coupling and timescales.',
           fontsize=9,color=INK)
    f.savefig(OUT/'quantum_gating_model.png',dpi=185)
    f.savefig(OUT/'quantum_gating_model.svg')
    return f


def animation_gif(p):
    t,r,peq,pfast,pslow,base=solve(p)
    fig,(a,b)=plt.subplots(1,2,figsize=(11.5,4.9),gridspec_kw={'width_ratios':[1,1.05]})
    fig.subplots_adjust(top=.77,bottom=.2,wspace=.22,left=.045,right=.98)
    fig.suptitle('Can an approaching substrate bias the enzyme before binding?',fontsize=15,fontweight='bold',y=.96)
    frames=np.linspace(0,len(t)-1,85).astype(int)
    b.plot(t,100*peq,ls='--',color=GREY,lw=1.6,label='instantaneous equilibrium')
    b.plot(t,100*pfast,color=TEAL,lw=2.6,label='fast local mode')
    b.plot(t,100*pslow,color=ORANGE,lw=2.6,label='slow mode')
    b.axhline(100*base,color=INK,lw=1,ls=':')
    cursor=b.axvline(0,color=INK,alpha=.4)
    ptf,=b.plot([],[],'o',color=TEAL,ms=8); pts,=b.plot([],[],'o',color=ORANGE,ms=8)
    b.set(xlim=(0,p.duration),ylim=(100*base-1,100*np.max(peq)+2),xlabel='Unbound approach time (ns)',
          ylabel='Ready-state population (%)',title='Kinetic response vs equilibrium')
    b.grid(alpha=.17); b.legend(loc='upper left',frameon=False,fontsize=8)
    info=fig.text(.5,.055,'',ha='center',color=INK,fontsize=10)
    def update(n):
        idx=frames[n]
        a.clear()
        scene(a,idx/(len(t)-1),show_labels=True)
        a.set_title(f'Effective separation r = {r[idx]:.2f} nm',loc='left',fontsize=11)
        cursor.set_xdata([t[idx],t[idx]])
        ptf.set_data([t[idx]],[100*pfast[idx]])
        pts.set_data([t[idx]],[100*pslow[idx]])
        info.set_text(f't = {t[idx]:.2f} ns   |   equilibrium: {100*peq[idx]:.1f}%   fast: {100*pfast[idx]:.1f}%   slow: {100*pslow[idx]:.1f}%')
        return cursor,ptf,pts,info
    ani=FuncAnimation(fig,update,frames=len(frames),interval=75,blit=False,repeat=True)
    ani.save(OUT/'quantum_gating_approach.gif',writer=PillowWriter(fps=14),dpi=93)
    plt.close(fig)


def interactive(p):
    from matplotlib.widgets import Slider
    from matplotlib.gridspec import GridSpec
    fig=plt.figure(figsize=(10,6.7))
    fig.subplots_adjust(bottom=.36,top=.9)
    ax=fig.add_subplot(111)
    t,r,eq,fast,slow,base=solve(p)
    ln_eq,=ax.plot(t,100*eq,color=GREY,ls='--',label='instantaneous equilibrium')
    ln_f,=ax.plot(t,100*fast,color=TEAL,lw=2.5,label='fast mode')
    ln_s,=ax.plot(t,100*slow,color=ORANGE,lw=2.5,label='slow mode')
    ax.axhline(100*base,color=INK,ls=':',label='baseline')
    ax.set(xlabel='Unbound approach time (ns)',ylabel='Ready-state population (%)',
           title='Interactive pre-reactive gating toy model',ylim=(10,65))
    ax.legend(frameon=False); ax.grid(alpha=.15)
    a1=fig.add_axes([.2,.24,.65,.034]); a2=fig.add_axes([.2,.17,.65,.034]); a3=fig.add_axes([.2,.10,.65,.034])
    s1=Slider(a1,'Debye λ (nm)',.2,2.5,valinit=p.l_debye)
    s2=Slider(a2,'Coupling',0,6.,valinit=p.coupling)
    s3=Slider(a3,'τ fast (ns)',.05,8.,valinit=p.tau_fast)
    def refresh(_):
        q=replace(p,l_debye=s1.val,coupling=s2.val,tau_fast=s3.val)
        _,_,new_eq,new_fast,new_slow,_=solve(q)
        ln_eq.set_ydata(100*new_eq); ln_f.set_ydata(100*new_fast); ln_s.set_ydata(100*new_slow)
        fig.canvas.draw_idle()
    for s in (s1,s2,s3):s.on_changed(refresh)
    plt.show()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gif',action='store_true',help='also render animation')
    ap.add_argument('--interactive',action='store_true',help='open Matplotlib parameter sliders')
    args=ap.parse_args()
    canvas(); p=Params(); fig=static_figure(p)
    plt.close(fig)
    if args.gif: animation_gif(p)
    if args.interactive: interactive(p)
    print('Saved',OUT/'quantum_gating_model.png','and',OUT/'quantum_gating_model.svg')
    if args.gif: print('Saved',OUT/'quantum_gating_approach.gif')

if __name__=='__main__':main()
