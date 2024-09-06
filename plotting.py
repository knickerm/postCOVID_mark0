import numpy as np
from matplotlib import pyplot as plt

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D#
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.patches as mpatches


def plot_different_shocks(data, figsize = (30,15),inflationx = [0.92,1.1], wagex=[0.9,1.05], fragx=[0,1], cbx=[0,0.1], unempx=[0,1], bankruptx=[-0.01,0.2], savingsx = [0, 18000], colors_seq=[], shock=True, setzero = 3800, p_shock=0, cmap='viridis', cf=False, s=[1,2,3],
                        shocks=['original', 'ext_prod', 'price', 'price_and_ext_prod'], xaxis=[4000, 6000], linewidth=6.5, fontsize = 16, legendfontsize = 13, t_endx=120, no_shocks=False, save = '', ncol=2, red=False):
    
    fontsize= fontsize-1
    t_start = xaxis[0]
    t_end = 4009
    shockk = shock
    if True: #shock
        start = t_start- 10
        end = t_start + t_endx
    else:
        start = xaxis[0]
        end = xaxis[1]
    lw = linewidth
    fig, ax1 = plt.subplots(3, 3, figsize=figsize)
    if cf == False:
        left, bottom, width, height = [0.55, 0.485, 0.1, 0.125] #0.409, 0.815
    else:
        left, bottom, width, height = [0.55, 0.77, 0.1, 0.125]#[0.415, 0.81, 0.1, 0.125]
    ax2 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.2153, 0.485, 0.1, 0.125]
    ax3 = fig.add_axes([left, bottom, width, height])

    aleft, aright = [], []
    xdata = data[shocks[0]].index[start:end]-setzero
    
    #cmap = plt.get_cmap(cmap, len(shocks))

    if red == False:
        def cmap(i):
            cm = ['mediumblue', 'darkorange', 'green']
            return(cm[i])
    if red == True:
        def cmap(i):
            cm = ['mediumblue', 'darkorange', 'darkred']
            return(cm[i])

    linestyles = ['-', '--', '-.']
    
    
    for i, shock in enumerate(shocks):
        df = data[shock][start:end]
        print(shock, df.shape, start, end, data[shock].shape)
        if colors_seq!=[]:
            i = colors_seq[i]
        ax1[0, 2].plot(xdata, (df['Pavg']-1)*12, linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')
        ax1[0, 2].set_title('Inflation Rate, $\\pi$ (annual)', fontsize=fontsize+1)
        ax1[2, 0].plot(xdata, df['Wavg'], linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')
        ax1[2, 0].set_title(r'Real Wages, $\langle$ W $\rangle$', fontsize=fontsize)
        ax1[0, 0].plot(xdata, df['min-ytot'], linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock}')
        ax1[0, 0].set_title(r'Total Output, $\langle$ Y $\rangle$', fontsize=fontsize+1)
        ax1[1, 0].plot(xdata, df['S'], linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')    
        ax1[1, 0].set_title('Savings, S', fontsize=fontsize)
        ax1[0, 1].plot(xdata, df['u'], linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')
        ax1[0, 1].set_title('Unemployment, u', fontsize=fontsize+1)
        #ax1[1, 1].plot(xdata, df['bust-af'], linestyle='-', lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')
        #ax1[1, 1].set_title('Bankruptcy Rate', fontsize=fontsize)
        ax1[2, 2].plot(xdata, df['tau_tar'], linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'$\\tau^T$, {shock.replace("_", " ")} ')
        #ax1[2, 2].plot(xdata, df['tau_meas'], linestyle='--', lw=lw, c=cmap(i), label=f'$\\tau^R$, {shock.replace("_", " ")} ')
        ax1[2, 2].set_title('Expectation Anchor, $\\tau^T$', fontsize=fontsize+1)
        #ax1[1, 2].plot(xdata, df['Atot'], linestyle='-', lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')
        #ax1[1, 2].set_title('Total Cash Balance', fontsize=fontsize)
        ax1[1, 2].plot(xdata, df['rho']*12, linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")}')
        ax1[1, 2].set_title('Central Bank Interest,  $\\rho_0$ (annual)', fontsize=fontsize+1)
        ax1[1, 1].plot(xdata, df['frag'], linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')
        ax1[1, 1].set_title('Fragility, $\\Phi$', fontsize=fontsize+1)

        ax1[2, 1].plot(xdata, df['propensity'], linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")}')
        ax1[2, 1].set_title('Consumption Budget, $C_B$', fontsize=fontsize+1)

        
        # insets
        ax2.plot(xdata, df['bust-af'], linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')
        ax3.plot(xdata, df['Dtot'].div(df['min-ytot']), linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')
        #ax3.set_title(r'$\frac{Demand}{Output}$ Ratio', fontsize=fontsize)
        
    
    ax1[0, 2].set_ylim(inflationx)
    ax1[1, 2].set_ylim(cbx)
    ax1[2, 0].set_ylim(wagex)
    ax1[2, 2].set_ylim([-0.05,1.05])
    ax1[1, 0].set_ylim(savingsx)
    ax1[0, 0].set_ylim([0,15000])
    ax1[0, 1].set_ylim(unempx)
    ax1[1, 1].set_ylim(fragx)
    ax3.set_ylim(0.9,1.1)
    #ax1[2, 2].set_ylim(0.5,1.2)
    
    
    #ax3.axhline(y=1, linestyle='--', lw=2, c='black')
    ax1[1, 0].axhline(y=10000, linestyle='--', lw=2, c='black')
    ax1[0, 0].axhline(y=10000, linestyle='--', lw=2, c='black')
    #ax1[2, 2].axhline(y=1, linestyle='--', lw=2, c='black')
    #ax1[0, 1].axhline(y=0.11)
    ax1[0, 2].axhline(y=0.0275, linestyle='--', lw=2, c='black')
    #ax1[0, 1].axhline(y=0.0)
    #ax1[2, 1].axhline(y=0.9)
    for i in range(3):
        for j in range(3):
            if no_shocks==False:
                if 1 in s:
                    ax1[i, j].axvspan(4000-200-setzero,4005-200-setzero, facecolor='0.3', alpha=0.5)
                if 2 in s:
                    ax1[i, j].axvspan(4005-200-setzero, 4005-200-setzero+10, facecolor='0.6', alpha=0.5)
                #ax1[i, j].axvspan(4005-200-setzero+10, 4005-200-setzero+10+23, facecolor='0.8', alpha=0.5)
                if 3 in s:
                    ax1[i, j].axvspan(4005-200-setzero+10, 4000-200-setzero+33, facecolor='0.8', alpha=0.5)
            
            ax1[i,j].set_xticks([0, 24, 48, 72, 96])
            if i == 2:
                #ax1[i, j].set_xlabel('Time in Months', fontsize=fontsize)
                #ax1[i,j].set_xticks([0,20,40,60,80,100,120])
                
                #ax1[i,j].set_xticklabels([0,20,40,60,80,100,120])
                ax1[i,j].set_xticklabels(["Feb-20", "Feb-22","Feb-24","Feb-26","Feb-28",])
            else:
                #ax1[i,j].tick_params(axis='x', colors='white')
                plt.setp(ax1[i,j].get_xticklabels(), visible=False)
            ax1[i, j].tick_params(axis='both', labelsize=fontsize-3)
            #ax1[i, j].legend(fontsize=legendfontsize)

    #inset
    ax3.tick_params(axis='both', labelsize=fontsize-6)
    ax3.set_title(r'$\frac{Demand}{Output}$', fontsize=fontsize-6, pad=10)
    if no_shocks==False:
        if 1 in s:
            ax3.axvspan(4000-200-setzero,4005-200-setzero, facecolor='0.3', alpha=0.5)
        if 2 in s:
            ax3.axvspan(4005-200-setzero, 4005-200-setzero+10, facecolor='0.6', alpha=0.5)
        #ax3.axvspan(4005-200-setzero+10, 4005-200-setzero+10+23, facecolor='0.8', alpha=0.5)
        if 3 in s:
            ax3.axvspan(4005-200-setzero+10, 4000-200-setzero+33, facecolor='0.8', alpha=0.5)
    plt.setp(ax3.get_xticklabels(), visible=False)
    
    ax2.tick_params(axis='both', labelsize=fontsize-6)
    ax2.set_title('Bankruptcy Rate', fontsize=fontsize-6)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax2.set_ylim(bankruptx)
    if bankruptx[1] < 0.1:
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
        
    if no_shocks==False:
        if 1 in s:
            ax2.axvspan(4000-200-setzero,4005-200-setzero, facecolor='0.3', alpha=0.5)
        if 2 in s:
            ax2.axvspan(4005-200-setzero, 4005-200-setzero+10, facecolor='0.6', alpha=0.5)
        #ax2.axvspan(4005-200-setzero+10, 4005-200-setzero+10+23, facecolor='0.8', alpha=0.5)
        if 3 in s:
            ax2.axvspan(4005-200-setzero+10, 4000-200-setzero+33, facecolor='0.8', alpha=0.5)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    ax1[0, 1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax1[0, 2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1[1, 2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  
    #line = Line2D([0], [0], label='$\\tau^T$', color='black', lw=lw)
    #dotted_line = Line2D([0], [0], label='$\\tau^R$', linestyle='--', color='black', lw=lw)
    #ax1[2, 2].legend(handles=[line, dotted_line], fontsize=legendfontsize+6)
    
    if no_shocks:
        #ax1[0,0].set_ylim([9500,10500])
        #ax1[1,0].set_ylim([9000,14000])
        ax1[0, 0].set_ylim(savingsx)
        ax1[0,2].set_ylim(inflationx) #0.05
        ax1[1,2].set_ylim([0.01,0.025])
        ax1[0,1].set_ylim([0,0.02])
        ax1[1,1].set_ylim([0.3, 1])
        ax1[2,1].set_ylim([9850, 10300])
        ax3.set_ylim([0.99,1.02])
        ax1[0, 1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
        ax2.set_ylim([0,0.05])
    for i in range(3):
        for j in range(3):
            ax1[i,j].grid(alpha=0.7, zorder=0)
    ax3.grid(alpha=0.7, zorder=0)
    ax2.grid(alpha=0.7, zorder=0)
    handles, labels = ax1[0,0].get_legend_handles_labels()
    if ncol==3:
        fig.legend(handles, labels, fontsize=fontsize-1, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=ncol, columnspacing=1)
    if ncol==2:
        fig.legend(handles, labels, fontsize=fontsize-1, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=ncol, columnspacing=1)
    plt.tight_layout()
    if save != '':
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()



def plot_simplified(data, shocks, t_start=[4000, 6000], t_endx=120, xaxis=[4000, 6000],
                     fontsize=22, setzero=3800, cmap='viridis', save='', s=[1,2,3], ncols=8,
                     ylim_u=[0,0.15],yticks_pi=[-0.4, -0.2, 0.0, 0.2 ], ylim_pi=[-0.4,0.3], yticks_u=[0.0 ,0.1],
                     ylim_real=[-0.02,0.01], ylim_realexp=[-0.02,0.01], yticks_real=[-0.02, -0.015, -0.01, -0.005, 0.0, 0.005], yticks_realexp=[-0.02, -0.015, -0.01, -0.005, 0.0, 0.005]):
    
    #fig, axs = plt.subplots(1, 2, figsize=(12,4))
    
    fig = plt.figure(figsize=(15,8))
    gs = gridspec.GridSpec(4, 3, width_ratios=[2, 2, 1.1])#, width_ratios=[1, 2], height_ratios=[4, 1])
    axs = [fig.add_subplot(gs[:2, 0]), fig.add_subplot(gs[:2, 1])]
    axs2 = [fig.add_subplot(gs[2:, 0]), fig.add_subplot(gs[2:, 1])]
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, 2])
    ax6 = fig.add_subplot(gs[3, 2])
    #left, bottom, width, height = [0.62, 0.49, 0.12, 0.3]#[0.415, 0.81, 0.1, 0.125]
    #ax2 = fig.add_axes([left, bottom, width, height])

    t_start = xaxis[0]
    t_end = 4009
    shockk = True
    if True: #shock
        start = t_start- 10
        end = t_start + t_endx
    else:
        start = xaxis[0]
        end = xaxis[1]
    lw = 4.5

    def cmap(i):
        cm = ['mediumblue', 'darkorange', 'green']
        return(cm[i])

    linestyles = ['-', '--', '-.']
    aleft, aright = [], []
    xdata = data[shocks[0]].index[start:end]-setzero
    shock_x = [0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.]
    shock_y = []
    for i, shock in enumerate(shocks):

            xdata = data[shocks[0]].index[start:end]-setzero
            if 4800 in data[shock].index:
                print(shock)
                data[shock].drop(4800, inplace=True)
            xdata = xdata[:len(data[shock].iloc[start:end])]
            axs[0].plot(xdata, data[shock][start:end]['u'], linestyle=linestyles[i], lw=lw, c=cmap(i))
            #axs[0, 1].scatter(shock_x[i], phi_dict[shock]['u_max'], color=cmap(i), s=m)

            axs[1].plot(xdata, (data[shock][start:end]['Pavg']-1)*12, linestyle=linestyles[i], lw=lw, c=cmap(i), label=shock)
            #axs[1, 1].scatter(shock_x[i], phi_dict[shock]['inflation'], color=cmap(i), s=m)
            ax4.plot(xdata, data[shock][start:end]['tau_tar'], linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')
            #ax2.plot(xdata, (data[shock][start:end]['rhom']-data[shock][start:end]['pi-used'])*12, lw=lw, c=cmap(i))
            ax3.plot(xdata, data[shock][start:end]['rho']*12, linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')

            axs2[1].plot(xdata, (data[shock][start:end]['rhom']-data[shock][start:end]['pi-used'])*12, linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')
            axs2[0].plot(xdata, (data[shock][start:end]['rhom']-(data[shock][start:end]['Pavg']-1))*12, linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')
            ax6.plot(xdata, (data[shock][start:end]['rhop']-data[shock][start:end]['pi-used'])*12, linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')
            ax5.plot(xdata, (data[shock][start:end]['rhop']-(data[shock][start:end]['Pavg']-1))*12, linestyle=linestyles[i], lw=lw, c=cmap(i), label=f'{shock.replace("_", " ")} ')
    for i in range(2):
        if 1 in s:
            axs[i].axvspan(4000-200-setzero,4005-200-setzero, facecolor='0.3', alpha=0.5)
            axs2[i].axvspan(4000-200-setzero,4005-200-setzero, facecolor='0.3', alpha=0.5)
            #ax2.axvspan(4000-200-setzero,4005-200-setzero, facecolor='0.3', alpha=0.5)
            ax3.axvspan(4000-200-setzero,4005-200-setzero, facecolor='0.3', alpha=0.5)
            ax4.axvspan(4000-200-setzero,4005-200-setzero, facecolor='0.3', alpha=0.5)
            ax5.axvspan(4000-200-setzero,4005-200-setzero, facecolor='0.3', alpha=0.5)
            ax6.axvspan(4000-200-setzero,4005-200-setzero, facecolor='0.3', alpha=0.5)
        if 2 in s:
            axs[i].axvspan(4005-200-setzero, 4005-200-setzero+10, facecolor='0.6', alpha=0.5)
            axs2[i].axvspan(4005-200-setzero, 4005-200-setzero+10, facecolor='0.6', alpha=0.5)
            #ax2.axvspan(4005-200-setzero, 4005-200-setzero+10, facecolor='0.6', alpha=0.5)
            ax3.axvspan(4005-200-setzero, 4005-200-setzero+10, facecolor='0.6', alpha=0.5)
            ax4.axvspan(4005-200-setzero, 4005-200-setzero+10, facecolor='0.6', alpha=0.5)
            ax5.axvspan(4005-200-setzero, 4005-200-setzero+10, facecolor='0.6', alpha=0.5)
            ax6.axvspan(4005-200-setzero, 4005-200-setzero+10, facecolor='0.6', alpha=0.5)
        if 3 in s:
            axs[i].axvspan(4005-200-setzero+10, 4000-200-setzero+10+23, facecolor='0.8', alpha=0.5)
            axs2[i].axvspan(4005-200-setzero+10, 4000-200-setzero+10+23, facecolor='0.8', alpha=0.5)
            #ax2.axvspan(4005-200-setzero+10, 4000-200-setzero+10+23, facecolor='0.8', alpha=0.5)
            ax3.axvspan(4005-200-setzero+10, 4000-200-setzero+10+23, facecolor='0.8', alpha=0.5)
            ax4.axvspan(4005-200-setzero+10, 4000-200-setzero+10+23, facecolor='0.8', alpha=0.5)
            ax5.axvspan(4005-200-setzero+10, 4000-200-setzero+10+23, facecolor='0.8', alpha=0.5)
            ax6.axvspan(4005-200-setzero+10, 4000-200-setzero+10+23, facecolor='0.8', alpha=0.5)

        #axs[i].set_xlabel('Time in Months', fontsize=fontsize)
        #axs[i].set_xticks([0,20,40,60,80,100,120])
        #axs[i].set_xticklabels([0,20,40,60,80,100,120])
        axs[i].set_xticks([0, 24, 48, 72, 96])
        axs2[i].set_xticks([0, 24, 48, 72, 96])
        #ax1[i,j].set_xticklabels([0,20,40,60,80,100,120])
        axs[i].set_xticklabels(["Feb-20", "Feb-22","Feb-24","Feb-26","Feb-28",])
        axs2[i].set_xticklabels(["Feb-20", "Feb-22","Feb-24","Feb-26","Feb-28",])
        ax4.set_xticks([0, 48, 96])
        ax4.set_xticklabels(["Feb-20", "Feb-24", "Feb-28",])
        ax5.set_xticks([0, 48, 96])
        ax6.set_xticks([0, 48, 96])
        ax6.set_xticklabels(["Feb-20", "Feb-24", "Feb-28",])

        #axs[i].set_xlabel('Central Bank Strength, $\phi_\pi$', fontsize=fontsize)


            ##ax1[i,j].tick_params(axis='x', colors='white')
            #plt.setp(axs[i].get_xticklabels(), visible=False)
        axs[i].tick_params(axis='both', labelsize=fontsize-3)
        axs2[i].tick_params(axis='both', labelsize=fontsize-3)
        axs[i].yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        axs2[i].yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    axs[0].set_title('Unemployment, u', fontsize=fontsize)
    axs[1].set_title('Inflation Rate, $\pi$ (annual)', fontsize=fontsize)
    axs2[0].set_title('Real Loan Rate, $(\\rho^l - \pi)$', fontsize=fontsize)
    axs2[1].set_title('Expected Real Loan Rate, $(\\rho^l - \hat{\pi})$', fontsize=fontsize)
    #plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    plt.setp(ax5.get_xticklabels(), visible=False)
    ax5.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax6.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    #ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    # set ax3 yticks
    #ax3.set_yticks([0.012])
    
    #ax3.set_xticks([0, 0.2])
    #ax3.set_xticklabels([0, 0.2])
    #ax2.tick_params(axis='both', labelsize=fontsize-6)
    ax4.tick_params(axis='both', labelsize=fontsize-3)
    ax3.tick_params(axis='both', labelsize=fontsize-3)
    ax5.tick_params(axis='both', labelsize=fontsize-3)
    ax6.tick_params(axis='both', labelsize=fontsize-3)
    ax4.set_title('Expectation Anchor, $\\tau^T$', fontsize=fontsize-2)
    ax5.set_title('Real Deposit Rate, $(\\rho^d - \pi)$', fontsize=fontsize-2)
    ax6.set_title('Expected Real Deposit Rate, $(\\rho^d - \hat{\pi})$', fontsize=fontsize-2)
    #ax2.set_title('Real Interest, $(\\rho^l - \hat{\pi})$', fontsize=fontsize-5)
    ax3.set_title('Interest Rate,  $\\rho_0$ (annual)', fontsize=fontsize-2)
    #axs[0].set_title(r'Unemployment Max., $\max (u)$', fontsize=fontsize)
    #axs[1].set_title(r'Inflation Rate Average, $\langle \pi \rangle$ (annual)', fontsize=fontsize)

    #axs[0, 0].set_ylim([0,1])
    axs[0].set_ylim(ylim_u)
    axs[1].set_ylim(ylim_pi)
    axs2[0].set_ylim(ylim_real)
    axs2[1].set_ylim(ylim_realexp)
    ax4.set_ylim([-0.07,1.07])
    #ax2.set_ylim([-0.08,0.07])
    #axs[1].set_ylim([0,0.3])

    axs[1].set_yticks(yticks_pi)
    axs[0].set_yticks(yticks_u)
    axs2[0].set_yticks(yticks_real)
    axs2[1].set_yticks(yticks_realexp)
    axs[0].grid(alpha=0.7, zorder=0)
    axs[1].grid(alpha=0.7, zorder=0)
    axs2[0].grid(alpha=0.7, zorder=0)
    axs2[1].grid(alpha=0.7, zorder=0)
    #ax2.grid(alpha=0.7, zorder=0)
    ax4.grid(alpha=0.7, zorder=0)
    ax3.grid(alpha=0.7, zorder=0)
    ax5.grid(alpha=0.7, zorder=0)
    ax6.grid(alpha=0.7, zorder=0)
    #axs[1].set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    #axs[i, 1].set_xticklabels(shock_x)
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.526, -0.1), ncol=ncols, fontsize=fontsize-3, ncols=2)

    plt.tight_layout()
    plt.savefig(save, bbox_inches='tight')
