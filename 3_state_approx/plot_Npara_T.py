import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import setup
from matplotlib.ticker import MaxNLocator # added
import seaborn as sns
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})


def depth_2_Npara(depth):
    L=31
    Npara = depth * (L-1) * 16
    return Npara

def chi_2_Npara(chi):
    L=31
    Npara = 2 * L * chi**2
    return Npara

if __name__ == '__main__':

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
    plt.subplots_adjust(hspace=0.001)
    c_list = []
    mps_list = []
    ##### g=1.4, h=0.0 #####
    ##### 1 - F = 1e-4 #####
    c_list.append( (0.84, depth_2_Npara(2)) )
    c_list.append( (1.41, depth_2_Npara(3)) )
    c_list.append( (2.04, depth_2_Npara(4)) )
    c_list.append( (2.51, depth_2_Npara(5)) )

    mps_list.append( (0.86, chi_2_Npara(4)) )
    mps_list.append( (1.47, chi_2_Npara(8)) )
    mps_list.append( (2.09, chi_2_Npara(16)) )
    mps_list.append( (2.83, chi_2_Npara(32)) )

    ax1.plot(*zip(*c_list), 'x--', label='circuit')
    ax1.plot(*zip(*mps_list), 'x--', label='mps')

    # ax1.set_title(u'$g=1.4, h=0.0$')
    y_max = 8000
    ax1.set_ylim([0, y_max])
    ax1.text(0.7, y_max-2000, r'$g=1.4, h=0.$', fontsize=12)
    # ax1.legend(loc='lower right')
    # ax1.set_ylabel(u'num para')

    c_list = []
    mps_list = []
    ##### g=1.4, h=0.1 #####
    ##### 1 - F = 1e-4 #####
    c_list.append( (0.8295, depth_2_Npara(2)) )
    c_list.append( (1.25, depth_2_Npara(3)) )
    c_list.append( (1.61, depth_2_Npara(4)) )
    c_list.append( (1.95, depth_2_Npara(5)) )

    mps_list.append( (0.87, chi_2_Npara(4)) )
    mps_list.append( (1.47, chi_2_Npara(8)) )
    mps_list.append( (2.10, chi_2_Npara(16)) )
    mps_list.append( (2.84, chi_2_Npara(32)) )

    ax2.plot(*zip(*c_list), 'x--', label='circuit')
    ax2.plot(*zip(*mps_list), 'x--', label='mps')

    # plt.title(u'$g=1.4, h=0.1$')
    ax2.set_ylim([0, y_max])
    ax2.text(0.7, y_max-2000, r'$g=1.4, h=0.1$', fontsize=12)
    # ax2.legend(loc='lower right')
    ax2.set_ylabel(u'Number of parameters')
    nbins = len(ax2.get_yticklabels()) # added
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper')) # added


    c_list = []
    mps_list = []
    ##### g=1.4, h=0.9045 #####
    ##### 1 - F = 1e-4 #####
    c_list.append( (0.66, depth_2_Npara(2)) )
    c_list.append( (0.99, depth_2_Npara(3)) )
    c_list.append( (1.30, depth_2_Npara(4)) )
    c_list.append( (1.48, depth_2_Npara(5)) )

    mps_list.append( (0.88, chi_2_Npara(4)) )
    mps_list.append( (1.53, chi_2_Npara(8)) )
    mps_list.append( (2.31, chi_2_Npara(16)) )
    mps_list.append( (3.13, chi_2_Npara(32)) )

    ax3.plot(*zip(*c_list), 'x--', label='circuit')
    ax3.plot(*zip(*mps_list), 'x--', label='mps')

    # plt.title(u'$g=1.4, h=0.9045$')
    ax3.set_ylim([0, y_max])
    ax3.text(0.7, y_max - 2000, r'$g=1.4, h=0.9045$', fontsize=12)
    ax3.legend(loc='lower right')
    # ax3.set_ylabel(u'num para')
    ax3.set_xlabel(u'T')
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper')) # added



    plt.savefig('figure/cf_mps_circuit_Npara.png')
    plt.savefig('figure/cf_mps_circuit_Npara.pdf')
    plt.show()


