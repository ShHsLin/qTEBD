import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
sys.path.append('../')
import setup
from matplotlib.ticker import MaxNLocator # added
import seaborn as sns
# sns.set()
# sns.set_style("whitegrid", {'axes.grid' : False})
current_palette = sns.color_palette()

c_color = [sns.color_palette("Blues")[5], sns.color_palette("Blues")[2]]
mps_color = [sns.color_palette("coolwarm", 7)[6], sns.color_palette("coolwarm", 7)[5]]

color_dot = sns.color_palette("coolwarm", 7)[4]
color_dot2 = sns.color_palette("Blues")[2]
color_fit = sns.color_palette("coolwarm", 7)[6]
color_fit2 = sns.color_palette("Blues")[5]
markersize = 4.


def linear_f(x, a, b):
    return a * x + b

def exp_f(x, a, b, c):
    return a * np.exp(b * x) + c

def depth_2_Npara(depth):
    L=31
    Npara = depth * (L-1) * 16
    return Npara

def chi_2_Npara(chi, L=31):
    Npara_old = 2 * L * chi**2

    bond_list = []
    for idx in range(L+1):
        bond_list.append(np.amin([2**(L-idx),2**idx, chi]))

    assert len(bond_list) == (L+1)
    bond_list = np.array(bond_list)
    Npara = np.sum(bond_list[:-1]*bond_list[1:]*2)
    # print("chi=", chi, "new estimate: ", Npara, " old estimate : ", Npara_old, "diff : ", (Npara-Npara_old)/Npara)

    return Npara

if __name__ == '__main__':

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, figsize=(3.5,4.6))
    plt.subplots_adjust(hspace=0.07)


    mps_list = []
    ##### g=1.4, h=0.0 #####
    ##### 1 - F = 1e-4 #####
    mps_list.append( (0.3125, chi_2_Npara(2)) )
    mps_list.append( (0.396, chi_2_Npara(3)) )
    mps_list.append( (0.86, chi_2_Npara(4)) )
    mps_list.append( (0.95, chi_2_Npara(5)) )
    mps_list.append( (1.03, chi_2_Npara(6)) )
    mps_list.append( (1.15, chi_2_Npara(7)) )
    mps_list.append( (1.47, chi_2_Npara(8)) )
    mps_list.append( (1.56, chi_2_Npara(9)) )

    mps_list.append( (1.60, chi_2_Npara(10)) )
    mps_list.append( (1.68, chi_2_Npara(11)) )
    mps_list.append( (1.74, chi_2_Npara(12)) )
    mps_list.append( (1.80, chi_2_Npara(13)) )
    mps_list.append( (1.88, chi_2_Npara(14)) )
    mps_list.append( (2.03, chi_2_Npara(15)) )

    mps_list.append( (2.09, chi_2_Npara(16)) )
    mps_list.append( (2.18, chi_2_Npara(17)) )
    mps_list.append( (2.23, chi_2_Npara(18)) )
    mps_list.append( (2.28, chi_2_Npara(19)) )
    mps_list.append( (2.32, chi_2_Npara(20)) )
    mps_list.append( (2.36, chi_2_Npara(21)) )
    mps_list.append( (2.41, chi_2_Npara(22)) )

    mps_list.append( (2.44, chi_2_Npara(23)) )
    mps_list.append( (2.47, chi_2_Npara(24)) )
    mps_list.append( (2.52, chi_2_Npara(25)) )
    mps_list.append( (2.59, chi_2_Npara(26)) )
    mps_list.append( (2.62, chi_2_Npara(27)) )
    mps_list.append( (2.68, chi_2_Npara(28)) )
    mps_list.append( (2.73, chi_2_Npara(29)) )
    mps_list.append( (2.77, chi_2_Npara(30)) )
    mps_list.append( (2.80, chi_2_Npara(31)) )

    mps_list.append( (2.83, chi_2_Npara(32)) )

    ax1.plot(*zip(*mps_list), 'o', markersize=markersize, color=color_dot, label='mps')

    x_data, y_data = zip(*mps_list)
    x_data = np.array(x_data)[[0,2,6,14,30]]
    y_data = np.array(y_data)[[0,2,6,14,30]]

    popt, pcov = curve_fit(exp_f, x_data, y_data)
    x_data = np.arange(x_data[0], x_data[-1]+0.01, (x_data[-1]-x_data[0])/100)
    ax1.plot(x_data, exp_f(x_data, *popt), '--', color=color_fit,
             label='fit' # : a=%g, b=%g, c=%g' % tuple(popt)
            )

    print('a=%g, b=%g, c=%g' % tuple(popt))


    # ax1.set_title(u'$g=1.4, h=0.0$')
    y_max = 50000
    dy = y_max / 5
    ax1.set_ylim([0, y_max])
    ax1.text(0.25, y_max-dy, r'$h=0$', fontsize=10.)
    # ax1.legend(loc='lower right')
    # ax1.set_ylabel(u'num para')

    # nbins=4
    # ax1.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper')) # added

    mps_list = []
    ##### g=1.4, h=0.1 #####
    ##### 1 - F = 1e-4 #####
    mps_list.append( (0.3125, chi_2_Npara(2)) )
    mps_list.append( (0.396, chi_2_Npara(3)) )
    mps_list.append( (0.87, chi_2_Npara(4)) )
    mps_list.append( (0.95, chi_2_Npara(5)) )
    mps_list.append( (1.03, chi_2_Npara(6)) )
    mps_list.append( (1.14, chi_2_Npara(7)) )
    mps_list.append( (1.47, chi_2_Npara(8)) )
    mps_list.append( (1.56, chi_2_Npara(9)) )

    mps_list.append( (1.60, chi_2_Npara(10)) )
    mps_list.append( (1.68, chi_2_Npara(11)) )
    mps_list.append( (1.74, chi_2_Npara(12)) )
    mps_list.append( (1.80, chi_2_Npara(13)) )
    mps_list.append( (1.88, chi_2_Npara(14)) )
    mps_list.append( (2.03, chi_2_Npara(15)) )

    mps_list.append( (2.10, chi_2_Npara(16)) )
    mps_list.append( (2.18, chi_2_Npara(17)) )
    mps_list.append( (2.23, chi_2_Npara(18)) )
    mps_list.append( (2.28, chi_2_Npara(19)) )
    mps_list.append( (2.32, chi_2_Npara(20)) )
    mps_list.append( (2.36, chi_2_Npara(21)) )
    mps_list.append( (2.41, chi_2_Npara(22)) )

    mps_list.append( (2.44, chi_2_Npara(23)) )
    mps_list.append( (2.47, chi_2_Npara(24)) )
    mps_list.append( (2.52, chi_2_Npara(25)) )
    mps_list.append( (2.59, chi_2_Npara(26)) )
    mps_list.append( (2.63, chi_2_Npara(27)) )
    mps_list.append( (2.68, chi_2_Npara(28)) )
    mps_list.append( (2.73, chi_2_Npara(29)) )
    mps_list.append( (2.78, chi_2_Npara(30)) )
    mps_list.append( (2.80, chi_2_Npara(31)) )


    mps_list.append( (2.84, chi_2_Npara(32)) )

    ax2.plot(*zip(*mps_list), 'o', markersize=markersize, color=color_dot, label='mps')


    x_data, y_data = zip(*mps_list)
    x_data = np.array(x_data)[[0,2,6,14,30]]
    y_data = np.array(y_data)[[0,2,6,14,30]]

    popt, pcov = curve_fit(exp_f, x_data, y_data)
    x_data = np.arange(x_data[0], x_data[-1]+0.01, (x_data[-1]-x_data[0])/100)
    ax2.plot(x_data, exp_f(x_data, *popt), '--', color=color_fit,
             label='fit'  #: a=%g, b=%g, c=%g' % tuple(popt)
            )

    print('a=%g, b=%g, c=%g' % tuple(popt))


    # plt.title(u'$g=1.4, h=0.1$')
    ax2.set_ylim([0, y_max])
    ax2.text(0.25, y_max-dy, r'$h=0.1$', fontsize=10.)
    # ax2.legend(loc='lower right')
    ax2.set_ylabel(u'Number of parameters')

    # nbins = len(ax2.get_yticklabels()) # added
    # nbins=4
    # ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper')) # added


    mps_list = []
    ##### g=1.4, h=0.9045 #####
    ##### 1 - F = 1e-4 #####
    mps_list.append( (0.3125, chi_2_Npara(2)) )
    mps_list.append( (0.396, chi_2_Npara(3)) )
    mps_list.append( (0.88, chi_2_Npara(4)) )
    mps_list.append( (0.97, chi_2_Npara(5)) )
    mps_list.append( (1.04, chi_2_Npara(6)) )
    mps_list.append( (1.15, chi_2_Npara(7)) )

    mps_list.append( (1.53, chi_2_Npara(8)) )
    mps_list.append( (1.62, chi_2_Npara(9)) )
    mps_list.append( (1.68, chi_2_Npara(10)) )
    mps_list.append( (1.74, chi_2_Npara(11)) )
    mps_list.append( (1.81, chi_2_Npara(12)) )
    mps_list.append( (1.86, chi_2_Npara(13)) )
    mps_list.append( (1.95, chi_2_Npara(14)) )
    mps_list.append( (2.13, chi_2_Npara(15)) )

    mps_list.append( (2.31, chi_2_Npara(16)) )
    mps_list.append( (2.34, chi_2_Npara(17)) )
    mps_list.append( (2.38, chi_2_Npara(18)) )
    mps_list.append( (2.43, chi_2_Npara(19)) )
    mps_list.append( (2.47, chi_2_Npara(20)) )
    mps_list.append( (2.51, chi_2_Npara(21)) )
    mps_list.append( (2.56, chi_2_Npara(22)) )

    mps_list.append( (2.61, chi_2_Npara(23)) )
    mps_list.append( (2.69, chi_2_Npara(24)) )
    mps_list.append( (2.77, chi_2_Npara(25)) )
    mps_list.append( (2.82, chi_2_Npara(26)) )
    mps_list.append( (2.87, chi_2_Npara(27)) )
    mps_list.append( (2.93, chi_2_Npara(28)) )
    mps_list.append( (3.00, chi_2_Npara(29)) )
    mps_list.append( (3.04, chi_2_Npara(30)) )
    mps_list.append( (3.09, chi_2_Npara(31)) )


    mps_list.append( (3.13, chi_2_Npara(32)) )

    ax3.plot(*zip(*mps_list), 'o', markersize=markersize, color=color_dot, label='mps')


    x_data, y_data = zip(*mps_list)
    x_data = np.array(x_data)[[0,2,6,14,30]]
    y_data = np.array(y_data)[[0,2,6,14,30]]

    popt, pcov = curve_fit(exp_f, x_data, y_data)
    x_data = np.arange(x_data[0], x_data[-1]+0.01, (x_data[-1]-x_data[0])/100)
    ax3.plot(x_data, exp_f(x_data, *popt), '--', color=color_fit,
             label='fit'  # : a=%g, b=%g, c=%g' % tuple(popt)
            )

    print('a=%g, b=%g, c=%g' % tuple(popt))


    # plt.title(u'$g=1.4, h=0.9045$')
    ax3.set_ylim([0, y_max])
    ax3.text(0.25, y_max-dy, r'$h=0.9045$', fontsize=10.)
    # ax3.legend(loc='lower right')
    # ax1.legend(loc='center right')
    # ax2.legend(loc='center right')
    ax3.legend(loc='center left', fontsize=10.)
    # ax3.set_ylabel(u'num para')
    ax3.set_xlabel(u'$Jt^*$')

    # nbins=4
    # ax3.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper')) # added


    ax3.set_xlim([0., 3.2])

    plt.subplots_adjust(left=0.20, top=0.95, right=0.95)
    plt.savefig('figure/mps_Npara_fit.pdf')
    plt.show()

    plt.close(fig)


    ######################################################################
    ######################################################################
    ######################################################################
    ######################################################################


    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, figsize=(3.5,4.6))
    plt.subplots_adjust(hspace=0.07)



    c_list = []
    ##### g=1.4, h=0.0 #####
    ##### 1 - F = 1e-4 #####
    c_list.append( (0.3125, chi_2_Npara(2)) )
    c_list.append( (0.84, depth_2_Npara(2)) )
    c_list.append( (1.41, depth_2_Npara(3)) )
    c_list.append( (2.04, depth_2_Npara(4)) )
    c_list.append( (2.51, depth_2_Npara(5)) )
    c_list.append( (2.79, depth_2_Npara(6)) )

    ax1.plot(*zip(*c_list), 'o', markersize=markersize, color=color_dot2, label='circuit')

    x_data, y_data = zip(*c_list)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    popt, pcov = curve_fit(linear_f, x_data, y_data)
    x_data = np.arange(x_data[0], x_data[-1]+0.01, (x_data[-1]-x_data[0])/100)
    ax1.plot(x_data, linear_f(x_data, *popt), '--', color=color_fit2,
             label='fit' # : a=%g, b=%g' % tuple(popt)
            )

    print('a=%g, b=%g' % tuple(popt))


    # ax1.set_title(u'$g=1.4, h=0.0$')
    y_max = 3200
    dy = y_max / 5
    ax1.set_ylim([0, y_max])
    ax1.text(0.25, y_max-dy, r'$h=0$', fontsize=10.)
    # ax1.legend(loc='lower right')
    # ax1.set_ylabel(u'num para')

    # nbins=4
    # ax1.yaxis.set_major_locator(MaxNLocator(nbins=nbins)) # added


    c_list = []
    ##### g=1.4, h=0.1 #####
    ##### 1 - F = 1e-4 #####
    c_list.append( (0.3125, chi_2_Npara(2)) )
    c_list.append( (0.8295, depth_2_Npara(2)) )
    c_list.append( (1.25, depth_2_Npara(3)) )
    c_list.append( (1.61, depth_2_Npara(4)) )
    c_list.append( (1.95, depth_2_Npara(5)) )
    c_list.append( (2.25, depth_2_Npara(6)) )

    ax2.plot(*zip(*c_list), 'o', markersize=markersize, color=color_dot2, label='circuit')


    x_data, y_data = zip(*c_list)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    popt, pcov = curve_fit(linear_f, x_data, y_data)
    x_data = np.arange(x_data[0], x_data[-1]+0.01, (x_data[-1]-x_data[0])/100)
    ax2.plot(x_data, linear_f(x_data, *popt), '--', color=color_fit2,
             label='fit'  #: a=%g, b=%g' % tuple(popt)
            )

    print('a=%g, b=%g' % tuple(popt))


    # plt.title(u'$g=1.4, h=0.1$')
    ax2.set_ylim([0, y_max])
    ax2.text(0.25, y_max-dy, r'$h=0.1$', fontsize=10.)
    # ax2.legend(loc='lower right')
    ax2.set_ylabel(u'Number of parameters')


    # nbins = len(ax2.get_yticklabels()) # added
    # ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper')) # added



    c_list = []
    ##### g=1.4, h=0.9045 #####
    ##### 1 - F = 1e-4 #####
    c_list.append( (0.3125, chi_2_Npara(2)) )
    c_list.append( (0.66, depth_2_Npara(2)) )
    c_list.append( (0.99, depth_2_Npara(3)) )
    c_list.append( (1.30, depth_2_Npara(4)) )
    c_list.append( (1.48, depth_2_Npara(5)) )
    c_list.append( (1.75, depth_2_Npara(6)) )

    ax3.plot(*zip(*c_list), 'o', markersize=markersize, color=color_dot2, label='circuit')


    x_data, y_data = zip(*c_list)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    popt, pcov = curve_fit(linear_f, x_data, y_data)
    x_data = np.arange(x_data[0], x_data[-1]+0.01, (x_data[-1]-x_data[0])/100)
    ax3.plot(x_data, linear_f(x_data, *popt), '--', color=color_fit2,
             label='fit'  # : a=%g, b=%g' % tuple(popt)
            )

    print('a=%g, b=%g' % tuple(popt))


    # plt.title(u'$g=1.4, h=0.9045$')
    ax3.set_ylim([0, y_max])
    ax3.text(0.25, y_max-dy, r'$h=0.9045$', fontsize=10.)
    # ax3.legend(loc='lower right')
    # ax1.legend(loc='center right')
    # ax2.legend(loc='center right')
    ax3.legend(loc='center right', fontsize=10.)
    # ax3.set_ylabel(u'num para')
    ax3.set_xlabel(u'$Jt^*$')

    # ax3.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper')) # added


    ax3.set_xlim([0., 3.2])

    plt.subplots_adjust(left=0.20, top=0.95, right=0.95)
    plt.savefig('figure/circuit_Npara_fit.pdf')
    plt.show()


