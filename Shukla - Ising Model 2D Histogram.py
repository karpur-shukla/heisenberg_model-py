# -*- coding: utf-8 -*-

''' Here, we create a static 2D N-by-M Ising grid of spins up and down, an update mechanism to
    update the spin at every site, and finally include the presence of an inter-spin coupling and an
    external magnetic field in the grids. This script then performs a histogram analysis of the
    lattices generated. This is part of an attempt to recreate the weighted histogram analysis
    method (WHAM) seen in A. Ferrenberg & R. Swendsen, Phys. Rev. Lett. 61, 23 (1988) and
    A. Ferrenberg & R. Swendsen, Phys. Rev. Lett. 63, 12 (1989). We're specifically looking at the
    two-state Ising model, i.e. with spins ±1/2.'''


# This section imports the libraries necessary to run the program.
import math
import matplotlib
import numpy
import random
import time


# This section stores the time at the start of the program.
program_start_time = time.clock()

'''Since we're interested in the amount of time the program will run in, we'll store the time at the
   beginning of the program using time.clock(), and compare it to the time at the end (again using
   time.clock(), applied to a different variable name. time.clock() just takes the time at a given
   moment; it's up to us to store it properly.'''


# This section sets the simulation parameters.
x_len = 8              # x_len is the number of sites in each row.
y_len = 8              # y_len is the number of rows in each column.
size = x_len * y_len   # size simply keeps the total number of sites handy.

MC_num = 1000000       # MC_num is the number of Monte Carlo updates.
sweeps = 100           # sweeps is the number of parameter sweeps.
hist_bin_size = 1      # hist_bin_size is the size of the bins of the histograms.
MC_therm_steps = 10000 # MC_therm_steps is the number of initial thermalisation steps.

h_start = 0.0          # h_start is the starting external field.
h_end = 0.0            # h_end is the ending external field.
h_hist = 0.0           # h_hist is the histogram external field.

T_start = 0.1          # T_start is the starting temperature.
T_end = 5.1            # T_end is the ending temperature.
T_hist = 2.5           # T_hist is the histogram temperature.

b_start = 1/T_start    # b_start is the value of beta corresponding to the starting temperature.
b_end = 1/T_end        # b_end is the value of beta corresponding to the ending temperature.
b_hist = 1/T_hist      # b_hist is the value of beta corresponding to the histogram temperature.

Jx_start = 1.0         # Jx_start is the starting x-direction coupling constant.
Jx_end = 1.0           # Jx_end is the ending x-direction coupling constant.
Jx_hist = 1.0          # Jx_hist is the histogram x-direction coupling constant.

Jy_start = 1.0         # Jy_start is the starting y-direction coupling constant.
Jy_end = 1.0           # Jy_end is the ending y-direction coupling constant.
Jy_hist = 1.0          # Jy_hist is the histogram x-direction coupling constant.


# This section creates the initial system, a static 2D array of spins (up or down).
initial_grid = numpy.random.choice([-1, 1], size = [y_len, x_len])

''' Note that this is faster than my original choice of how to initialise the system: 
    initial_grid = [[-1.0 if random.random() <= 0.5 else 1.0 for cube in xrange(x_len)] for row in xrange(y_len)]'''


# This function provides a printed version of the 2D Ising grid.
def print_grid(grating):
    Ising_grid_printed = []

    for chain in grating:
        IG_single_row = []

        for entry in chain:
            if entry == -1.0:
                IG_single_row.append("-")
            elif entry == +1.0:
                IG_single_row.append("+")
            else:
                raise ArithmeticError("Ising spin must be +1.0 or -1.0")

        IG_single_row_printed = " ".join(IG_single_row)
        Ising_grid_printed.append(IG_single_row_printed)

    for IG_row in Ising_grid_printed:
        print IG_row
    
    return Ising_grid_printed


# This function performs a single Monte Carlo update.
def MC_update(lat, h, Jx, Jy, T):
    x_size = len(lat[0])
    y_size = len(lat)
    beta = 1.0 / T

    for y in xrange(y_size):
        for x in xrange(x_size):
            dE = 0.0
            dE += h * lat[y][x]
            dE += Jx * lat[y][(x-1)%x_size] * lat[y][x]
            dE += Jx * lat[y][(x+1)%x_size] * lat[y][x]
            dE += Jy * lat[(y-1)%y_size][x] * lat[y][x]
            dE += Jy * lat[(y+1)%y_size][x] * lat[y][x]
            if random.random() < math.exp(-2*beta*dE):
                lat[y][x] = -lat[y][x]

    return lat

''' Following Swendsen's remark, I'll exploit the fact that exp(0) = 1 and that P = exp(-beta*E),
    which here is P = exp(-2*beta*h*spin). Since we have P as 1 for E < 0 and exp(-beta*E) for
    E > 0, it suffices to compare the result of random.random() with exp(-2*beta*h*spin). This is 
    the standard thing we do with the Metropolis-Hastings algorithm, but exploiting the fact that
    exp(0) = 1 simplifies matters, since it lets us collapse the min(1, exp(-a)) comparison into a
    single line.

    Note that here, I iterate over the array size, rather than the array elements. This is because
    I'm not sure how to call individual entries in the 2D array aside from listing their
    positions.'''


# This function retrieves the magnetisation and the energy.
def lat_props(trel, mu, ccx, ccy, temp):
    net_M = 0.0
    net_E = 0.0
    x_size = len(trel[0])
    y_size = len(trel)
    sites = float(x_size * y_size)

    for y_pt in xrange(y_size):
        for x_pt in xrange(x_size):
            net_M += trel[y_pt][x_pt]
            net_E += -mu * trel[y_pt][x_pt]
            net_E += -ccx * trel[y_pt][(x_pt+1)%x_size] * trel[y_pt][x_pt]
            net_E += -ccy * trel[(y_pt+1)%y_size][x_pt] * trel[y_pt][x_pt]

    lat_m = net_M/sites
    lat_e = net_E/sites

    return (net_M, lat_m, net_E, lat_e)


# This function performs the MC thermalisation.
def MC_thermal(collec, therm_steps, mag_field, couplx, couply, t):
    now_collec = collec

    for indiv_step in xrange(therm_steps):
        now_collec = MC_update(now_collec, mag_field, couplx, couply, t)        

    return now_collec


# This function performs several Monte Carlo updates, with the number of Monte Carlo updates specified by MC_iter.
def many_MC(array, MC_iter, ext_field, cc_x, cc_y, tepl):
    MC_M = [0] * MC_iter
    MC_E = [0] * MC_iter
    
    x_dist = len(array[0])
    y_dist = len(array)
    points = float(x_dist * y_dist)

    b = 1.0/tepl

    now_lat = array
    
    for update in xrange(MC_iter):
        now_update = MC_update(array, ext_field, cc_x, cc_y, tepl)
        now_props = lat_props(now_update, ext_field, cc_x, cc_y, tepl)
        now_lat = now_update
        
        MC_M[update] = now_props[0]
        MC_E[update] = now_props[2]
    
    avg_M = numpy.mean(MC_M, axis = None)
    avg_m = float(avg_M / points)
    avg_E = numpy.mean(MC_E, axis = None)
    avg_e = float(avg_E / points)
    sus = b * numpy.var(MC_M, axis = None) / points
    cv = math.pow(b, 2) * numpy.var(MC_E, axis = None) / points

    return (now_lat, avg_M, avg_m, avg_E, avg_e, sus, cv, MC_M, MC_E)


# This section creates and plots the histograms of the average total energy and average total magnetisation of each generated lattice.
def MC_hist(grid, MC_steps, mag_mom, coupl_x, coupl_y, tymherr, bin_size, hist_therm_steps):
    thermalised_grid = MC_thermal(grid, hist_therm_steps, mag_mom, coupl_x, coupl_y, tymherr)
    MC_results = many_MC(thermalised_grid, MC_steps, mag_mom, coupl_x, coupl_y, tymherr)

    MC_int_M_array = numpy.rint(MC_results[7])
    M_range = numpy.arange(min(MC_int_M_array), max(MC_int_M_array) + bin_size + 1, bin_size)
    M_hist = numpy.histogram(MC_int_M_array, bins = M_range)
    M_hist_x = numpy.delete(M_hist[1], len(M_hist[1]) - 1, axis = None)
    M_hist_y = M_hist[0]

    MC_int_E_array = numpy.rint(MC_results[8])
    E_range = numpy.arange(min(MC_int_E_array), max(MC_int_E_array) + bin_size + 1, bin_size)
    E_hist = numpy.histogram(MC_int_E_array, bins = E_range)
    E_hist_x = numpy.delete(E_hist[1], len(E_hist[1]) - 1, axis = None)
    E_hist_y = E_hist[0]
    
    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.suptitle("Magnetisation Histogram", family = "Gill Sans MT", fontsize = 16)
    matplotlib.pyplot.xlabel(r"Total Magnetisation ($\langle M \rangle$)", family = "Gill Sans MT")
    matplotlib.pyplot.bar(M_hist_x, M_hist_y)
    matplotlib.pyplot.show()

    matplotlib.pyplot.figure(2)
    matplotlib.pyplot.suptitle("Energy Histogram", family = "Gill Sans MT", fontsize = 16)
    matplotlib.pyplot.xlabel(r"Total Energy ($\langle E \rangle$)", family = "Gill Sans MT")
    matplotlib.pyplot.bar(E_hist_x, E_hist_y)
    matplotlib.pyplot.show()

    return (MC_results[7], MC_results[8], M_hist_x, M_hist_y, E_hist_x, E_hist_y)


# Here, we run the simulation. For testing, we also print the actual arrays; these commands are then commented out as necessary.
print "Initial 2D Ising Grid:"
print "                      "
print_grid(initial_grid)
print "                      "
print "                      "
updated_grid = many_MC(array = initial_grid, MC_iter = 10, ext_field = 0.0, cc_x = 1.0, cc_y = 1.0, tepl = 2.0)
print "Updated 2D Ising Grid:"
print "                      "
print_grid(updated_grid[0])
output_1NN = MC_hist(initial_grid, MC_num, h_hist, Jx_hist, Jy_hist, T_hist, hist_bin_size, MC_therm_steps)

'''print output_1NN[1]
print output_1NN[4]
print output_1NN[5]
print len(output_1NN[1])
print len(output_1NN[4])
print len(output_1NN[5])'''

# This section stores the time at the end of the program.
program_end_time = time.clock()
total_program_time = program_end_time - program_start_time
print "                      "
print "Program run time: %f seconds" % (total_program_time)
print "Program run time per site per MC sweep: %6g seconds" % (total_program_time / (MC_num * sweeps * size))

'''Note: To find out how long the program takes, we take the difference of time.clock() evaluated at
   the beginning of the program and at the end of the program. Here, we take the time at the end of
   the program, and define the total program time.'''
