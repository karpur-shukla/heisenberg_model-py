''' Here, we create a static 2D N-by-M Ising grid of spins up and down, an update mechanism to
    update the spin at every site, and finally include the presence of an external magnetic field in
    the grids.'''


# This section imports the libraries necessary to run the program.
import math
import matplotlib
import numpy
import random
import time
import tabulate


# This section stores the time at the start of the program.
program_start_time = time.clock()

'''Since we're interested in the amount of time the program will run in, we'll store the time at the
   beginning of the program using time.clock(), and compare it to the time at the end (again using
   time.clock(), applied to a different variable name. time.clock() just takes the time at a given
   moment; it's up to us to store it properly.'''


# This section sets the simulation parameters.
x_len = 8                    # x_len is the length of each row in the 2D grid.
y_len = 8                    # y_len is the number of rows in the 2D grid.
size = x_len * y_len         # size is the size of the array.

MC_sweeps = 1000000          # MC_sweeps is the number of Monte Carlo sweeps.
MC_therm_steps = 10000       # MC_therm_steps is the number of initial thermalisation steps.

h_start = -5.0               # h_start is the starting external field.
h_end = 5.0                  # h_end is the ending external field.

T_start = 1.0                # T_start is the starting temperature in multiples of Tc (the critical temperature).
T_end = 1.0                  # T_end is the ending temperature.

b_start = 1.0/T_start        # b_start is the value of beta corresponding to T_start.
b_end = 1.0/T_end            # b_end is the value of beta corresponding to T_end.

Jx_start = 0.0               # Jx_start is the starting x-direction nearest-neighbour coupling strength.
Jx_end = 0.0                 # Jx_end is the ending x-direction nearest-neighbour coupling strength.

Jy_start = 0.0               # Jy_start is the starting y-direction nearest-neighbour coupling strength.
Jy_end = 0.0                 # Jy_end is the ending y-direction nearest-neighbour coupling strength.

data_pts = 40                # data_pts is the number of sweep data points for the graphs/tables.


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
    x_len = len(lat[0])
    y_len = len(lat)
    beta = 1.0/T

    for y_pos in xrange(y_len):
        for x_pos in xrange(x_len):
            dE = 0.0
            dE += h * lat[y_pos][x_pos]
            dE += Jx * lat[y_pos][(x_pos-1)%x_len] * lat[y_pos][x_pos]
            dE += Jx * lat[y_pos][(x_pos+1)%x_len] * lat[y_pos][x_pos]
            dE += Jy * lat[(y_pos-1)%y_len][x_pos] * lat[y_pos][x_pos]
            dE += Jy * lat[(y_pos+1)%y_len][x_pos] * lat[y_pos][x_pos]
            if random.random() <= math.exp(-2*beta*dE):
                lat[y_pos][x_pos] = -lat[y_pos][x_pos]
 
    return lat

''' Following Swendsen's remark, I'll exploit the fact that exp(0) = 1 and that P = exp(-beta*E),
    which here is P = exp(-2*beta*h*spin). Since we have P as 1 for E < 0 and exp(-beta*E) for
    E > 0, it suffices to compare the result of random.random() with exp(-2*beta*h*spin). This is 
    the standard thing we do with the Metropolis-Hastings algorithm, but exploiting the fact that
    exp(0) = 1 simplifies matters, since it lets us collapse the min(1, exp(-a)) comparison into a
    single line.

    Note that here, I iterate over the array size, rather than the array elements. This is because
    I'm not sure how to call individual entries in the 2D array aside from listing their positions. 
    Specifically, I use "for row in xrange(len(lat))" and "for item in xrange(len(lat[row]))" rather
    than "for row in lat" and "for item in row" respectively (i.e. I iterate over the array size,
    rather than the array elements). This is because I'm not sure how to call individual entries in
    the 2D array aside from listing their positions. (My attempt at iterating over the elements for
    the 2D update mechanism seems to not have worked; the update mechanism didn't actually do
    anything when I tried this. The failed update mechanism is shown below.)

def update_grid(lat):
    for row in lat:
        for item in row:
            if random.random() <= math.exp(-beta*h*item):
                item = -item
    return lat '''


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
def many_MC_updates(array, MC_iter, ext_field, cc_x, cc_y, tepl):
    MC_M = 0.0
    MC_M2 = 0.0
    MC_m = 0.0
    MC_m2 = 0.0
    MC_E = 0.0
    MC_E2 = 0.0
    MC_e = 0.0
    MC_e2 = 0.0

    now_grid = array
    array_x_size = len(array[0])
    array_y_size = len(array)
    array_sites = float(array_x_size * array_y_size)
    b = 1.0/tepl

    for indiv_step in xrange(MC_iter):
        now_grid = MC_update(now_grid, ext_field, cc_x, cc_y, tepl)        
        now_props = lat_props(now_grid, ext_field, cc_x, cc_y, tepl)
        MC_M += now_props[0]
        MC_M2 += math.pow(now_props[0], 2)
        MC_m += now_props[1]
        MC_m2 += math.pow(now_props[1], 2)
        MC_E += now_props[2]
        MC_E2 += math.pow(now_props[2], 2)
        MC_e += now_props[3]
        MC_e2 += math.pow(now_props[3], 2)

    avg_M = float(MC_M/MC_iter)
    avg_M2 = float(MC_M2/MC_iter)
    avg_m = float(MC_m/MC_iter)
    avg_m2 = float(MC_m2/MC_iter)
    avg_E = float(MC_E/MC_iter)
    avg_E2 = float(MC_E2/MC_iter)
    avg_e = float(MC_e/MC_iter)
    avg_e2 = float(MC_e2/MC_iter)
    susc = b * (avg_m2 - math.pow(avg_m, 2)) * array_sites
    cv = (math.pow(b, 2) * (avg_e2 - math.pow(avg_e, 2))) * array_sites

    return (now_grid, avg_M, avg_M2, avg_m, avg_m2, avg_E, avg_E2, avg_e, avg_e2, susc, cv)

''' To write cv in terms of the total energy rather than the per-site energy, we have:
    cv = (math.pow(b, 2) * (avg_E2 - math.pow(avg_E, 2))) / array_sites. '''


#This function lets us sweep our MC simulation over different values of the magnetic moment, h.
def h_sweep(grid, h_min, h_max, sweep_Jx, sweep_Jy, sweep_T, MC_steps, points, h_therm_steps):
    h_sweep_m_vals = []
    h_sweep_e_vals = []
    h_sweep_susc_vals = []
    h_sweep_cv_vals = []

    h_step = float((h_max-h_min)/points)
    
    if h_step <= 0.0:
        raise ArithmeticError("sweep range must be a positive nonzero float")
    
    else:
        h_now = h_min
        sweep_b = 1.0/sweep_T

        while h_now <= h_max:
            now_lat = MC_thermal(grid, h_therm_steps, h_min, sweep_Jx, sweep_Jy, sweep_T)
            MC_results_now = many_MC_updates(now_lat, MC_steps, h_now, sweep_Jx, sweep_Jy, sweep_T)
            now_lat = MC_results_now[0]
        
            ideal_m_now = math.tanh(sweep_b*h_now)
            m_diff_now = MC_results_now[3] - ideal_m_now
            h_now_m_vals = [sweep_T, h_now, MC_results_now[3], ideal_m_now, m_diff_now]
            h_sweep_m_vals.append(h_now_m_vals)

            ideal_e_now = -h_now * math.tanh(sweep_b*h_now)
            e_diff_now = MC_results_now[7] - ideal_e_now
            h_now_e_vals = [sweep_T, h_now, MC_results_now[7], ideal_e_now, e_diff_now]
            h_sweep_e_vals.append(h_now_e_vals)

            ideal_susc_now = sweep_b * math.pow(1/math.cosh(sweep_b*h_now), 2)
            susc_diff_now = MC_results_now[9] - ideal_susc_now
            h_now_susc_vals = [sweep_T, h_now, MC_results_now[9], ideal_susc_now, susc_diff_now]
            h_sweep_susc_vals.append(h_now_susc_vals)

            ideal_cv_now = math.pow(sweep_b, 2) * math.pow(h_now, 2) * math.pow(1/math.cosh(sweep_b*h_now), 2)
            cv_diff_now = MC_results_now[10] - ideal_cv_now
            h_now_cv_vals = [sweep_T, h_now, MC_results_now[10], ideal_cv_now, cv_diff_now]
            h_sweep_cv_vals.append(h_now_cv_vals)

            h_now += h_step

    return (h_sweep_m_vals, h_sweep_e_vals, h_sweep_susc_vals, h_sweep_cv_vals)


#This function lets us sweep our MC simulation over different values of the temperature.
def T_sweep(lat_in, T_min, T_max, spect_h, spect_Jx, spect_Jy, MC_num, points, T_therm_steps):
    T_sweep_mag_vals = []
    T_sweep_ener_vals = []
    T_sweep_chi_vals = []
    T_sweep_spec_heat_vals = []

    T_step = float((T_max-T_min)/points)
    
    if T_step <= 0.0:
        raise ArithmeticError("sweep range must be a positive nonzero float")
    
    else:
        T_curr = T_min
        b_curr = 1.0/T_curr

        while T_curr <= T_max:
            curr_lat = MC_thermal(lat_in, T_therm_steps, spect_h, spect_Jx, spect_Jy, T_min)
            MC_results_now = many_MC_updates(curr_lat, MC_num, spect_h, spect_Jx, spect_Jy, T_curr)
            curr_lat = MC_results_now[0]
        
            ideal_mag_now = math.tanh(b_curr*spect_h)
            mag_diff_now = MC_results_now[3] - ideal_mag_now
            T_now_mag_vals = [T_curr, spect_h, MC_results_now[3], ideal_mag_now, mag_diff_now]
            T_sweep_mag_vals.append(T_now_mag_vals)

            ideal_ener_now = -spect_h * math.tanh(b_curr*spect_h)
            ener_diff_now = MC_results_now[7] - ideal_ener_now
            T_now_ener_vals = [T_curr, spect_h, MC_results_now[7], ideal_ener_now, ener_diff_now]
            T_sweep_ener_vals.append(T_now_ener_vals)

            ideal_chi_now = b_curr * math.pow(1/math.cosh(b_curr*spect_h), 2)
            chi_diff_now = MC_results_now[9] - ideal_chi_now
            T_now_chi_vals = [T_curr, spect_h, MC_results_now[9], ideal_chi_now, chi_diff_now]
            T_sweep_chi_vals.append(T_now_chi_vals)

            ideal_spec_heat_now = math.pow(b_curr, 2) * math.pow(spect_h, 2) * math.pow(1/math.cosh(b_curr*spect_h), 2)
            spec_heat_diff_now = MC_results_now[10] - ideal_spec_heat_now
            T_now_spec_heat_vals = [T_curr, spect_h, MC_results_now[10], ideal_spec_heat_now, spec_heat_diff_now]
            T_sweep_spec_heat_vals.append(T_now_spec_heat_vals)
            T_curr += T_step

    return (T_sweep_mag_vals, T_sweep_ener_vals, T_sweep_chi_vals, T_sweep_spec_heat_vals)


# Here, we run the simulation. For testing, we also print the actual arrays; these commands are then commented out as necessary.
print "Initial 2D Ising Grid:"
print "                      "
print_grid(initial_grid)
print "                      "
print "                      "

h_sweep_grids = h_sweep(initial_grid, h_start, h_end, Jx_start, Jy_start, T_start, MC_sweeps, data_pts, MC_therm_steps)

swept_h_m_grid = numpy.array(h_sweep_grids[0])
swept_h_e_grid = numpy.array(h_sweep_grids[1])
swept_h_susc_grid = numpy.array(h_sweep_grids[2])
swept_h_cv_grid = numpy.array(h_sweep_grids[3])

swept_h_range = numpy.linspace(-5.0, 5.0, 10000, endpoint = True)

swept_h_T = swept_h_m_grid[:,0]
swept_h_ext_field = swept_h_m_grid[:,1]

swept_h_m = swept_h_m_grid[:,2]
swept_h_ideal_m = numpy.tanh(b_start * swept_h_range)

swept_h_e = swept_h_e_grid[:,2]
swept_h_ideal_e = -swept_h_range * numpy.tanh(b_start * swept_h_range)

swept_h_susc = swept_h_susc_grid[:,2]
swept_h_ideal_susc = b_start * numpy.power(1/numpy.cosh(b_start*swept_h_range), 2.0)

swept_h_cv = swept_h_cv_grid[:,2]
swept_h_ideal_cv = math.pow(b_start, 2.0) * numpy.power(swept_h_range, 2.0) * numpy.power(1/numpy.cosh(b_start*swept_h_range), 2.0)

print "                      "
print tabulate.tabulate(swept_h_m_grid, headers = ["Temp.", "Ext. Field", "Sim. <m>", "Calc. <m>", "<m> err."], floatfmt=".7f")

print "                      "
print tabulate.tabulate(swept_h_e_grid, headers = ["Temp.", "Ext. Field", "Sim. <e>", "Calc. <e>", "<e> err."], floatfmt=".7f")

print "                      "
print tabulate.tabulate(swept_h_susc_grid, headers = ["Temp.", "Ext. Field", u"Sim. \u03c7", u"Calc. \u03c7", u"\u03c7 err."], floatfmt=".7f")

print "                      "
print tabulate.tabulate(swept_h_cv_grid, headers = ["Temp.", "Ext. Field", "Sim. c_v", "Calc. c_v", "c_v err."], floatfmt=".7f")

matplotlib.pyplot.figure(1)
matplotlib.pyplot.suptitle("Per Site Magnetisation", family = "Gill Sans MT", fontsize = 16)
matplotlib.pyplot.xlabel(r"External Magnetic Field ($h$)", family = "Gill Sans MT")
matplotlib.pyplot.ylabel(r"Per Site Magnetisation ($\langle m \rangle$)", family = "Gill Sans MT")
matplotlib.pyplot.plot(swept_h_range, swept_h_ideal_m, linestyle = "-", color = "#2824A7")
matplotlib.pyplot.scatter(swept_h_ext_field, swept_h_m, color = "#FFAA00")

matplotlib.pyplot.figure(2)
matplotlib.pyplot.suptitle("Average Per-Site Energy", family = "Gill Sans MT", fontsize = 16)
matplotlib.pyplot.xlabel(r"External Magnetic Field ($h$)", family = "Gill Sans MT")
matplotlib.pyplot.ylabel(r"Average Per-Site Energy ($\langle u \rangle$)", family = "Gill Sans MT")
matplotlib.pyplot.plot(swept_h_range, swept_h_ideal_e, linestyle = "-", color = "#2824A7")
matplotlib.pyplot.scatter(swept_h_ext_field, swept_h_e, color = "#FFAA00")
matplotlib.pyplot.show()

matplotlib.pyplot.figure(3)
matplotlib.pyplot.suptitle("Susceptibility", family = "Gill Sans MT", fontsize = 16)
matplotlib.pyplot.xlabel(r"External Magnetic Field ($h$)", family = "Gill Sans MT")
matplotlib.pyplot.ylabel(r"Susceptibility ($\chi$)", family = "Gill Sans MT")
matplotlib.pyplot.plot(swept_h_range, swept_h_ideal_susc, linestyle = "-", color = "#2824A7")
matplotlib.pyplot.scatter(swept_h_ext_field, swept_h_susc, color = "#FFAA00")
matplotlib.pyplot.show()

matplotlib.pyplot.figure(4)
matplotlib.pyplot.suptitle("Specific Heat at Constant Volume", family = "Gill Sans MT", fontsize = 16)
matplotlib.pyplot.xlabel(r"External Magnetic Field ($h$)", family = "Gill Sans MT")
matplotlib.pyplot.ylabel(r"Specific Heat at Constant Volume ($c_v$)", family = "Gill Sans MT")
matplotlib.pyplot.plot(swept_h_range, swept_h_ideal_cv, linestyle = "-", color = "#2824A7")
matplotlib.pyplot.scatter(swept_h_ext_field, swept_h_cv, color = "#FFAA00")
matplotlib.pyplot.show()


# This section stores the time at the end of the program.
program_end_time = time.clock()
total_program_time = program_end_time - program_start_time
print "                      "
print "Program run time: %f seconds" % (total_program_time)
print "Program run time per site per MC sweep: %f seconds" % (total_program_time/(MC_sweeps*data_pts))

'''Note: To find out how long the program takes, we take the difference of time.clock() evaluated at
   the beginning of the program and at the end of the program. Here, we take the time at the end of
   the program, and define the total program time.'''