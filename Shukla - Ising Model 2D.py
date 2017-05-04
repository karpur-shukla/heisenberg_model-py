''' Here, we create a static 2D N-by-M Ising grid of spins up and down, an update mechanism to
    update the spin at every site, and finally include the presence of an inter-spin coupling and an
    external magnetic field in the grids.'''


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
x_len = 8              # x_len is the number of sites in each row.
y_len = 8              # y_len is the number of rows in each column.
size = x_len * y_len   # size simply keeps the total number of sites handy.

MC_num = 10000       # MC_num is the number of Monte Carlo updates.
sweeps = 50            # sweeps is the number of parameter sweeps.
MC_therm_steps = 100 # MC_therm_steps is the number of initial thermalisation steps.

h_start = 0.0          # h_start is the starting external field.
h_end = 0.0            # h_end is the ending external field.
h_hist = 0.0           # h_hist is the histogram external field.

T_start = 0.1          # T_start is the starting temperature.
T_end = 5.1            # T_end is the ending temperature.
T_hist = 2.0           # T_hist is the histogram temperature.

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
def MC_update(grid, h, Jx, Jy, T):
    x_size = len(grid[0])
    y_size = len(grid)
    b = 1.0 / T
    grid_M = 0.0
    grid_E = 0.0

    for y in xrange(y_size):
        for x in xrange(x_size):
            dE = 0.0
            dE += h * grid[y][x]
            dE += Jx * grid[y][(x-1)%x_size] * grid[y][x]
            dE += Jx * grid[y][(x+1)%x_size] * grid[y][x]
            dE += Jy * grid[(y-1)%y_size][x] * grid[y][x]
            dE += Jy * grid[(y+1)%y_size][x] * grid[y][x]
            if random.random() < math.exp(-2*b*dE):
                grid[y][x] = -grid[y][x]
            grid_M += grid[y][x]
            grid_E += Jx * grid[y][(x-1)%x_size] * grid[y][x]
            grid_E += Jy * grid[(y-1)%y_size][x] * grid[y][x]

    return (grid, grid_M, grid_E)

''' Following Swendsen's remark, I'll exploit the fact that exp(0) = 1 and that P = exp(-beta*E),
    which here is P = exp(-2*beta*h*spin). Since we have P as 1 for E < 0 and exp(-beta*E) for
    E > 0, it suffices to compare the result of random.random() with exp(-2*beta*h*spin). This is 
    the standard thing we do with the Metropolis-Hastings algorithm, but exploiting the fact that
    exp(0) = 1 simplifies matters, since it lets us collapse the min(1, exp(-a)) comparison into a
    single line.

    Note that here, I iterate over the array size, rather than the array elements. This is because
    I'm not sure how to call individual entries in the 2D array aside from listing their
    positions.'''


# This function performs the MC thermalisation.
def MC_thermal(collec, therm_steps, mag_field, couplx, couply, t):
    now_collec = collec

    for indiv_step in xrange(therm_steps):
        now_collec = MC_update(now_collec, mag_field, couplx, couply, t)[0]

    return now_collec


# This function performs several Monte Carlo updates, with the number of Monte Carlo updates specified by MC_iter.
def many_MC(lat, MC_iter, mu, cc_x, cc_y, t):
    MC_M = []
    MC_E = []
    
    x_dist = len(lat[0])
    y_dist = len(lat)
    sites = float(x_dist * y_dist)

    beta = 1.0/t

    now_lat = lat
    
    for update in xrange(MC_iter):
        now_update = MC_update(lat, mu, cc_x, cc_y, t)
        now_lat = now_update[0]
        
        MC_M.append(now_update[1])
        MC_E.append(now_update[2])
    
    avg_M = numpy.mean(MC_M, axis = None)
    avg_m = float(avg_M / sites)
    avg_E = numpy.mean(MC_E, axis = None)
    avg_e = float(avg_E / sites)
    sus = beta * numpy.var(MC_M, axis = None) / sites
    cv = math.pow(beta, 2) * numpy.var(MC_E, axis = None) / sites

    return (now_lat, avg_M, avg_m, avg_E, avg_e, sus, cv)


# This function defines the hyperbolic secant squared function, used in the ideal values, via numpy.
def sech2(params):
    sech_params = 1/numpy.cosh(params)
    sech2_params = numpy.power(sech_params, 2.0)

    return sech2_params


# This function gives us the ideal values for <m>, <u>, the susceptibility, and the specific heat for the 0NN case.
def ideal_vals_0NN(ideal_T_0NN, ideal_h_0NN):
    ideal_b_0NN = 1.0 / ideal_T_0NN
    ideal_bh_0NN = ideal_b_0NN * ideal_h_0NN

    ideal_0NN_m = numpy.tanh(ideal_bh_0NN)
    ideal_0NN_u = -ideal_h_0NN * numpy.tanh(ideal_bh_0NN)
    ideal_0NN_sus = ideal_b_0NN * sech2(ideal_bh_0NN)
    ideal_0NN_cv = numpy.power(ideal_bh_0NN, 2.0) * sech2(ideal_bh_0NN)

    return (ideal_0NN_m, ideal_0NN_u, ideal_0NN_sus, ideal_0NN_cv)


# This function sweeps across values of the external field and temperature for the 0NN case.
def sweep_0NN(lat_i_0NN, h_min_0NN, h_max_0NN, T_min_0NN, T_max_0NN, MC_iter_0NN, points_0NN, therm_steps_0NN):
    sweep_0NN_m_vals = []
    sweep_0NN_u_vals = []
    sweep_0NN_chi_vals = []
    sweep_0NN_cv_vals = []

    h_now_0NN = h_min_0NN
    T_now_0NN = T_min_0NN
    
    h_step_0NN = float((h_max_0NN - h_min_0NN) / points_0NN)
    T_step_0NN = float((T_max_0NN - T_min_0NN) / points_0NN)
    
    for point_0NN in xrange(points_0NN + 1):
        #(collec, therm_steps, mag_field, couplx, couply, t)
        now_lat_0NN = MC_thermal(lat_i_0NN, therm_steps_0NN, h_now_0NN, 0.0, 0.0, T_now_0NN)
        MC_results_now_0NN = many_MC(now_lat_0NN, MC_iter_0NN, h_now_0NN, 0.0, 0.0, T_now_0NN)
        ideal_vals_now_0NN = ideal_vals_0NN(T_now_0NN, h_now_0NN)

        ideal_m_now_0NN = ideal_vals_now_0NN[0]
        m_diff_now_0NN = ideal_m_now_0NN - MC_results_now_0NN[2]
        m_vals_now_0NN = [T_now_0NN, h_now_0NN, ideal_m_now_0NN, m_diff_now_0NN]
        sweep_0NN_m_vals.append(m_vals_now_0NN)

        ideal_u_now_0NN = ideal_vals_now_0NN[1]
        u_diff_now_0NN = ideal_u_now_0NN - MC_results_now_0NN[4]
        u_vals_now_0NN = [T_now_0NN, h_now_0NN, ideal_u_now_0NN, u_diff_now_0NN]
        sweep_0NN_u_vals.append(u_vals_now_0NN)

        ideal_chi_now_0NN = ideal_vals_now_0NN[2]
        chi_diff_now_0NN = ideal_chi_now_0NN - MC_results_now_0NN[5]
        chi_vals_now_0NN = [T_now_0NN, h_now_0NN, ideal_chi_now_0NN, chi_diff_now_0NN]
        sweep_0NN_chi_vals.append(chi_vals_now_0NN)

        ideal_cv_now_0NN = ideal_vals_now_0NN[3]
        cv_diff_now_0NN = ideal_cv_now_0NN - MC_results_now_0NN[6]
        cv_vals_now_0NN = [T_now_0NN, h_now_0NN, ideal_cv_now_0NN, cv_diff_now_0NN]
        sweep_0NN_cv_vals.append(cv_vals_now_0NN)

        h_now_0NN += h_step_0NN
        T_now_0NN += T_step_0NN

    return (sweep_0NN_m_vals, sweep_0NN_u_vals, sweep_0NN_chi_vals, sweep_0NN_cv_vals)


# This function sweeps across values of the external field and temperature for the 1NN 2D case.
def sweep_1NN(lat_i_1NN, h_min_1NN, h_max_1NN, Jx_min_1NN, Jx_max_1NN, Jy_min_1NN, Jy_max_1NN, T_min_1NN, T_max_1NN, MC_iter_1NN, points_1NN, therm_steps_1NN):
    sweep_vals = []

    h_now_1NN = h_min_1NN
    Jx_now_1NN = Jx_min_1NN
    Jy_now_1NN = Jy_min_1NN
    T_now_1NN = T_min_1NN

    h_step_1NN = float((h_max_1NN - h_min_1NN) / points_1NN)
    Jx_step_1NN = float((Jx_max_1NN - Jx_min_1NN) / points_1NN)
    Jy_step_1NN = float((Jy_max_1NN - Jy_min_1NN) / points_1NN)
    T_step_1NN = float((T_max_1NN - T_min_1NN) / points_1NN)
    
    point_1NN = 0
    while point_1NN <= points_1NN:
        curr_lat_1NN = MC_thermal(lat_i_1NN, therm_steps_1NN, h_now_1NN, Jx_now_1NN, Jy_now_1NN, T_now_1NN)
        MC_results_now_1NN = many_MC(curr_lat_1NN, MC_iter_1NN, h_now_1NN, Jx_now_1NN, Jy_now_1NN, T_now_1NN)
        results_now = [T_now_1NN, h_now_1NN, Jx_now_1NN, Jy_now_1NN, MC_results_now_1NN[2], MC_results_now_1NN[4], MC_results_now_1NN[5], MC_results_now_1NN[6]]
        sweep_vals.append(results_now)
        
        h_now_1NN += h_step_1NN
        Jx_now_1NN += Jx_step_1NN
        Jy_now_1NN += Jy_step_1NN
        T_now_1NN += T_step_1NN
        point_1NN += 1
    
    return sweep_vals


# This function provides the outputs (the relevant graphs and tables) for the 1NN 2D case.
def output_1NN(grid_i_1NN, h_i_1NN, h_f_1NN, Jx_i_1NN, Jx_f_1NN, Jy_i_1NN, Jy_f_1NN, T_i_1NN, T_f_1NN, MC_num_1NN, pts_1NN, thrm_stps_1NN):
    vals_1NN = numpy.array(sweep_1NN(grid_i_1NN, h_i_1NN, h_f_1NN, Jx_i_1NN, Jx_f_1NN, Jy_i_1NN, Jy_f_1NN, T_i_1NN, T_f_1NN, MC_num_1NN, pts_1NN, thrm_stps_1NN))

    T_vals_1NN = vals_1NN[:,0]
    m_vals_1NN = vals_1NN[:,4]
    e_vals_1NN = vals_1NN[:,5]
    sus_vals_1NN = vals_1NN[:,6]
    cv_vals_1NN = vals_1NN[:,7]
    
    print tabulate.tabulate(vals_1NN, headers = ["Temp.", "Ext. Field", "x-dir. cc", "y-dir. cc", "Sim. <m>", "Sim. <e>", u"Sim. \u03c7", "Sim. c_v"], floatfmt=".7f")

    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.suptitle("Per Site Magnetisation", family = "Gill Sans MT", fontsize = 16)
    matplotlib.pyplot.xlabel(r"Temperature ($T$)", family = "Gill Sans MT")
    matplotlib.pyplot.ylabel(r"Per Site Magnetisation ($\langle m \rangle$)", family = "Gill Sans MT")
    matplotlib.pyplot.scatter(T_vals_1NN, m_vals_1NN)
    matplotlib.pyplot.show()

    matplotlib.pyplot.figure(2)
    matplotlib.pyplot.suptitle("Per Site Energy", family = "Gill Sans MT", fontsize = 16)
    matplotlib.pyplot.xlabel(r"Temperature ($T$)", family = "Gill Sans MT")
    matplotlib.pyplot.ylabel(r"Per Site Energy ($\langle u \rangle$)", family = "Gill Sans MT")
    matplotlib.pyplot.scatter(T_vals_1NN, e_vals_1NN)
    matplotlib.pyplot.show()

    matplotlib.pyplot.figure(3)
    matplotlib.pyplot.suptitle("Magnetic Susceptibility", family = "Gill Sans MT", fontsize = 16)
    matplotlib.pyplot.xlabel(r"Temperature ($T$)", family = "Gill Sans MT")
    matplotlib.pyplot.ylabel(r"Magnetic Susceptibility ($\chi$)", family = "Gill Sans MT")
    matplotlib.pyplot.scatter(T_vals_1NN, sus_vals_1NN)
    matplotlib.pyplot.show()

    matplotlib.pyplot.figure(4)
    matplotlib.pyplot.suptitle("Specific Heat (at Constant Volume and Number of Particles)", family = "Gill Sans MT", fontsize = 16)
    matplotlib.pyplot.xlabel(r"Temperature ($T$)", family = "Gill Sans MT")
    matplotlib.pyplot.ylabel(r"Specific Heat ($c_v$)", family = "Gill Sans MT")
    matplotlib.pyplot.scatter(T_vals_1NN, cv_vals_1NN)
    matplotlib.pyplot.show()

    return None


# Here, we run the simulation. For testing, we also print the actual arrays; these commands are then commented out as necessary.
print "Initial 2D Ising Grid:"
print "                      "
print_grid(initial_grid)
print "                      "
print "                      "
updated_grid = many_MC(initial_grid, MC_num, h_start, Jx_start, Jy_start, T_start)
print "Updated 2D Ising Grid:"
print "                      "
print_grid(updated_grid[0])
output_1NN(initial_grid, h_start, h_end, Jx_start, Jx_end, Jy_start, Jy_start, T_start, T_end, MC_num, sweeps, MC_therm_steps)


# This section stores the time at the end of the program.
program_end_time = time.clock()
total_program_time = program_end_time - program_start_time
print "                      "
print "Program run time: %f seconds" % (total_program_time)
print "Program run time per site per MC sweep: %6g seconds" % (total_program_time / (MC_num * sweeps * size))

'''Note: To find out how long the program takes, we take the difference of time.clock() evaluated at
   the beginning of the program and at the end of the program. Here, we take the time at the end of
   the program, and define the total program time.'''