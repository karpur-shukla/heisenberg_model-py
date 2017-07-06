# -*- coding: utf-8 -*-

''' Here, we create a static 2D N-by-M Ising grid of spins up and down, an update mechanism to
    update the spin at every site, and finally include the presence of an external magnetic field in
    the grids. We're specifically looking at the two-state Ising model, i.e. with spins ±1/2, but
    with no inter-site coupling. '''


# This section imports the libraries necessary to run the program.
import math
import matplotlib
import numpy
import random
import time
import tabulate


# This section stores the time at the start of the program.
program_start_time = time.clock()

''' Since we're interested in the amount of time the program will run in, we'll store the time at
    the beginning of the program using time.clock(), and compare it to the time at the end (again
    using time.clock(), applied to a different variable name. time.clock() just takes the time at a
    given moment; it's up to us to store it properly. '''


# This section sets the simulation parameters.
x_len = 8                    # x_len is the length of each row in the 2D grid.
y_len = 8                    # y_len is the number of rows in the 2D grid.
size = x_len * y_len         # size is the size of the array.

MC_sweeps = 1000000          # MC_sweeps is the number of Monte Carlo sweeps.
MC_therm_steps = 10000       # MC_therm_steps is the number of thermalisation steps.

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

kNN_2pt_G_dist = 1           # kNN_2pt_G_dist is the distance at which we're looking at the kth nearest-neighbour two-point Green function.
kNN_2pt_G_conn = False       # kNN_2pt_G_conn tells us whether we're looking at the two-point disconnected or the two-point connected Green function.

data_pts = 40                # data_pts is the number of sweep data points for the graphs/tables.


# This section creates the initial system, a static 2D array of spins (up or down).
initial_grid = numpy.random.choice([-1, 1], size = [y_len, x_len])

''' Note that this is faster than my original choice of how to initialise the system: 
    initial_grid = [[-1.0 if random.random() <= 0.5 else 1.0 for cube in xrange(x_len)] for row in xrange(y_len)] '''


# This function provides a printed version of the 2D Ising grid.
def print_grid(grating):
    Ising_grid_printed = []
    
    for chain in grating:
        IG_single_row = []
        
        for entry in chain:
            if entry == -1.0:
                IG_single_row += ["-"]
            elif entry == +1.0:
                IG_single_row += ["+"]
            else:
                raise ArithmeticError("Ising spin must be +1.0 or -1.0")
        
        IG_single_row_printed = " ".join(IG_single_row)
        Ising_grid_printed += [IG_single_row_printed]
    
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
            dE += Jx * lat[y_pos][(x_pos-1) % x_len] * lat[y_pos][x_pos]
            dE += Jx * lat[y_pos][(x_pos+1) % x_len] * lat[y_pos][x_pos]
            dE += Jy * lat[(y_pos-1) % y_len][x_pos] * lat[y_pos][x_pos]
            dE += Jy * lat[(y_pos+1) % y_len][x_pos] * lat[y_pos][x_pos]
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


# This function retrieves the magnetisation, energy, and kth nearest-neighbour two-point Green function of a given lattice.
def lat_props(trel, mu, ccx, ccy, temp, dist, conn):
    net_M = 0.0
    net_E = 0.0
    net_corr = 0.0
    
    x_size = len(trel[0])
    y_size = len(trel)
    
    sites = float(x_size * y_size)

    for y_pt in xrange(y_size):
        for x_pt in xrange(x_size):
            curr_site = trel[y_pt][x_pt]
            next_site_down = trel[(y_pt + dist) % y_size][x_pt]
            next_site_right = trel[y_pt][(x_pt + dist) % x_size]
            
            net_M += curr_site
            
            net_E += -mu * curr_site
            net_E += -ccx * trel[y_pt][(x_pt + 1) % x_size] * curr_site
            net_E += -ccy * trel[(y_pt + 1) % y_size][x_pt] * curr_site
            
            net_corr += curr_site * next_site_down
            net_corr += curr_site * next_site_right
            
    
    lat_m = net_M / sites
    lat_e = net_E / sites
    disc_corr_func = net_corr / sites
    conn_corr_func = disc_corr_func - (lat_m ** 2.0)
    
    if conn == True:
        return (net_M, lat_m, net_E, lat_e, conn_corr_func)
    
    elif conn == False:
        return (net_M, lat_m, net_E, lat_e, disc_corr_func)
    
    else:
        raise TypeError("'conn' must be of type bool")

''' Note that this gives either the kth nearest-neighbour two-point connected correlation function
    (i.e. G^(2)_c(i, i+k) = <x_i x_(i+k)> - <x_i><x_(i+k)>) or the kth nearest-neighbour two-point
    disconnected correlation function (i.e. G^(2)(i, i+k) = <x_i x_(i+k)>), depending on whether or
    not we have conn = True or conn = False. Since <x_i> = <x_(i+k)> = m (the average per-site
    magnetisation), the two-point connected correlation function just substitutes m^2 for
    <x_i><x_(i+k)>. '''


# This function performs the MC thermalisation.
def MC_thermal(collec, therm_steps, mag_field, couplx, couply, t):
    now_collec = collec

    for indiv_step in xrange(therm_steps):
        now_collec = MC_update(now_collec, mag_field, couplx, couply, t)        

    return now_collec


# This function performs several Monte Carlo updates, with the number of Monte Carlo updates specified by MC_iter.
def many_MC(array, MC_iter, ext_field, cc_x, cc_y, tepl, therm_steps_per_sample, many_MC_G_dist, many_MC_G_corr):
    MC_M = [0] * MC_iter
    MC_E = [0] * MC_iter
    MC_G = [0] * MC_iter
    
    x_dist = len(array[0])
    y_dist = len(array)
    points = float(x_dist * y_dist)

    b = 1.0/tepl

    now_lat = array
    
    for update in xrange(MC_iter):
        now_lat = MC_thermal(now_lat, therm_steps_per_sample, ext_field, cc_x, cc_y, tepl)
        now_update = MC_update(now_lat, ext_field, cc_x, cc_y, tepl)
        now_props = lat_props(now_update, ext_field, cc_x, cc_y, tepl, many_MC_G_dist, many_MC_G_corr)
        now_lat = now_update
        
        MC_M[update] = now_props[0]
        MC_E[update] = now_props[2]
        MC_G[update] = now_props[4]
    
    avg_M = numpy.mean(MC_M, axis = None)
    avg_m = float(avg_M / points)
    avg_E = numpy.mean(MC_E, axis = None)
    avg_e = float(avg_E / points)
    avg_G = numpy.mean(MC_G, axis = None)
    
    cv = math.pow(b, 2) * numpy.var(MC_E, axis = None) / points

    if ext_field != 0.0:
        sus = b * numpy.var(MC_M, axis = None) / points
    else:
        sus = b * numpy.var(MC_M, axis = None) / (points ** 2.0)
    
    return (now_lat, avg_M, avg_m, avg_E, avg_e, avg_G, sus, cv, MC_M, MC_E, MC_G)

''' We need to do this for the susceptibility in the case of h = 0 because in this specific case, we
    have no interactions whatsoever. Thus, we're looking at the standard deviation of a set of ±1
    values picked at random; since there's no scale dependence, multiplying by array_sites in this
    specific case will give us an extraneous factor of array_sites. To write cv in terms of the
    total energy rather than the per-site energy, we have:
    cv = (math.pow(b, 2) * (avg_E2 - math.pow(avg_E, 2))) / array_sites. '''


# This function lets us sweep our MC simulation over different values of the magnetic moment, h.
def h_sweep(grid_in, h_min, h_max, sweep_Jx, sweep_Jy, sweep_T, MC_steps, points, h_therm_steps, h_sweep_G_dist, h_sweep_G_conn):
    h_sweep_m_vals = []
    h_sweep_e_vals = []
    h_sweep_susc_vals = []
    h_sweep_cv_vals = []

    h_step = float((h_max-h_min)/points)
    
    if h_step == 0.0:
        raise ArithmeticError("sweep range must be a nonzero float")
    
    else:
        h_now = h_min
        sweep_b = 1.0/sweep_T
        
        if h_step > 0.0:
            while h_now <= h_max:
                MC_results_now = many_MC(grid_in, MC_steps, h_now, sweep_Jx, sweep_Jy, sweep_T, h_therm_steps, h_sweep_G_dist, h_sweep_G_conn)
                now_lat = MC_results_now[0]
                
                ideal_m_now = math.tanh(sweep_b*h_now)
                m_diff_now = MC_results_now[2] - ideal_m_now
                h_sweep_m_vals += [[h_now, MC_results_now[2], ideal_m_now, m_diff_now]]
                
                ideal_e_now = -h_now * math.tanh(sweep_b*h_now)
                e_diff_now = MC_results_now[4] - ideal_e_now
                h_sweep_e_vals += [[h_now, MC_results_now[4], ideal_e_now, e_diff_now]]
                
                ideal_susc_now = sweep_b * math.pow(1/math.cosh(sweep_b*h_now), 2)
                susc_diff_now = MC_results_now[6] - ideal_susc_now
                h_sweep_susc_vals += [[h_now, MC_results_now[6], ideal_susc_now, susc_diff_now]]
                
                ideal_cv_now = math.pow(sweep_b, 2) * math.pow(h_now, 2) * math.pow(1/math.cosh(sweep_b*h_now), 2)
                cv_diff_now = MC_results_now[7] - ideal_cv_now
                h_sweep_cv_vals += [[h_now, MC_results_now[7], ideal_cv_now, cv_diff_now]]
                
                h_now += h_step
        
        else:
            while h_now >= h_max:
                MC_results_now = many_MC(grid_in, MC_steps, h_now, sweep_Jx, sweep_Jy, sweep_T, h_therm_steps, h_sweep_G_dist, h_sweep_G_conn)
                now_lat = MC_results_now[0]
                
                ideal_m_now = math.tanh(sweep_b*h_now)
                m_diff_now = MC_results_now[2] - ideal_m_now
                h_sweep_m_vals += [[h_now, MC_results_now[2], ideal_m_now, m_diff_now]]
                
                ideal_e_now = -h_now * math.tanh(sweep_b*h_now)
                e_diff_now = MC_results_now[4] - ideal_e_now
                h_sweep_e_vals += [[h_now, MC_results_now[4], ideal_e_now, e_diff_now]]
                
                ideal_susc_now = sweep_b * math.pow(1/math.cosh(sweep_b*h_now), 2)
                susc_diff_now = MC_results_now[6] - ideal_susc_now
                h_sweep_susc_vals += [[h_now, MC_results_now[6], ideal_susc_now, susc_diff_now]]
                
                ideal_cv_now = math.pow(sweep_b, 2) * math.pow(h_now, 2) * math.pow(1/math.cosh(sweep_b*h_now), 2)
                cv_diff_now = MC_results_now[7] - ideal_cv_now
                h_sweep_cv_vals += [[h_now, MC_results_now[7], ideal_cv_now, cv_diff_now]]
                
                h_now += h_step
    
    return (h_sweep_m_vals, h_sweep_e_vals, h_sweep_susc_vals, h_sweep_cv_vals)


# This function lets us sweep our MC simulation over different values of the temperature.
def T_sweep(lat_in, T_min, T_max, spect_h, spect_Jx, spect_Jy, MC_num, points, T_therm_steps, T_sweep_G_dist, T_sweep_G_conn):
    T_sweep_mag_vals = []
    T_sweep_ener_vals = []
    T_sweep_chi_vals = []
    T_sweep_spec_heat_vals = []
    
    T_step = float((T_max-T_min)/points)
    
    if T_step <= 0.0:
        raise ArithmeticError("sweep range must be a nonzero float")
    
    else:
        T_curr = T_min
        b_curr = 1.0/T_curr
        
        if T_step > 0.0:
            while T_curr <= T_max:
                MC_results_now = many_MC(lat_in, MC_num, spect_h, spect_Jx, spect_Jy, T_curr, T_therm_steps, T_sweep_G_dist, T_sweep_G_conn)
                curr_lat = MC_results_now[0]
                
                ideal_mag_now = math.tanh(b_curr*spect_h)
                mag_diff_now = MC_results_now[2] - ideal_mag_now
                T_sweep_mag_vals += [[T_curr, MC_results_now[2], ideal_mag_now, mag_diff_now]]
                
                ideal_ener_now = -spect_h * math.tanh(b_curr*spect_h)
                ener_diff_now = MC_results_now[4] - ideal_ener_now
                T_sweep_ener_vals += [[T_curr, MC_results_now[4], ideal_ener_now, ener_diff_now]]
                
                ideal_chi_now = b_curr * math.pow(1/math.cosh(b_curr*spect_h), 2)
                chi_diff_now = MC_results_now[6] - ideal_chi_now
                T_sweep_chi_vals += [[T_curr, MC_results_now[6], ideal_chi_now, chi_diff_now]]
                
                ideal_spec_heat_now = math.pow(b_curr, 2) * math.pow(spect_h, 2) * math.pow(1/math.cosh(b_curr*spect_h), 2)
                spec_heat_diff_now = MC_results_now[7] - ideal_spec_heat_now
                T_sweep_spec_heat_vals += [[T_curr, MC_results_now[7], ideal_spec_heat_now, spec_heat_diff_now]]
                T_curr += T_step

        else:
            while T_curr >= T_max:
                MC_results_now = many_MC(lat_in, MC_num, spect_h, spect_Jx, spect_Jy, T_curr, T_therm_steps, T_sweep_G_dist, T_sweep_G_conn)
                curr_lat = MC_results_now[0]
                
                ideal_mag_now = math.tanh(b_curr*spect_h)
                mag_diff_now = MC_results_now[2] - ideal_mag_now
                T_sweep_mag_vals += [[T_curr, MC_results_now[2], ideal_mag_now, mag_diff_now]]
                
                ideal_ener_now = -spect_h * math.tanh(b_curr*spect_h)
                ener_diff_now = MC_results_now[4] - ideal_ener_now
                T_sweep_ener_vals += [[T_curr, MC_results_now[4], ideal_ener_now, ener_diff_now]]
                
                ideal_chi_now = b_curr * math.pow(1/math.cosh(b_curr*spect_h), 2)
                chi_diff_now = MC_results_now[6] - ideal_chi_now
                T_sweep_chi_vals += [[T_curr, MC_results_now[6], ideal_chi_now, chi_diff_now]]
                
                ideal_spec_heat_now = math.pow(b_curr, 2) * math.pow(spect_h, 2) * math.pow(1/math.cosh(b_curr*spect_h), 2)
                spec_heat_diff_now = MC_results_now[7] - ideal_spec_heat_now
                T_sweep_spec_heat_vals += [[T_curr, MC_results_now[7], ideal_spec_heat_now, spec_heat_diff_now]]
                T_curr += T_step
    
    return (T_sweep_mag_vals, T_sweep_ener_vals, T_sweep_chi_vals, T_sweep_spec_heat_vals)


# This function lets us sweep our MC simulation over different values of the coupling constants.
def coup_sweep(kattam_in, Jx_min, Jx_max, Jy_min, Jy_max, h_over_range, T_over_range, MC_pts, step_points, coup_therm_steps, J_sweep_G_dist, J_sweep_G_conn):
    coup_sweep_mg_vals = []
    coup_sweep_u_vals = []
    coup_sweep_sscpt_vals = []
    coup_sweep_spc_heat_vals = []
    
    Jx_step = float((Jx_max - Jx_min)/step_points)
    Jy_step = float((Jy_max - Jy_min)/step_points)
    
    if Jx_step == 0.0 and Jy_step == 0.0:
        raise ArithmeticError("sweep range must be a nonzero float")
    
    else:
        Jx_jetzt = Jx_min
        Jy_jetzt = Jy_min
        b_over_range = 1.0/T_over_range
        
        if Jx_step >= 0.0 and Jy_step >= 0.0:
            while Jx_jetzt <= Jx_max and Jy_jetzt <= Jy_max:
                MC_results_now = many_MC(kattam_in, MC_pts, h_over_range, Jx_jetzt, Jy_jetzt, T_over_range, coup_therm_steps, J_sweep_G_dist, J_sweep_G_conn)
                curr_kattam = MC_results_now[0]
                
                ideal_mag_now = math.tanh(b_over_range*h_over_range)
                mag_diff_now = MC_results_now[2] - ideal_mag_now
                coup_sweep_mg_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[2], ideal_mag_now, mag_diff_now]]
                
                ideal_ener_now = -h_over_range * math.tanh(b_over_range*h_over_range)
                ener_diff_now = MC_results_now[4] - ideal_ener_now
                coup_sweep_u_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[4], ideal_ener_now, ener_diff_now]]
                
                ideal_chi_now = b_over_range * math.pow(1/math.cosh(b_over_range*h_over_range), 2)
                chi_diff_now = MC_results_now[6] - ideal_chi_now
                coup_sweep_sscpt_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[6], ideal_chi_now, chi_diff_now]]
                
                ideal_spec_heat_now = math.pow(b_over_range, 2) * math.pow(h_over_range, 2) * math.pow(1/math.cosh(b_over_range*h_over_range), 2)
                spec_heat_diff_now = MC_results_now[7] - ideal_spec_heat_now
                coup_sweep_spc_heat_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[7], ideal_spec_heat_now, spec_heat_diff_now]]
                
                Jx_jetzt += Jx_step
                Jy_jetzt += Jy_step
        
        elif Jx_step >= 0.0 and Jy_step <= 0.0:
            while Jx_jetzt <= Jx_max and Jy_jetzt >= Jy_max:
                MC_results_now = many_MC(curr_kattam, MC_pts, h_over_range, Jx_jetzt, Jy_jetzt, T_over_range, coup_therm_steps, J_sweep_G_dist, J_sweep_G_conn)
                curr_kattam = MC_results_now[0]
                
                ideal_mag_now = math.tanh(b_over_range*h_over_range)
                mag_diff_now = MC_results_now[2] - ideal_mag_now
                coup_sweep_mg_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[2], ideal_mag_now, mag_diff_now]]
                
                ideal_ener_now = -h_over_range * math.tanh(b_over_range*h_over_range)
                ener_diff_now = MC_results_now[4] - ideal_ener_now
                coup_sweep_u_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[4], ideal_ener_now, ener_diff_now]]
                
                ideal_chi_now = b_over_range * math.pow(1/math.cosh(b_over_range*h_over_range), 2)
                chi_diff_now = MC_results_now[6] - ideal_chi_now
                coup_sweep_sscpt_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[6], ideal_chi_now, chi_diff_now]]
                
                ideal_spec_heat_now = math.pow(b_over_range, 2) * math.pow(h_over_range, 2) * math.pow(1/math.cosh(b_over_range*h_over_range), 2)
                spec_heat_diff_now = MC_results_now[7] - ideal_spec_heat_now
                coup_sweep_spc_heat_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[7], ideal_spec_heat_now, spec_heat_diff_now]]
                
                Jx_jetzt += Jx_step
                Jy_jetzt += Jy_step
        
        elif Jx_step <= 0.0 and Jy_step >= 0.0:
            while Jx_jetzt >= Jx_max and Jy_jetzt <= Jy_max:
                MC_results_now = many_MC(curr_kattam, MC_pts, h_over_range, Jx_jetzt, Jy_jetzt, T_over_range, coup_therm_steps, J_sweep_G_dist, J_sweep_G_conn)
                curr_kattam = MC_results_now[0]
                
                ideal_mag_now = math.tanh(b_over_range*h_over_range)
                mag_diff_now = MC_results_now[2] - ideal_mag_now
                coup_sweep_mg_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[2], ideal_mag_now, mag_diff_now]]
                
                ideal_ener_now = -h_over_range * math.tanh(b_over_range*h_over_range)
                ener_diff_now = MC_results_now[4] - ideal_ener_now
                coup_sweep_u_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[4], ideal_ener_now, ener_diff_now]]
                
                ideal_chi_now = b_over_range * math.pow(1/math.cosh(b_over_range*h_over_range), 2)
                chi_diff_now = MC_results_now[6] - ideal_chi_now
                coup_sweep_sscpt_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[6], ideal_chi_now, chi_diff_now]]
                
                ideal_spec_heat_now = math.pow(b_over_range, 2) * math.pow(h_over_range, 2) * math.pow(1/math.cosh(b_over_range*h_over_range), 2)
                spec_heat_diff_now = MC_results_now[7] - ideal_spec_heat_now
                coup_sweep_spc_heat_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[7], ideal_spec_heat_now, spec_heat_diff_now]]
                
                Jx_jetzt += Jx_step
                Jy_jetzt += Jy_step
    
        else:
            while Jx_jetzt >= Jx_max and Jy_jetzt >= Jy_max:
                MC_results_now = many_MC(curr_kattam, MC_pts, h_over_range, Jx_jetzt, Jy_jetzt, T_over_range, coup_therm_steps, J_sweep_G_dist, J_sweep_G_conn)
                curr_kattam = MC_results_now[0]
                
                ideal_mag_now = math.tanh(b_over_range*h_over_range)
                mag_diff_now = MC_results_now[2] - ideal_mag_now
                coup_sweep_mg_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[2], ideal_mag_now, mag_diff_now]]
                
                ideal_ener_now = -h_over_range * math.tanh(b_over_range*h_over_range)
                ener_diff_now = MC_results_now[4] - ideal_ener_now
                coup_sweep_u_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[4], ideal_ener_now, ener_diff_now]]
                
                ideal_chi_now = b_over_range * math.pow(1/math.cosh(b_over_range*h_over_range), 2)
                chi_diff_now = MC_results_now[6] - ideal_chi_now
                coup_sweep_sscpt_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[6], ideal_chi_now, chi_diff_now]]
                
                ideal_spec_heat_now = math.pow(b_over_range, 2) * math.pow(h_over_range, 2) * math.pow(1/math.cosh(b_over_range*h_over_range), 2)
                spec_heat_diff_now = MC_results_now[7] - ideal_spec_heat_now
                coup_sweep_spc_heat_vals += [[Jx_jetzt, Jy_jetzt, MC_results_now[7], ideal_spec_heat_now, spec_heat_diff_now]]
                
                Jx_jetzt += Jx_step
                Jy_jetzt += Jy_step
    
    return (coup_sweep_mg_vals, coup_sweep_u_vals, coup_sweep_sscpt_vals, coup_sweep_spc_heat_vals)

''' This does provide information for the 1NN case if we chose to do 1NN stuff, but that really
    should be handled by the 1NN script. '''


# This function provides the outputs (the relevant graphs and tables) for the 0NN case.
def output_0NN(sweep_param, rooster_in, MC_amt, data_pt_amt, therm_step_amt, sweep_param_vals):
    matplotlib.pyplot.close("all")
    
    if sweep_param == "h":
        
        h_init = sweep_param_vals[0]
        h_final = sweep_param_vals[1]
        Jx_h_sweep = sweep_param_vals[2]
        Jy_h_sweep = sweep_param_vals[3]
        T_h_sweep = sweep_param_vals[4]
        G_dist_h_sweep = sweep_param_vals[5]
        G_conn_h_sweep = sweep_param_vals[6]
        
        b_h_sweep = 1.0/T_h_sweep
        h_sweep_grids = h_sweep(rooster_in, h_init, h_final, Jx_h_sweep, Jy_h_sweep, T_h_sweep, MC_amt, data_pt_amt, therm_step_amt, G_dist_h_sweep, G_conn_h_sweep)
        swept_h_range = numpy.linspace(h_init, h_final, MC_amt * 1000, endpoint = True)
        
        swept_h_m_grid = numpy.array(h_sweep_grids[0])
        swept_h_e_grid = numpy.array(h_sweep_grids[1])
        swept_h_susc_grid = numpy.array(h_sweep_grids[2])
        swept_h_cv_grid = numpy.array(h_sweep_grids[3])
        
        swept_h_ext_field = swept_h_m_grid[:,0]
        swept_h_m = swept_h_m_grid[:,1]
        swept_h_ideal_m = numpy.tanh(b_h_sweep * swept_h_range)
        swept_h_e = swept_h_e_grid[:,1]
        swept_h_ideal_e = -swept_h_range * numpy.tanh(b_h_sweep * swept_h_range)
        swept_h_susc = swept_h_susc_grid[:,1]
        swept_h_ideal_susc = b_start * numpy.power(1/numpy.cosh(b_h_sweep*swept_h_range), 2.0)
        swept_h_cv = swept_h_cv_grid[:,1]
        swept_h_ideal_cv = math.pow(b_h_sweep, 2.0) * numpy.power(swept_h_range, 2.0) * numpy.power(1/numpy.cosh(b_h_sweep*swept_h_range), 2.0)
        
        print "T = %.4f, Jx = %.4f, Jy = %.4f" % (T_h_sweep, Jx_h_sweep, Jy_h_sweep)
        print "                      "
        print tabulate.tabulate(swept_h_m_grid, headers = ["Ext. Field", "Sim. <m>", "Calc. <m>", "<m> err."], floatfmt=".7f")
        
        print "                      "
        print tabulate.tabulate(swept_h_e_grid, headers = ["Ext. Field", "Sim. <e>", "Calc. <e>", "<e> err."], floatfmt=".7f")
        
        print "                      "
        print tabulate.tabulate(swept_h_susc_grid, headers = ["Ext. Field", u"Sim. \u03c7", u"Calc. \u03c7", u"\u03c7 err."], floatfmt=".7f")
        
        print "                      "
        print tabulate.tabulate(swept_h_cv_grid, headers = ["Ext. Field", "Sim. c_v", "Calc. c_v", "c_v err."], floatfmt=".7f")
        
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
        
        return
    
    elif sweep_param == "T":

        T_init = sweep_param_vals[0]
        T_final = sweep_param_vals[1]
        h_T_sweep = sweep_param_vals[2]
        Jx_T_sweep = sweep_param_vals[3]
        Jy_T_sweep = sweep_param_vals[4]
        G_dist_T_sweep = sweep_param_vals[5]
        G_conn_T_sweep = sweep_param_vals[6]

        T_sweep_grids = T_sweep(rooster_in, T_init, T_final, h_T_sweep, Jx_T_sweep, Jy_T_sweep, MC_amt, data_pt_amt, therm_step_amt, G_dist_T_sweep, G_conn_T_sweep)
        swept_T_range = numpy.linspace(T_init, T_final, MC_amt * 1000, endpoint = True)
        swept_b_range = numpy.linspace(1.0/T_final, 1.0/T_init, MC_amt * 1000, endpoint = True)
        
        swept_T_m_grid = numpy.array(T_sweep_grids[0])
        swept_T_e_grid = numpy.array(T_sweep_grids[1])
        swept_T_susc_grid = numpy.array(T_sweep_grids[2])
        swept_T_cv_grid = numpy.array(T_sweep_grids[3])
        
        swept_T_vals = swept_T_m_grid[:,0]
        swept_T_m = swept_T_m_grid[:,1]
        swept_T_ideal_m = numpy.tanh(swept_b_range * h_T_sweep)
        swept_T_e = swept_T_e_grid[:,1]
        swept_T_ideal_e = -h_T_sweep * numpy.tanh(swept_b_range * h_T_sweep)
        swept_T_susc = swept_T_susc_grid[:,1]
        swept_T_ideal_susc = b_start * numpy.power(1/numpy.cosh(swept_b_range*h_T_sweep), 2.0)
        swept_T_cv = swept_T_cv_grid[:,1]
        swept_T_ideal_cv = numpy.power(swept_b_range, 2.0) * math.pow(h_T_sweep, 2.0) * numpy.power(1/numpy.cosh(swept_b_range*h_T_sweep), 2.0)
        
        print "T = %.4f, Jx = %.4f, Jy = %.4f" % (T_h_sweep, Jx_h_sweep, Jy_h_sweep)
        print "                      "
        print tabulate.tabulate(swept_T_m_grid, headers = ["Ext. Field", "Sim. <m>", "Calc. <m>", "<m> err."], floatfmt=".7f")
        
        print "                      "
        print tabulate.tabulate(swept_T_e_grid, headers = ["Ext. Field", "Sim. <e>", "Calc. <e>", "<e> err."], floatfmt=".7f")
        
        print "                      "
        print tabulate.tabulate(swept_T_susc_grid, headers = ["Ext. Field", u"Sim. \u03c7", u"Calc. \u03c7", u"\u03c7 err."], floatfmt=".7f")
        
        print "                      "
        print tabulate.tabulate(swept_T_cv_grid, headers = ["Ext. Field", "Sim. c_v", "Calc. c_v", "c_v err."], floatfmt=".7f")
        
        matplotlib.pyplot.figure(1)
        matplotlib.pyplot.suptitle("Per Site Magnetisation", family = "Gill Sans MT", fontsize = 16)
        matplotlib.pyplot.xlabel(r"Temperature ($T$)", family = "Gill Sans MT")
        matplotlib.pyplot.ylabel(r"Per Site Magnetisation ($\langle m \rangle$)", family = "Gill Sans MT")
        matplotlib.pyplot.plot(swept_T_range, swept_T_ideal_m, linestyle = "-", color = "#2824A7")
        matplotlib.pyplot.scatter(swept_T_vals, swept_T_m, color = "#FFAA00")
        
        matplotlib.pyplot.figure(2)
        matplotlib.pyplot.suptitle("Average Per-Site Energy", family = "Gill Sans MT", fontsize = 16)
        matplotlib.pyplot.xlabel(r"Temperature ($T$)", family = "Gill Sans MT")
        matplotlib.pyplot.ylabel(r"Average Per-Site Energy ($\langle u \rangle$)", family = "Gill Sans MT")
        matplotlib.pyplot.plot(swept_T_range, swept_T_ideal_e, linestyle = "-", color = "#2824A7")
        matplotlib.pyplot.scatter(swept_T_vals, swept_T_e, color = "#FFAA00")
        matplotlib.pyplot.show()
        
        matplotlib.pyplot.figure(3)
        matplotlib.pyplot.suptitle("Susceptibility", family = "Gill Sans MT", fontsize = 16)
        matplotlib.pyplot.xlabel(r"Temperature ($T$)", family = "Gill Sans MT")
        matplotlib.pyplot.ylabel(r"Susceptibility ($\chi$)", family = "Gill Sans MT")
        matplotlib.pyplot.plot(swept_T_range, swept_T_ideal_susc, linestyle = "-", color = "#2824A7")
        matplotlib.pyplot.scatter(swept_T_vals, swept_T_susc, color = "#FFAA00")
        matplotlib.pyplot.show()
        
        matplotlib.pyplot.figure(4)
        matplotlib.pyplot.suptitle("Specific Heat at Constant Volume", family = "Gill Sans MT", fontsize = 16)
        matplotlib.pyplot.xlabel(r"Temperature ($T$)", family = "Gill Sans MT")
        matplotlib.pyplot.ylabel(r"Specific Heat at Constant Volume ($c_v$)", family = "Gill Sans MT")
        matplotlib.pyplot.plot(swept_T_range, swept_T_ideal_cv, linestyle = "-", color = "#2824A7")
        matplotlib.pyplot.scatter(swept_T_vals, swept_T_cv, color = "#FFAA00")
        matplotlib.pyplot.show()
    
    else:
        raise ValueError('sweep_param must be "h" or "T"')
    
    return

''' sweep_param_vals holds all of the sweep-related values:
        * For h sweep, sweep_param_vals = [h_init, h_final, Jx_h_sweep, Jy_h_sweep, T_h_sweep]
        * For T sweep, sweep_param_vals = [T_init, T_final, h_T_sweep, Jx_T_sweep, Jy_T_sweep]
        * For J sweep, sweep_param_vals = [Jx_init, Jx_final, Jy_init, Jy_final, h_J_sweep, T_J_sweep]

    Meanwhile, "rooster" is Dutch and Afrikaans for "grid". 
    
    I didn't give results for the 1NN case, since that's given by the 1NN script. '''


# Here, we run the simulation. For testing, we also print the actual arrays; these commands are then commented out as necessary.
print "Initial 2D Ising Grid:"
print "                      "
print_grid(initial_grid)
print "                      "
print "                      "


h_sweep_out = output_0NN("h", initial_grid, MC_sweeps, data_pts, MC_therm_steps, [h_start, h_end, Jx_start, Jy_start, T_start, kNN_2pt_G_dist, kNN_2pt_G_conn])


# This section stores the time at the end of the program.
program_end_time = time.clock()
total_program_time = program_end_time - program_start_time
print "                      "
print "Program run time: %f seconds" % (total_program_time)
print "Program run time per site per MC sweep: %f seconds" % (total_program_time/(MC_sweeps * MC_therm_steps * data_pts))

''' Note: To find out how long the program takes, we take the difference of time.clock() evaluated
    at the beginning of the program and at the end of the program. Here, we take the time at the end
    of the program, and define the total program time. '''
