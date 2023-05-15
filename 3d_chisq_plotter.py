"""
Script that plots chi-sqaured surface and optimal
models in the grid.

Also prints out physical parameters and uncertainties
of optimal models.

author: Ciarán Furey.
"""

from lp26_models import load_models, get_renorm_lines

from matplotlib import lines

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spinterp
import matplotlib.patches as mpatches
import scipy.stats as sps

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": "Helvetica",
# })

class Chisq3D:
    """Plots chisqs for each abundance.

    Clicking on the plot will plot the chisq curves
    at the gravity closest to the x position of the
    mouse.
    """
    def __init__(self,lines_in_data, data_errors, models, model_names,
                 model_line_names, lines_to_use, data_names=None,
                 fig=None, ax=None):

        self.lines_in_data = lines_in_data
        self.data_errors = data_errors
        self.models = models # Model lines
        self.model_names = model_names
        self.model_line_names = model_line_names
        self.lines_to_use = lines_to_use
        self.data_names = data_names

        self.fig = fig
        self.ax = ax

        sorted_idxs = np.argsort(self.model_names)
        self.model_names = np.array(self.model_names)[sorted_idxs]
        self.models = np.array(self.models)[sorted_idxs]

        # array that will store model lines to use for chisq depending
        # on data_names input.
        self.model_lines_to_use = []

        # FInd where in data the desired lines are.
        self.indices = np.where(np.in1d(self.data_names, self.lines_to_use))[0]
        # Find where in models the desired lines are.
        self.model_line_indices= np.where(np.in1d(self.model_line_names,
                                         self.lines_to_use))[0]

        # Get lines and errors from data.
        self.data_lines_to_use = np.array(self.lines_in_data)[self.indices]
        self.data_errs_to_use = []
        for i in self.indices:
            self.data_errs_to_use.append(self.data_errors[i])


        # Also store abundances, gravities, and Temps.
        abuns = []
        gravs = []
        temps = []

        for model, name in zip(self.models, self.model_names):
            lines_to_use = []
            for i in self.model_line_indices:
                lines_to_use.append(model[i])
            self.model_lines_to_use.append(lines_to_use)
            abuns.append(float(name[:4]))
            gravs.append(float(name[5:9]))
            temps.append(float(name[10:-5]))

        self.unique_abuns = np.unique(abuns)
        self.unique_gravs = np.unique(gravs)
        self.unique_temps = np.unique(temps)

        self.no_abuns = len(self.unique_abuns)
        self.no_gravs = len(self.unique_gravs)
        self.no_temps = len(self.unique_temps)

        chi_sq_for_each_line = np.zeros((len(self.lines_to_use), len(self.models)))
        for i, model in enumerate(self.model_lines_to_use): # go through each model
            for j, (model_line, data_line) in enumerate(zip(model, self.data_lines_to_use)):
                # go through each line to calculate chisq for

                # axs[i].plot(model_line[0], model_line[1], alpha=0.3)
                # Interpolate function to interp model flux to data
                # wavelength.
                line_interp_func = spinterp.interp1d(model_line[0],
                                                    model_line[1])
                                                    #fill_value="extrapolate")
                # Now get model fluxes at data wavelengths.
                line_interp = line_interp_func(data_line[0])

                # Calculate the chisq stat
                chisq = np.sum(((data_line[1] - line_interp)/self.data_errs_to_use[j])**2)

                # Store the chisq for line i of model j.
                chi_sq_for_each_line[j][i] = chisq

        # Now calculate total chisq for each line.
        chisqs = np.sum(chi_sq_for_each_line, axis=0)

        self.t_grid = np.empty((self.no_abuns, self.no_gravs,
                               self.no_temps,2))
        self.chisq_grid = np.zeros((self.no_abuns, self.no_gravs,
                               self.no_temps), dtype=float)
        self.model_line_grid = np.zeros((self.no_abuns, self.no_gravs,
                               self.no_temps), dtype=object)

        model_idx = 0
        for i in range(self.no_abuns):
            # Current abundance.
            ab = str(self.unique_abuns[i])
            for j in range(self.no_gravs):
                # Current log(g)
                gr = str(self.unique_gravs[j])
                for k in range(self.no_temps):
                    # Current Teff.
                    te = str(self.unique_temps[k])

                    # Need to check if model successfully ran, can omit
                    # from grid if not.
                    # To do so, check if the current parameters of the loop
                    # are in the model name. If it did not run, does not
                    # appear in the model names list.
                    if ((ab in self.model_names[model_idx]) and \
                        (gr in self.model_names[model_idx]) and \
                        (te in self.model_names[model_idx]) and \
                        model_idx < len(self.model_names)):

                        self.t_grid[i][j][k] = [float(gr),float(te)]#model_names[model_idx]
                        self.chisq_grid[i][j][k] = chisqs[model_idx]
                        self.model_line_grid[i][j][k] = self.model_lines_to_use[model_idx]
                        model_idx += 1
                    else:
                        # If it didnt appear, set it as a nan, for masking.
                        self.t_grid[i][j][k]= np.nan
                        self.chisq_grid[i][j][k] = np.nan
                        self.model_line_grid[i][j][k] = np.nan

        self.print_opt_params()

        self.plot_chisq()

    def plot_chisq(self):

        grav_range = np.linspace(self.unique_gravs.min(), self.unique_gravs.max(), 100)
        temp_range = np.linspace(self.unique_temps.min(), self.unique_temps.max(), 100)

        grv_grid, tmp_grid = np.meshgrid(grav_range, temp_range, indexing="ij")

        self.gravs_mesh, self.temps_mesh = np.meshgrid(self.unique_gravs, self.unique_temps, indexing="ij")

        self.fig, self.axs = plt.subplots(1, 3, figsize=(17,6))

        grav_range = np.linspace(self.unique_gravs.min(), self.unique_gravs.max(), 100)
        temp_range = np.linspace(self.unique_temps.min(), self.unique_temps.max(), 100)
        grv_grid, tmp_grid = np.meshgrid(grav_range, temp_range)#, indexing="ij")

        x_range = np.array([grv_grid.flatten(), tmp_grid.flatten()]).T
        for i, (colour, label) in enumerate(zip(["blue", "red", "green"], self.unique_abuns)):
            X_train = self.t_grid[i].flatten().reshape(-1, 2)

            xnans = np.argwhere(~np.isnan(X_train[:,0])).flatten()

            X_train = X_train[xnans]
            y_train = self.chisq_grid[i].flatten()[xnans]

            chisq_interpd2 = spinterp.griddata(X_train, y_train, x_range, method="cubic")

            grid_nan = np.argwhere(~np.isnan(chisq_interpd2.reshape(100,100)))

            self.axs[i].errorbar(self.best_ts[i][:,0], self.best_ts[i][:,1],yerr=self.best_ts[i][:,2],
                                 fmt="o", c="k", ls="")
            surf = self.axs[i].pcolor(grav_range, temp_range, chisq_interpd2.reshape(100,100), cmap="gist_yarg")
            cb = self.fig.colorbar(surf, ax=self.axs[i])
            cb.ax.tick_params(labelsize=18)
            if i == 2:
                cb.set_label(label=r"$X^2$ (CIII $\lambda$1176)", size=20)
            self.axs[i].set_xlabel(r"log($g$/cm s$^{-2}$)", size=20)
            self.axs[i].set_title(rf"[C] = {str(label)} C$_\odot$", size=20, y=1.02)
            self.axs[i].tick_params(which="major", direction="inout", top=True, right=True, length=8, labelsize=18)

        self.axs[0].set_ylabel(r"T$_{\rm eff}$ (kK)", size=20)

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.tight_layout()
        plt.show()



    def onclick(self, event):
        """
        Register clicks in the plot.
        Left click on an axis = plot the chi-squared curves of the log(g) 
        closest to the mouse x position.
        """

        toolbar = plt.get_current_fig_manager().toolbar
        if event.button == 1 and toolbar.mode == "" and event.key != "shift" and type(event.xdata) != type(None):
            self.get_grav(event.xdata)

    def get_grav(self, grav):
        
        grav = float(grav)
        nearest_grav = self.unique_gravs[np.argmin(np.abs(self.unique_gravs - grav))]
        # Find models at this gravity
        model_idxs = [str(nearest_grav) in name for name in self.model_names]

        chisq_idx = np.argmin(np.abs(self.unique_gravs - grav))

        # get chisqs of these models
        model_chisqs = self.chisq_grid[:,chisq_idx]

        self.chisq_fig = plt.figure(figsize=(8,8))
        self.chisq_ax = self.chisq_fig.add_subplot(111)

        # styles for each abundance.
        markerstyles = ["o", "s", "^"]
        linestyles = ["-", "--", ":"]
        colours = ["b", "r", "g"]

        handles = []
        labels = []
        for i, (chisq, ms, ls, abun, col) in enumerate(zip(model_chisqs,
                                                      markerstyles,
                                                      linestyles,
                                                      self.unique_abuns,
                                                      colours)):
            nans = np.argwhere(~np.isnan(chisq)).flatten()
            chisq = chisq[nans]
            temps = self.unique_temps[nans]
            trange_to_plot =  np.linspace(temps.min(), temps.max(), 100)
            min_chisq_idx = np.nanargmin(chisq)
            if min_chisq_idx not in [0,len(chisq)-1]:
                if min_chisq_idx == 1:
                    idx_left = 1
                    idx_right = 3
                elif min_chisq_idx == len(chisq) - 1:
                    idx_left = 2
                    idx_right = 1
                else:
                    idx_left, idx_right = 2, 2
            elif min_chisq_idx == 0:
                idx_left = 0
                idx_right = 2
            elif min_chisq_idx == len(chisq) - 1:
                idx_left = 2
                idx_right = 0
            try:
                # Fit parabola to idx_left points to the left of the minimum
                # value, and idx_right points to the right of the minimmum.
                coefs_of_parab = np.polyfit(temps[min_chisq_idx-idx_left : min_chisq_idx+idx_right],
                                            chisq[min_chisq_idx-idx_left : min_chisq_idx+idx_right],
                                            2)
                # Set up grid of values to interpolate parabola to.
                t_grid = np.linspace(temps[min_chisq_idx-idx_left], temps[min_chisq_idx+idx_right], 20000)
                # Now interpolate this parabola onto the grid.
                interped_parabola = np.poly1d(coefs_of_parab)
                # And get corresponding chisqs of the grid of params.
                chisqs_interps = interped_parabola(t_grid)
                # Best fit minimum chisq is min of this interpolated parabola.
                minchisq = chisqs_interps.min()
                # And find parameter corresponding to this chisq.
                best_fit_param = t_grid[np.argmin(chisqs_interps)]

                # Need to interpolate both sides of parabola seperately,
                # since there are two values where change in chisq is 1.

                # First interpolate over the grid for values > best_fit_param
                # and find upper interval bound.
                chisq_interp_upper = spinterp.interp1d(chisqs_interps[t_grid > best_fit_param],
                                                    t_grid[t_grid > best_fit_param])
                param_upper_1sig = chisq_interp_upper(minchisq+1)
                
                # Interpolate for values <= best_fit_param to find lower
                # interval bound.
                chisq_interp_lower = spinterp.interp1d(chisqs_interps[t_grid <= best_fit_param],
                                                    t_grid[t_grid <= best_fit_param])
                param_lower_1sig = chisq_interp_lower(minchisq+1)

                labels.append(str(abun) + r" C$_{\odot}$, T$_{\rm eff}$ = (%.2f $\pm$ %.2f) kK" % (best_fit_param, best_fit_param-param_lower_1sig))
                handles.append(lines.Line2D([],[], marker=ms, ls=ls, c=col,ms=7))
                l = self.chisq_ax.plot(trange_to_plot, interped_parabola(trange_to_plot), ls=ls, c=col)
                self.chisq_ax.axvline(param_upper_1sig, alpha=0.8, ls="--", c=col)
                self.chisq_ax.axvline(param_lower_1sig, alpha=0.8, ls="--", c=col)

                self.chisq_ax.scatter(best_fit_param, minchisq, c=col, marker="+")
                self.chisq_ax.scatter(temps, chisq, marker=ms, color=col)
            except:
                
                self.chisq_ax.scatter(temps, chisq, marker=ms, color=col)

        self.chisq_ax.set_xlabel(r"T$_{\rm eff}$ (kK)", size=15)
        self.chisq_ax.set_ylabel(r"$X^2$", size=15)
        self.chisq_ax.legend(handles, labels)

        self.chisq_ax.set_title(r"log($g$ / cm s$^{-2}$) = "+str(nearest_grav), size=15)
        plt.show(block=True)

    def print_opt_params(self):
        """Calculate and print out optmial parameters"""
        print("%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-13s %-10s" % ("C/Csun", "log(g)", "R/Rsun", "dR/Rsun", "M/Msun", "dM/Msun", "Teff(K)", "dTeff(K)", "log(L/Lsun)", "dLogL/Lsun"))
        self.best_ts = np.zeros((self.no_abuns, self.no_gravs, 3)) # log(g), teff, and dteff for each [C]-log(g) pair
        dirs = []
        for i in range(self.no_abuns):
            # Current abundance.
            ab = str(self.unique_abuns[i])
            for j in range(self.no_gravs):
                # Current log(g)
                gr = str(self.unique_gravs[j])
                # for k in range(self.no_temps):
                #     # Current Teff.
                #     te = str(self.unique_temps[k])
                chisq = self.chisq_grid[i][j]
                nans = np.argwhere(~np.isnan(chisq)).flatten()
                chisq = chisq[nans]
                temps = self.unique_temps[nans]
                trange_to_plot =  np.linspace(temps.min(), temps.max(), 100)
                # try:
                min_chisq_idx = np.nanargmin(chisq)
                if min_chisq_idx not in [0,len(chisq)-1]:
                    if min_chisq_idx == 1:
                        idx_left = 1
                        idx_right = 3
                    elif min_chisq_idx == len(chisq) - 1:
                        idx_left = 2
                        idx_right = 1
                    else:
                        idx_left, idx_right = 2, 2
                elif min_chisq_idx == 0:
                    idx_left = 0
                    idx_right = 2
                elif min_chisq_idx == len(chisq) - 1:
                    idx_left = 2
                    idx_right = 0
                # try:
                # Fit parabola to idx_left points to the left of the minimum
                # value, and idx_right points to the right of the minimmum.
                coefs_of_parab = np.polyfit(temps[min_chisq_idx-idx_left : min_chisq_idx+idx_right],
                                            chisq[min_chisq_idx-idx_left : min_chisq_idx+idx_right],
                                            2)

                # Set up grid of values to interpolate parabola to.
                t_grid = np.linspace(temps[min_chisq_idx-idx_left], temps[min_chisq_idx+idx_right], 20000)
                # Now interpolate this parabola onto the grid.
                interped_parabola = np.poly1d(coefs_of_parab)
                # And get corresponding chisqs of the grid of params.
                chisqs_interps = interped_parabola(t_grid)
                # Best fit minimum chisq is min of this interpolated parabola.
                minchisq = chisqs_interps.min()
                # And find parameter corresponding to this chisq.
                best_fit_param = t_grid[np.argmin(chisqs_interps)]
                # self.best
                wl = 8100.44
                zero_point = 1.1132e-9
                mag = 21.929

                # Convert wavelength to cm.
                wl_cm = wl*1e-8 # 1cm / 1e8 angstrom.

                # Distance to Leo P.
                distance_cm = 1.62 * 3.086e24  # 3.086e24 cm / 1 Mpc
                distance_cm_uncertainty = 0.15 * 3.086e24
                # Flux at this wavelength from magnitude, using magnitude
                # formula in Vega system.
                # Multiply by 1e8 to convert units from
                # erg/s/cm2/Ang -> erg/s/cm2/cm (since 1e8 A/cm).
                f_lambda = zero_point * (10**(-0.4*mag)) * 1e8 # erg/s/cm2/cm

                # Constrain with Rayleigh jeans approx of blackbody.
                # F = F0 * 10 ^{-0.4m} = (R/d)^2 * pi * B_nu(T_eff)
                prefactors = np.sqrt((wl_cm**4 * f_lambda)/(2*np.pi*2.998e10*1.381e-16))

                radius_cm = prefactors * distance_cm* (best_fit_param*1e3)**(-0.5) # np.sqrt((wl_cm**4  * distance_cm**2 * f_lambda)/(2*np.pi*2.998e10*1.381e-16*best_fit_param*1e3))
                radius_solar = radius_cm / 6.957e10 # 1 Rsun / 6.957e10 cm
                m_cgs = 10**self.unique_gravs[j] * radius_cm**2 / (6.674e-8)
                m_solar = m_cgs / 1.989e33
                # Bolometric luminosity using this (Teff, R) pair.
                L_cgs = 4*np.pi*(radius_cm**2)*5.67037441e-5*((best_fit_param*1e3)**4)
                L_solar = L_cgs / 3.828e33

                logL_solar = np.log10(L_solar) # 1Lsun / 3.828e33 erg/s

                log_g_km = 10**(self.unique_gravs[j]) * 1e-5 # 1km/s / 1e5cm/s

                # Need to interpolate both sides of parabola seperately,
                # since there are two values where change in chisq is 1.

                # First interpolate over the grid for values > best_fit_param
                # and find upper interval bound.
                chisq_interp_upper = spinterp.interp1d(chisqs_interps[t_grid > best_fit_param],
                                                    t_grid[t_grid > best_fit_param])
                # 1 dof: getting optimal T for each grav and each abundance.
                delta_chisq = sps.chi2.isf(2*sps.norm.sf(1),df=2)
                param_upper_1sig = chisq_interp_upper(minchisq+1) - best_fit_param

                dldr=8*np.pi*radius_cm*5.67037441e-5*((best_fit_param*1e3)**4)
                dldt=16*np.pi*(radius_cm**2)*5.67037441e-5*((best_fit_param*1e3)**3)
                drdt = -0.5*prefactors*distance_cm*((best_fit_param*1e3)**(-1.5))
                drdd = prefactors*((best_fit_param*1e3)**(-0.5))

                sigmad = distance_cm_uncertainty
                sigmaT =param_upper_1sig

                # Define radius uncertainty in the mean time, will be used later (the square root in sigmaL1)
                sigmaR = np.sqrt((drdt*sigmaT)**2 + (drdd*sigmad)**2)
                sigmaR_solar = sigmaR / 6.957e10

                sigmaL1 = dldr * sigmaR
                sigmaL2 = dldt * sigmaT
                dL_cgs = np.sqrt((sigmaL1)**2 + (sigmaL2)**2)

                # Now define mass uncertainty
                sigmaM_solar = abs(2*(10**self.unique_gravs[j]) * radius_cm *sigmaR/ (6.674e-8*3.828e33))
                dL_solar = dL_cgs / 3.828e33

                dirs.append(f"mdotupper-F814W-{ab}_{gr}-{(best_fit_param):.4f}-{logL_solar:.4f}")
                self.best_ts[i][j] = [gr, best_fit_param, sigmaT]
                print("%-10s %-10s %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-13.4f %-10.3f" % (ab, gr, radius_solar, sigmaR_solar, m_solar, sigmaM_solar, best_fit_param*1e3, param_upper_1sig*1e3, logL_solar, dL_solar/(np.log(10) * L_solar)))

    def plot_3d(self):
        if isinstance(self.fig, type(None)):
            self.fig = plt.figure(figsize=(7,7))
            self.ax = self.fig.add_subplot(111, projection="3d")
        handles = []
        
        for i, (colour, label) in enumerate(zip(["blue", "red", "green"], self.unique_abuns)):
            self.ax.plot_surface(self.gravs_mesh, self.temps_mesh, self.chisq_grid[i],
                                 color=colour, alpha=0.4)
            handles.append(mpatches.Patch(color=colour, label=str(label), alpha=1))


        self.ax.set_xlabel("log g")
        self.ax.set_ylabel("Teff (kK)")
        self.ax.set_zlabel("X^2")

        self.fig.legend(handles=handles)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()


if __name__ == "__main__":
    path_to_trgrid = "/tggrid/trgrid-F814W-"
    # Entire TR Grid.
    model_lines_tr, model_names_tr, line_names_tr = load_models(path_to_trgrid, indices_of_number=[5, 9], using_trgrid=True)

    data_lines1, uv_line_errs1, data_names1 = get_renorm_lines()
    p = Chisq3D(data_lines1, uv_line_errs1, model_lines_tr, model_names_tr, line_names_tr, ["CIII1176"], data_names=data_names1)
