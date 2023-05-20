"""
Script that reads in and plots the CIII1176
and CIV1550 lines from FUV speectrum of XMP LP26.

Also reads in FASTWIND models and calculates
upper limits to the mass loss rate.

Some documentation missing.

Process of reading in data is very hard coded,
in need of refactoring!

author: Ciarán Furey
"""

import os # for accessing files
import re # for editing name of spectral line

from glob import glob # function for efficient file searching
from matplotlib.ticker import AutoMinorLocator
from matplotlib.legend_handler import HandlerTuple
from matplotlib import lines

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spinterp


from param_grid import vink_mdot, krticka_mdot, bjorklund_mdot
from broaden import broaden

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": "Helvetica",
# })

D_LAMBDA = 60 # Wavelength range of line  to plot.
VSINI = 370 # [km/s]
RESOLUTION = 6000 # Resolution of broadened model line.

NO_LINES = 2

V_RAD = 283.0 # Radial velocity of Leo P, in [km/s]

# Directories where models are stored.
# (Hard coded, I know... might get around to optimising at some point...)
path_to_trgrid_01_3p42 = "/tggrid/trgrid-F814W-0.01_3.42-"
path_to_trgrid_01_3p67 = "/tggrid/trgrid-F814W-0.01_3.67-"
path_to_trgrid_01_3p92 = "/tggrid/trgrid-F814W-0.01_3.92-"
path_to_trgrid_01_4p17 = "/tggrid/trgrid-F814W-0.01_4.17-"
path_to_trgrid_01_4p42 = "/tggrid/trgrid-F814W-0.01_4.42-"
path_to_trgrid_03_3p42 = "/tggrid/trgrid-F814W-0.03_3.42-"
path_to_trgrid_03_3p67 = "/tggrid/trgrid-F814W-0.03_3.67-"
path_to_trgrid_03_3p92 = "/tggrid/trgrid-F814W-0.03_3.92-"
path_to_trgrid_03_4p17 = "/tggrid/trgrid-F814W-0.03_4.17-"
path_to_trgrid_03_4p42 = "/tggrid/trgrid-F814W-0.03_4.42-"
path_to_trgrid_05_3p42 = "/tggrid/trgrid-F814W-0.05_3.42-"
path_to_trgrid_05_3p67 = "/tggrid/trgrid-F814W-0.05_3.67-"
path_to_trgrid_05_3p92 = "/tggrid/trgrid-F814W-0.05_3.92-"
path_to_trgrid_05_4p17 = "/tggrid/trgrid-F814W-0.05_4.17-"
path_to_trgrid_05_4p42 = "/tggrid/trgrid-F814W-0.05_4.42-"

path_to_mdotgrid_01_3p42 = "/mdot_upper/mdotupper-F814W-0.01_3.42-"
path_to_mdotgrid_01_3p67 = "/mdot_upper/mdotupper-F814W-0.01_3.67-"
path_to_mdotgrid_01_3p92 = "/mdot_upper/mdotupper-F814W-0.01_3.92-"
path_to_mdotgrid_01_4p17 = "/mdot_upper/mdotupper-F814W-0.01_4.17-"
path_to_mdotgrid_01_4p42 = "/mdot_upper/mdotupper-F814W-0.01_4.42-"

path_to_mdotgrid_03_3p42 = "/mdot_upper/mdotupper-F814W-0.03_3.42-"
path_to_mdotgrid_03_3p67 = "/mdot_upper/mdotupper-F814W-0.03_3.67-"
path_to_mdotgrid_03_3p92 = "/mdot_upper/mdotupper-F814W-0.03_3.92-"
path_to_mdotgrid_03_4p17 = "/mdot_upper/mdotupper-F814W-0.03_4.17-"
path_to_mdotgrid_03_4p42 = "/mdot_upper/mdotupper-F814W-0.03_4.42-"

path_to_mdotgrid_05_3p42 = "/mdot_upper/mdotupper-F814W-0.05_3.42-"
path_to_mdotgrid_05_3p67 = "/mdot_upper/mdotupper-F814W-0.05_3.67-"
path_to_mdotgrid_05_3p92 = "/mdot_upper/mdotupper-F814W-0.05_3.92-"
path_to_mdotgrid_05_4p17 = "/mdot_upper/mdotupper-F814W-0.05_4.17-"
path_to_mdotgrid_05_4p42 = "/mdot_upper/mdotupper-F814W-0.05_4.42-"


def load_models(path, append_to_name="", up_to_index=None, indices_of_number=[1, -1],
                number_of_lines=NO_LINES, using_trgrid=True):
    """Load models from FASTWIND directory, given by "path". Returns
    the broadened lines for each model, the model names, and the line
    names.

    Tailored for when a grid of parameters are being modoelled, for
    example, mass loss rate or Teff, where the value of that
    parameter is given in the name of the model directory, as was done
    in this project.

    The name of the directory should be in the form:
        "{quantity_being_modelled}_{value_of_quantity}"
    where they have been separated by one underscore.

    For example, when determining the upper limits of the mass loss
    rates my directory names were:
        mdotupper-F814W-0.05_4.42-
    where the name is showing:
     - that it is determining the upper limit of the mass loss rate;
     - that I used photometry in the F814W band;
     - the carbon abundance (compared to solar)
     - log(g)

    All directories with this path are then loaded. An example directory
    with this path is
    mdotupper-F814W-0.05_4.42-41.2727-5.0085-1.0E-9
    where the final three values show the effective temperature, the
    log luminosity, and the mass loss rate of the model.

    Note where the underscore is here.

    glob does not return the model directories in numerical order,
    and this must be accounted for, since it can lead to unclear
    results when plotting. To resolve this, the directory name is split
    at the underscore, and then, to locate the numerical value of the
    model (e.g mass loss rate, or temperature), the two values in
    indices_of_number represent the first index of the number in
    {value_of_quantitiy}, and the last index, respectively.

    These are then used to sort the directory names in numerical order
    according to their name.

    If many directories are created, but not all of them have been
    modelled yet, the directories that have already been modelled can
    be accessed (provided they have been modelled in numerical order)
    with the up_to_index parameter, which will load the numerically
    sorted directory names up to index "up_to_index."

    Parameters
    ----------
    path : str
        Name of path to model directory. Should be in the form
            "path/to/fastwind/{quantity_being_modelled}_" or
            "path/to/fastwind/{quantity_being_modelled}"
        (see above for more details.)
    append_to_name : str
        Name to add to the "{value_of_quantity}" part of the model name
        (see above.) The default is "".
    up_to_index : int or None
        The number of models to load (see above for more detailed
        explanation.) The default is None (i.e. all models in directroy
        are loaded)
    indices_of_number : array-like, length 2
        The first and last indices of the numerical value in
        {value_of_quantitiy} (see above for more details.)

    Returns
    -------
    model_lines : 3D list
        3D array containing lines of each model.
        model_lines[i][j] corresponds to model i in the FASTWIND model
        directory and line j of model i, and model_lines[i][j][0] and
        model_lines[i][j][1] are the wavelengths and normalised flux
        of line j, respectively.
        Note that this is not a numpy array, since the lines contain
        different numbers of data points.
    model_names : list
        The names of each model, which are:
        {value_of_quantity} + append_to_name
    line_names : list
        The names of the lines being modelled. They are the same for
        each model.

    """

    # Get model folders in FASTWIND directory.
    model_folders = glob(f"{os.getcwd()}{path}*")

    # Store the model names, and numerical value, for sorting.
    # Since glob returns in arbitrary order.
    model_names = []
    sorting_nos = [] # list to store values to sort.
    # Also need array of the indices of which directories have lines
    # computed.
    available_indices = np.array([], dtype=int)

    for i, folder in enumerate(model_folders):
        # check if pformalsol didnt work for some reason
        if len(glob(f"{folder}/OUT.*")) < number_of_lines:
            pass
        else:
            model_name = folder.split("_")[-1] # Split at 3rd underscore

            sorting_no = model_name[indices_of_number[0]:indices_of_number[1]]

            sorting_nos.append(float(sorting_no)) # Store number for sorting.
            available_indices = np.append(available_indices, i)
            if using_trgrid:
                model_name = folder.split("_")[-2][-4:]+"-"+model_name

            model_names.append(model_name + append_to_name) # Store spectral type.

    sorted_nos = np.argsort(sorting_nos)
    sorted_idxs = available_indices[sorted_nos]

    # Now sort in increasing number.
    if up_to_index is None:
        model_names1 = np.array(model_names)[sorted_nos]
        model_folders1 = np.array(model_folders)[sorted_idxs]
    else:
        model_names1 = np.array(model_names)[sorted_nos][:up_to_index]
        model_folders1 = np.array(model_folders)[sorted_idxs][:up_to_index]
    # print(model_folders1, model_names1)
    # List to store every line of every model.
    # Not a numpy array since each line has different number of points.
    model_lines = []
    line_names = []

    # Now set a counter to find when first usable model appears in
    # the list of model directories. This will only matter if the first
    # model is not usable.
    first_usable_model = 0

    # Go through each model up to given index.
    for i, model in enumerate(model_folders1):
        line_files = glob(f"{model}/OUT.*") # Get the line files
        # if i == 0:
        #     print(line_files)
        if len(line_files) < number_of_lines:
            first_usable_model += 1
            pass
        else:
            array_for_lines = []
            for line_file in line_files:
                # if it is the first model, only need to store names
                # of line once.
                if first_usable_model == i:
                    # print(line_file.split("_"))
                    line_name = (line_file.split("_")[-2]).split(".")[-1]
                    if line_name == "HBETA":
                        line_name = line_name + "4860"
                    elif line_name == "HALPHA":
                        line_name = line_name + "6565"

                    line_names.append(line_name)

                    if float(line_name[-4:]) > 2000:
                        res = 2000 # MUSE resolution
                    else:
                        res = 6000 # COS resolution
                # Read model line.
                wave, flux = np.genfromtxt(line_file, usecols=[2,4], max_rows=161).T
                # Broaden line.
                wave_broad, flux_broad = broaden(wave, flux, VSINI, res)
                # Store.
                array_for_lines.append((wave_broad, flux_broad))
            model_lines.append(array_for_lines)

    return model_lines, model_names1, line_names


def get_line(norm_spec, lambda_min, lambda_max):
    """Function to get line from the normalised data, norm_spec,
    centred at lambda_cent and plotting in the range
    (lambda_min, lambda_max).

    Parameters
    ----------
    norm_spec : np.ndarray
        The normalised spectrum.
    lambda_min : int/float
        Minimum wavelength of range.
    lambda_max : int/float
        Maximum wavelength of range.

    Returns
    -------
    [wls_out, flx_out] : 2D list
        Wavelengths (index 0) and normalised flux (index 1)
        of line.

    """
    # Read wavelengths (wls) and fluxes.
    wls = norm_spec[:,0]
    fluxes = norm_spec[:,1]

    # Define range over which to plot the line.
    condition = (wls > lambda_min) & (wls < lambda_max)

    # Get spectrum within this range.
    wls_out = wls[condition]
    flx_out = fluxes[condition]

    return [wls_out, flx_out]

def get_renorm_lines(line_dir=None):
    """Function that obtains renormalised lines from
    the line_dir directory"""
    if line_dir is None:
        line_dir=os.getcwd()

    lines = glob(f"{line_dir}/line_clipnorm_*")
    line_names = []
    line_errs = []
    line_fluxs = []
    line_out = []

    for line in lines:
        name = line.split("_")[-1][:-4] # up to .dat
        # print(name)
        line_names.append(name)
        file = np.loadtxt(line)
        # print(file)
        line_out.append([file[:,0], file[:,1]])
        # print(file[:,0])
        # line_fluxs.append(file[:,1])
        if file.shape[1] == 3:
            line_errs.append(file[:,2])
        else:
            line_errs.append(np.full_like(line_fluxs, 0.05))

    return line_out, line_errs, line_names

def plot_models(lines_in_data, models_, model_names_, line_names,
                models_to_plot=[], lines_to_use=None, using_renorm=False,
                data_names=None):
    """Plot each line, and FASTWIND models of each line.

    Parameters
    ----------
    lines_in_data : list
        The lines from the spectrum.
    models_ : 3D list
        FASTWIND model lines ("model_lines" output from load_models()).
    model_names_ : list
        Names of fastwind models ("model_names" output from
        load_models()).
    line_names : list
        Names of lines ("line_names" output from load_models()).
    models_to_plot : list
        Which models to plot, should be a subsection of model_names.
        The default is []; i.e,, all model are plotted.

    Returns
    -------
    fig, axarr : matplotlib.figure.Figure, np.ndarray of matplotlib.axes._subplots.AxesSubplot
        FIgure and axes of the plot.

    """
    # Find models to be plotted, if models_to_plot input.
    if len(models_to_plot) != 0:
        indices_to_plot = np.where(np.in1d(model_names_, models_to_plot))
        model_lines_ = np.array(models_)[indices_to_plot]
        model_names_to_plot = np.array(model_names_)[indices_to_plot]
    else: # or else just plot all of the lines
        model_lines_ = models_
        model_names_to_plot = model_names_

    # FInd lines to plot
    if lines_to_use is not None:
        if type(lines_to_use) is not list: # if only one line is to be used
            lines_to_use = [lines_to_use]
        if not using_renorm:
            # # Indices of desired lines.
            indices = np.where(np.in1d(line_names, lines_to_use))[0]
            model_indices = indices
        else:
            # finding lines to be used, from directory
            indices = np.where(np.in1d(data_names, lines_to_use))[0]
            # finding lines to be used within each model
            model_indices = np.where(np.in1d(line_names, lines_to_use))[0]

        # Get lines from data array.
            line_names = data_names
        lines_in_data = np.array(lines_in_data)[indices]

    else:
        indices = range(len(line_names))
        model_indices = range(len(line_names))

    nlines = len(lines_in_data)

    if nlines <= 2:
        n_rows = 1
        n_cols = nlines
    else:
        n_rows = 2
        n_cols = nlines // n_rows + int((nlines % n_rows) != 0)

    fig, axarr = plt.subplots(n_rows, n_cols, figsize=(16,10))#,gridspec_kw = {"wspace": 0.3, "hspace":0.2})
    axarr = axarr.flatten()

    # Lists for handles and labels for legend.
    legend_labels = []
    legend_handles = []
    colours = []

    for i, data_line in enumerate(lines_in_data): # go thorugh each line in data.
        # plot data line.
        axarr[i].plot(data_line[0], data_line[1], "k-", lw=0.5, alpha=0.9)
        axarr[i].text(0.0, 1.01, line_names[indices[i]], transform=axarr[i].transAxes, ha="left", va="bottom")
        axarr[i].set_xlim(data_line[0][0], data_line[0][-1])

        colours = [] # List to store colours, so can change linestyles.

        # Lists to store the min and max fluxes of model line,
        # used for setting y limits.
        min_model_fluxes = []
        max_model_fluxes = []

        # Now go through each model of this line.
        for j, model in enumerate(model_lines_):
            # Plot the model line.
            plotted_line, = axarr[i].plot(model[model_indices[i]][0],
                                          model[model_indices[i]][1], alpha=0.8)
            min_model_fluxes.append(model[model_indices[i]][1].min())
            max_model_fluxes.append(model[model_indices[i]][1].max())

            # Find colour of model line.
            current_colour = plotted_line.get_color()

            if current_colour in colours:
                # Change linestyle of line if colour already in use.
                plotted_line.set_linestyle("--")

            colours.append(current_colour)

            if i == 0: # Store handles and labels for legend, only once.
                legend_labels.append(model_names_to_plot[j])
                legend_handles.append(plotted_line)

        axarr[i].set_ylim(min(min_model_fluxes)-0.1, max(max_model_fluxes)+0.1)
    axarr[n_cols-1].legend(legend_handles, legend_labels, loc=(0,0),
                           bbox_to_anchor=(1.05, 0.6),
                           bbox_transform=axarr[n_cols-1].transAxes)
    plt.tight_layout()
    plt.show()

    return fig, axarr

def plot_lines(data, errs, names, lines_to_use=None, using_renorm=True):
    """Plot the data lines."""
    # FInd lines to plot
    if lines_to_use is not None:
        if type(lines_to_use) is not list: # if only one line is to be used
            lines_to_use = [lines_to_use]
        if not using_renorm:
            # # Indices of desired lines.
            indices = np.where(np.in1d(names, lines_to_use))[0]
        else:
            # finding lines to be used, from directory
            indices = np.where(np.in1d(names, lines_to_use))[0]


        # Get lines from data array.
            line_names = names
        lines_in_data = np.array(data)[indices]
        errs_in_data = np.array(errs)[indices]

    else:
        indices = range(len(names))
        model_indices = range(len(names))

    nlines = len(lines_in_data)

    if nlines <= 2:
        n_rows = 1
        n_cols = nlines
    else:
        n_rows = 2
        n_cols = nlines // n_rows + int((nlines % n_rows) != 0)

    fig, axarr = plt.subplots(n_rows, n_cols, figsize=(10,5))#,gridspec_kw = {"wspace": 0.3, "hspace":0.2})
    axarr = axarr.flatten()

    for i, data_line in enumerate(lines_in_data): # go thorugh each line in data.
        # plot data line.
        axarr[i].plot(data_line[0], data_line[1], "k-", alpha=1)
        axarr[i].fill_between(data_line[0], data_line[1] - errs_in_data[i], data_line[1] + errs_in_data[i],
                              color="gray", alpha=0.3)
        # axarr[i].text(0.0, 1.01, line_names[indices[i]], transform=axarr[i].transAxes, ha="left", va="bottom",
        #               size=15)
        line_name = line_names[indices[i]]
        if line_name not in ["HALPHA", "HBETA"]:
            wl = re.findall(r'\d+', line_name)[0] # Find wavelength of transition
            transition = line_name.split(wl)[0] # Find ionisation of transition
            to_plot = transition+r" $\lambda$"+wl
        else:
            to_plot = line_name

        axarr[i].set_title(to_plot, size=15)
        axarr[i].set_xlabel(r"Wavelength (Å)", size=15)
        axarr[i].set_ylabel("Normalised Flux", size=15)
        axarr[i].tick_params(which="major", direction="inout", top=True, right=True, length=4, labelsize=12)
        axarr[i].set_xlim(data_line[0][0], data_line[0][-1])

    # plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

def round_to_2(x, dx):
    """Function that returns x to 2 significant digits.
    Used for determining uncertainties, and for printing
    values.

    ----------
    x : float
        Input number
    dx : float
        x uncertainty.

    Returns
    -------
    x_same_as_dx, dx_2sig_figs : tuple of float
        dx to 2 significant figures, and x at the
        same number of decimal points as this.

    """
    dx_2sig_figs = "%s" % float("%.2g" % dx)
    digits_after_point = len(dx_2sig_figs.split(".")[1])
    # Now see if, to 2 sigfigs, dx is 2 decimal points, but
    # the final digit is a 0.
    if (str(dx)[0] == "0") and (digits_after_point == 1):
        digits_after_point  += 1
        dx_2sig_figs += "0"

    x_same_as_dx = str(round(x,digits_after_point))

    if (len(x_same_as_dx.split(".")[1]) == 1) and (digits_after_point == 2):
        x_same_as_dx += "0"

    return x_same_as_dx, dx_2sig_figs

def chi_sq(lines_in_data, data_errors, models, model_names, model_line_names, lines_to_use,
           fit=None, xdata=None, xlabel=None, return_estimates=False, idx_left=2, idx_right=2,
           using_renorm=False, data_names=None, savename=None, show=True):
    """Perform chi-sqaured minimisation to obtain an MLE with 1 sigma
    confidence intervals (CIs), or else to obtain an upper limit of a
    quantity.

    only use using_renorm and data_names when using renormalised line
    """

    if type(lines_to_use) is not list: # if only one line is to be used
        lines_to_use = [lines_to_use]
    if not using_renorm:
        # # Indices of desired lines.
        indices = np.where(np.in1d(model_line_names, lines_to_use))[0]
        model_indices = indices
    else:
        indices = np.where(np.in1d(data_names, lines_to_use))[0]
        model_indices = np.where(np.in1d(model_line_names, lines_to_use))[0]

    # Get lines and errors from data.
    data_lines_to_use = np.array(lines_in_data)[indices]
    data_errs_to_use = []
    for i in indices:
        data_errs_to_use.append(data_errors[i])

    xlabel_split_for_printing = xlabel.split("(")
    quantity = xlabel_split_for_printing[0][:-1]
    unit = xlabel_split_for_printing[1][:-1]

    if show: # Close any figures opened before.
        plt.close("all")
    # Build figure and gridspec, where the data will be plotted on the
    # right, and chisq surface will be plotted on the left.
    chisq_fig = plt.figure(figsize=(10,5))

    chisq_gs = chisq_fig.add_gridspec(len(lines_to_use), 2, hspace=0.15, wspace=0.2, top=0.9, left=0.08, right=0.86, bottom=0.11)

    colours = [] # List to store colours.

    chi_sq_for_each_line = np.zeros((len(lines_to_use), len(models)))


    # Loop through the lines to calculate chisq for.
    for i, line in enumerate(data_lines_to_use):
        # Wavelengths of the line in the spectrum to interpolate model
        # lines to.
        data_wl = line[0]
        data_flux = np.array(line[1])

        # Build axis for line i.
        line_ax = chisq_fig.add_subplot(chisq_gs[i, 1])

        for j, model in enumerate(models):
            # Get the model lines.
            model_line_not_interpolated = np.array(model)[model_indices[i]]
            # Interpolate function to interp model flux to data
            # wavelength.
            line_interp_func = spinterp.interp1d(model_line_not_interpolated[0],
                                                 model_line_not_interpolated[1],
                                                 fill_value="extrapolate")
            # Now get model fluxes at data wavelengths.
            line_interp = line_interp_func(data_wl)

            # Calculate the chisq stat
            chisq = np.sum(((data_flux - line_interp)/data_errs_to_use[i])**2)

            # Store the chisq for line i of model j.
            chi_sq_for_each_line[i][j] = chisq

            if j == 0: # Only plot data once.
                line_ax.plot(data_wl, data_flux, "k") # Plot observed flux.
                # Plot shaded are representing uncertainties.
                line_ax.fill_between(data_wl,
                                     data_flux + data_errs_to_use[i],
                                     data_flux - data_errs_to_use[i],
                                     alpha=0.2, facecolor="grey",
                                     edgecolor="grey")

                # Modify line name for showing on plot, using re.

                line_name = model_line_names[model_indices[i]]
                if line_name not in ["HALPHA", "HBETA"]:
                    wl = re.findall(r'\d+', line_name)[0] # Find wavelength of transition
                    transition = line_name.split(wl)[0] # Find ionisation of transition
                    to_plot = transition+r" $\lambda$"+wl
                else:
                    to_plot = line_name
                line_ax.tick_params(axis="both", which="major", top=True, right=True, direction="inout", length=8, labelsize=11)
                line_ax.text(0.95, 0.05, to_plot, transform=line_ax.transAxes, size=16,
                             ha="right", va="bottom")

            # Reformat exponential into nice output.
            if "E" in model_names[i]:
                label_initial=f"{xdata[j]:.1e}" # label with exponential as "e"
                label_split = label_initial.split("e")
                label = label_split[0] + fr"$\times 10^{{{int(label_split[1])}}}$"
            else:
                label=f"{xdata[j]}"
            plotted_line, = line_ax.plot(data_wl, line_interp, label=label, alpha=0.7) # plot model.

            current_colour = plotted_line.get_color()

            # If plotted so many models that colour repeats, change the
            # linestyle to differentiate between models.
            if current_colour in colours:
                plotted_line.set_linestyle("--")

            colours.append(current_colour)

            line_ax.set_xlim(data_wl[0], data_wl[-1])

            if i == len(lines_to_use) - 1: # If at the bottom, set xlabel.
                line_ax.set_xlabel(r"Wavelength (Å)", size=15)
            if i == 0: # If at the top, put in legend.
                line_ax.legend(loc=1,bbox_to_anchor=(1.35, 1), bbox_transform=line_ax.transAxes,
                               title=f"{quantity} ({unit})")
            # Reset colour array once all models have been plotted.
            if j == len(models) - 1:
                colours = []

    # Now calculate total chisq for each line.
    chisqs = np.sum(chi_sq_for_each_line, axis=0)

    # Plot the chisq surface, and, if desired, calculate CIs or limits,
    # where the methods differ for each.

    chisq_ax = chisq_fig.add_subplot(chisq_gs[:, 0])
    chisq_ax1 = chisq_ax.twinx()


    if fit is not None:
        # First find lowest chisq of models.
        min_chisq_model = np.argmin(chisqs)

        # If CIs are being calculated:
        if fit=="CIs":
            # Define a linspace over values of all models to plot parabola.
            xrange_to_plot = np.linspace(xdata.min(), xdata.max(), 100)

            # Fit parabola to idx_left points to the left of the minimum
            # value, and idx_right points to the right of the minimmum.
            coefs_of_parab = np.polyfit(xdata[min_chisq_model-idx_left : min_chisq_model+idx_right],
                                        chisqs[min_chisq_model-idx_left : min_chisq_model+idx_right],
                                        2)
            # print(xdata[min_chisq_model-idx_left], xdata[min_chisq_model+idx_right], xdata[min_chisq_model])
            # Set up grid of values to interpolate parabola to.
            param_grid = np.linspace(xdata[min_chisq_model-idx_left], xdata[min_chisq_model+idx_right], 20000)
            # Now interpolate this parabola onto the grid.
            interped_parabola = np.poly1d(coefs_of_parab)
            # And get corresponding chisqs of the grid of params.
            chisqs_interps = interped_parabola(param_grid)
            # Best fit minimum chisq is min of this interpolated parabola.
            minchisq = chisqs_interps.min()
            # And find parameter corresponding to this chisq.
            best_fit_param = param_grid[np.argmin(chisqs_interps)]
            # Need to interpolate both sides of parabola seperately,
            # since there are two values where change in chisq is 1.

            # First interpolate over the grid for values > best_fit_param
            # and find upper interval bound.
            chisq_interp_upper = spinterp.interp1d(chisqs_interps[param_grid > best_fit_param],
                                                param_grid[param_grid > best_fit_param])
            param_upper_1sig = chisq_interp_upper(minchisq+1)
            # param_upper_3sig = chisq_interp_upper(minchisq+9)

            # Interpolate for values <= best_fit_param to find lower
            # interval bound.
            chisq_interp_lower = spinterp.interp1d(chisqs_interps[param_grid <= best_fit_param],
                                                param_grid[param_grid <= best_fit_param])
            param_lower_1sig = chisq_interp_lower(minchisq+1)
            # # param_lower_3sig = chisq_interp_lower(minchisq+9)

            # print(f"{quantity} = ({best_fit_param:.0f} +{(param_upper_1sig - best_fit_param):.0f} -{(best_fit_param - param_lower_1sig):.0f}) {unit} (1 sigma)")
            # print(f"T = ({best_fit_param:.2f} +{(param_upper_3sig - best_fit_param):.2f} -{(best_fit_param - param_lower_3sig):.2f}) K (3 sigma)")

            # Plot T in kK
            chisq_ax.plot(xdata, chisqs, "ko")
            # Plot parabola.
            chisq_ax.plot(xrange_to_plot, interped_parabola(xrange_to_plot), "-", c="grey")
            # Mark min of parabola.
            chisq_ax.scatter(best_fit_param, minchisq, c="grey", marker="+")
            # Plot 1 sigma CIs.
            chisq_ax.axvline(param_upper_1sig, c="grey", ls="-", alpha=0.7)
            chisq_ax.axvline(param_lower_1sig, c="grey", ls="-", alpha=0.7)
            # # Plot 3 sigma CIs.
            # chisq_ax.axvline(param_upper_3sig, c="grey", ls="--", alpha=0.7)
            # chisq_ax.axvline(param_lower_3sig, c="grey", ls="-", alpha=0.7)
            chisq_ax.xaxis.set_minor_locator(AutoMinorLocator())
            chisq_ax.yaxis.set_minor_locator(AutoMinorLocator())
            chisq_ax.tick_params(axis="both", which="both", top=True, direction="inout")
            chisq_ax.tick_params(axis="both", which="major", length=8, labelsize=11)
            chisq_ax.tick_params(axis="both", which="minor", length=3)
            # print(xlabel)
            chisq_ax.set_xlabel(xlabel ,size=15)
            chisq_ax.set_ylabel(r"$X^2$", size=15)
            vline_handles = lines.Line2D([],[], marker="|", ls="None", c="grey",ms=15)

            # Get optimal Teff to 2 sigfigs
            opt_val = round_to_2(best_fit_param, param_upper_1sig - best_fit_param)
            chisq_ax.legend([(vline_handles, vline_handles)], [r"$1\sigma$ CI"],
                            numpoints=1,handler_map={tuple: HandlerTuple(ndivide=None)},
                            title=rf"{quantity} = ({opt_val[0]} $\pm$ {opt_val[1]}) {unit}")

            return_vals = np.array([best_fit_param, param_upper_1sig - best_fit_param])

        # Now, if looking for an upper limit
        elif fit=="upper_lim":

            index_to_fit_from = np.where(np.abs(np.diff(chisqs)>0.025))[0][0]
            # Define a linspace over to plot parabola.
            xrange_to_plot = np.linspace(xdata[index_to_fit_from], xdata.max(), 100)
            # Fit parabola to idx_left points to the left of the minimum
            # value, and idx_right points to the right of the minimmum.
            coefs_of_parab = np.polyfit(xdata[index_to_fit_from : min_chisq_model+idx_right],
                                        chisqs[index_to_fit_from : min_chisq_model+idx_right],
                                        2)

            # Set up grid of values to interpolate parabola to.
            param_grid = np.linspace(xdata[index_to_fit_from], xdata[min_chisq_model+idx_right], 20000)
            # Now interpolate this parabola onto the grid.
            interped_parabola = np.poly1d(coefs_of_parab)

            # And get corresponding chisqs of the grid of params.
            chisqs_interps = interped_parabola(param_grid)
            # Best fit minimum chisq is min of this interpolated parabola.
            minchisq = chisqs_interps.min()
            # And find parameter corresponding to this chisq.
            best_fit_param = param_grid[np.argmin(chisqs_interps)]

            # Interpolate over the grid for values > param_best and find upper bounds.
            chisq_interp_upper = spinterp.interp1d(chisqs_interps[param_grid > best_fit_param],
                                                   param_grid[param_grid > best_fit_param])

            param_upper_1sig = chisq_interp_upper(minchisq+1)
            param_upper_3sig = chisq_interp_upper(minchisq+9)

            # Now calculating physical parameters of this model.
            logL = float(model_names[0][18:24])
            logg = float(model_names[0][5:9])

            teff = float(model_names[0][10:17])*1e3

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

            radius_cm = prefactors * distance_cm* (teff)**(-0.5)
            L_cgs = 3.828e33 * 10**logL
            radius_cm = np.sqrt(L_cgs / (4*np.pi*5.67037441e-5*(teff**4)))

            radius_solar = radius_cm / 6.957e10

            m_cgs = 10**logg * radius_cm**2 / (6.674e-8)
            m_solar = m_cgs / 1.989e33
            dldr=8*np.pi*radius_cm*5.67037441e-5*((teff)**4)
            dldt=16*np.pi*(radius_cm**2)*5.67037441e-5*((teff)**3)
            drdt = -0.5*prefactors*((teff**(-1.5)))
            drdd = prefactors*((teff)**(-0.5))

            sigmad = distance_cm_uncertainty
            sigmaT = param_upper_1sig

            # Define radius uncertainty in the mean time, will be used later (the square root in sigmaL1)
            sigmaR = np.sqrt((drdt*sigmaT)**2 + (drdd*sigmad)**2)
            # sigmaR_solar = sigmaR / 6.957e10

            sigmaL1 = dldr * sigmaR
            sigmaL2 = dldt * sigmaT
            dL_cgs = np.sqrt((sigmaL1)**2 + (sigmaL2)**2)

            # Now define mass uncertainty
            # sigmaM_solar = abs(2*(10**logg) * radius_cm *sigmaR/ (6.674e-8*3.828e33))
            dL_solar = dL_cgs / (np.log(10)*L_cgs)

            # Array used for calculating theoretical mass loss rates.
            mdot_inputs = np.array([[teff, logg, logL, radius_solar, m_solar]])

            # Now plot chi sqaured vals
            chisq_ax.plot(xdata, chisqs - minchisq, "ko")
            chisq_ax.plot(xrange_to_plot, interped_parabola(xrange_to_plot) - minchisq,
                          "-", c="grey")

            chisq_ax.plot([xdata[0], xdata[index_to_fit_from]],
                          [minchisq - minchisq, minchisq - minchisq],
                          "-", c="grey")
            chisq_ax.set_ylim(-0.05*(chisqs - minchisq).max(), (chisqs - minchisq).max() +0.05*(chisqs - minchisq).max())

            # # Plot 1 sigma limit.
            # label_initial_upper_1sig=f"{(param_upper_1sig):.1e}" # label with exponential as "e"
            # label_split_upper_1sig = label_initial_upper_1sig.split("e")
            # label_upper_1sig = label_split_upper_1sig[0] + fr"$\times 10^{{{int(label_split_upper_1sig[1])}}}$"
            # chisq_ax.axvline(param_upper_1sig, c="grey", ls="--", alpha=0.7,# label = r"$3\sigma$")
            #                 label = rf"$\dot{{M}}<$ {label_upper_1sig} {unit} ($1\sigma$)")

            # Plot 3 sigma limit.

            label_initial_upper_3sig=f"{(param_upper_3sig):.1e}" # label with exponential as "e"
            label_split_upper_3sig = label_initial_upper_3sig.split("e")
            label_upper_3sig = label_split_upper_3sig[0] + fr"$\times 10^{{{int(label_split_upper_3sig[1])}}}$"
            chisq_ax.axvline(param_upper_3sig, c="grey", ls="-.", alpha=0.7,# label = r"$3\sigma$")
                            label = rf"$\dot{{M}}<$ {label_upper_3sig} {unit} ($3\sigma$)")

            chisq_ax.xaxis.set_minor_locator(AutoMinorLocator())
            chisq_ax.yaxis.set_minor_locator(AutoMinorLocator())
            chisq_ax.tick_params(axis="both", which="both", top=True,direction="inout")
            chisq_ax.tick_params(axis="both", which="major", length=8, labelsize=11)
            chisq_ax.tick_params(axis="both", which="minor", length=3)

            chisq_ax.set_xlabel(xlabel ,size=15)
            chisq_ax.set_ylabel(r"$\Delta X^2$", size=15)

            chisq_ax.legend()

            return_vals = np.array([logL, param_upper_1sig, param_upper_3sig,  vink_mdot(mdot_inputs,0.03)[0], krticka_mdot(mdot_inputs, 0.03)[0], bjorklund_mdot(mdot_inputs, 0.03)[0], dL_solar])#, param_upper_5sig])

        # Now plot again on twin axis, so we can set y label to "Normalised Flux"

        chisq_ax1.set_yticks(chisq_ax.get_yticks()[1:-1])
        chisq_ax1.set_yticks(chisq_ax.get_yticks(minor=True), minor=True)
        chisq_ax1.set_yticklabels([])
        chisq_ax1.set_ylim(chisq_ax.get_ylim())
        chisq_ax1.tick_params(axis="both", which="both", direction="inout")
        chisq_ax1.tick_params(axis="both", which="major", length=8, labelsize=11)
        chisq_ax1.tick_params(axis="both", which="minor", length=3)
        chisq_ax1.set_ylabel("Normalised Flux", size=15)

    # If no fit is to be performed, just plot chisqs of each model.
    else:

        if xdata is None:
            chisq_ax.plot(chisqs, "ko-")
        else:
            chisq_ax.plot(xdata, chisqs, "ko-")
        # chisq_ax.set_xticks(range(len(model_names)))
        # chisq_ax.set_xticklabels(model_names)
        chisq_ax.xaxis.set_minor_locator(AutoMinorLocator())
        chisq_ax.yaxis.set_minor_locator(AutoMinorLocator())
        chisq_ax.tick_params(axis="both", which="both", top=True, right=True)
        chisq_ax.set_ylabel(r"$X^2$", size=15)

    model_name_for_title = model_names[0].split("-")
    ab, g = model_name_for_title[0], model_name_for_title[1]
    chisq_fig.suptitle(fr"[C] = {ab} C$_\odot$,"+r" log($g$/cm s$^{-2}$) = "+g, size=17)

    if savename is not None: # If want to save.
        if dir_to_save is None: # If no directory provided.
            dir_to_save = os.getcwd()
        plt.savefig(dir_to_save+savename, dpi=200)
    if show:
        plt.show()

    if return_estimates:
        return return_vals

if __name__ == "__main__":
    # Load in data
    data_lines1, uv_line_errs1, data_names1 = get_renorm_lines()
    plot_lines(data_lines1, uv_line_errs1, data_names1, lines_to_use=["CIV1550", "CIII1176"])
    # Loading in models.
    # Mdot Grid with 0.01 c sun, log(g)=3.42.
    model_lines_md01_3p42, model_names_md01_3p42, line_names_md01_3p42 = load_models(path_to_mdotgrid_01_3p42, indices_of_number=[20, None], using_trgrid=True)

    # # Mdot Grid with 0.01 c sun, log(g)=3.67.
    model_lines_md01_3p67, model_names_md01_3p67, line_names_md01_3p67 = load_models(path_to_mdotgrid_01_3p67, indices_of_number=[20, None], using_trgrid=True)

    # # Mdot Grid with 0.01 c sun, log(g)=3.92.
    model_lines_md01_3p92, model_names_md01_3p92, line_names_md01_3p92 = load_models(path_to_mdotgrid_01_3p92, indices_of_number=[20, None], using_trgrid=True)

    # Mdot Grid with 0.01 c sun, log(g)=4.17.
    model_lines_md01_4p17, model_names_md01_4p17, line_names_md01_4p17 = load_models(path_to_mdotgrid_01_4p17, indices_of_number=[20, None], using_trgrid=True)

    # Mdot Grid with 0.01 c sun, log(g)=4.42.
    model_lines_md01_4p42, model_names_md01_4p42, line_names_md01_4p42 = load_models(path_to_mdotgrid_01_4p42, indices_of_number=[20, None], using_trgrid=True)

    mdots_md01_3p42 = []
    for name in model_names_md01_3p42:
        # print(name[25:])
        mdots_md01_3p42.append(float(name[25:]))

    mdots_md01_3p67 = []
    for name in model_names_md01_3p67:
        # print(name[25:])
        mdots_md01_3p67.append(float(name[25:]))

    mdots_md01_3p92 = []
    for name in model_names_md01_3p92:
        # print(name[25:])
        mdots_md01_3p92.append(float(name[25:]))

    mdots_md01_4p17 = []
    for name in model_names_md01_4p17:
        # print(name[25:])
        mdots_md01_4p17.append(float(name[25:]))

    mdots_md01_4p42 = []
    for name in model_names_md01_4p42:
        # print(name[25:])
        mdots_md01_4p42.append(float(name[25:]))

    md01_3p42_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md01_3p42, model_names_md01_3p42, line_names_md01_3p42, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=3, xdata=np.array(mdots_md01_3p42), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)

    md01_3p67_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md01_3p67, model_names_md01_3p67, line_names_md01_3p67, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=4, xdata=np.array(mdots_md01_3p67), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)##, savename="example_upperlim.jpg")

    md01_3p92_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md01_3p92, model_names_md01_3p92, line_names_md01_3p92, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=4, xdata=np.array(mdots_md01_3p92), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)

    md01_4p17_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md01_4p17, model_names_md01_4p17, line_names_md01_4p17, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=4, xdata=np.array(mdots_md01_4p17), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)

    md01_4p42_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md01_4p42, model_names_md01_4p42, line_names_md01_4p42, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=5, xdata=np.array(mdots_md01_4p42), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)

    mdot_vs_l_01 = np.array([md01_3p42_parms, md01_3p67_parms, md01_3p92_parms, md01_4p17_parms, md01_4p42_parms])

    # Mdot Grid with 0.03 c sun, log(g)=3.42.
    model_lines_md03_3p42, model_names_md03_3p42, line_names_md03_3p42 = load_models(path_to_mdotgrid_03_3p42, indices_of_number=[20, None], using_trgrid=True)

    # Mdot Grid with 0.03 c sun, log(g)=3.67.
    model_lines_md03_3p67, model_names_md03_3p67, line_names_md03_3p67 = load_models(path_to_mdotgrid_03_3p67, indices_of_number=[20, None], using_trgrid=True)

    # Mdot Grid with 0.03 c sun, log(g)=3.92.
    model_lines_md03_3p92, model_names_md03_3p92, line_names_md03_3p92 = load_models(path_to_mdotgrid_03_3p92, indices_of_number=[20, None], using_trgrid=True)

    # Mdot Grid with 0.03 c sun, log(g)=4.17.
    model_lines_md03_4p17, model_names_md03_4p17, line_names_md03_4p17 = load_models(path_to_mdotgrid_03_4p17, indices_of_number=[20, None], using_trgrid=True)

    # Mdot Grid with 0.03 c sun, log(g)=4.42.
    model_lines_md03_4p42, model_names_md03_4p42, line_names_md03_4p42 = load_models(path_to_mdotgrid_03_4p42, indices_of_number=[20, None], using_trgrid=True)

    mdots_md03_3p42 = []
    for name in model_names_md03_3p42:
        # print(name[25:])
        mdots_md03_3p42.append(float(name[25:]))

    mdots_md03_3p67 = []
    for name in model_names_md03_3p67:
        # print(name[25:])
        mdots_md03_3p67.append(float(name[25:]))

    mdots_md03_3p92 = []
    for name in model_names_md03_3p92:
        # print(name[25:])
        mdots_md03_3p92.append(float(name[25:]))

    mdots_md03_4p17 = []
    for name in model_names_md03_4p17:
        # print(name[25:])
        mdots_md03_4p17.append(float(name[25:]))

    mdots_md03_4p42 = []
    for name in model_names_md03_4p42:
        # print(name[25:])
        mdots_md03_4p42.append(float(name[25:]))

    md03_3p42_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md03_3p42, model_names_md03_3p42, line_names_md03_3p42, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=3, xdata=np.array(mdots_md03_3p42), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)

    md03_3p67_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md03_3p67, model_names_md03_3p67, line_names_md03_3p67, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=4, xdata=np.array(mdots_md03_3p67), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)

    md03_3p92_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md03_3p92, model_names_md03_3p92, line_names_md03_3p92, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=4, xdata=np.array(mdots_md03_3p92), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)

    md03_4p17_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md03_4p17, model_names_md03_4p17, line_names_md03_4p17, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=4, xdata=np.array(mdots_md03_4p17), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)

    md03_4p42_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md03_4p42, model_names_md03_4p42, line_names_md03_4p42, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=3, xdata=np.array(mdots_md03_4p42), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)

    mdot_vs_l_03 = np.array([md03_3p42_parms, md03_3p67_parms, md03_3p92_parms, md03_4p17_parms, md03_4p42_parms])


    # Mdot Grid with 0.05 c sun, log(g)=3.42.
    model_lines_md05_3p42, model_names_md05_3p42, line_names_md05_3p42 = load_models(path_to_mdotgrid_05_3p42, indices_of_number=[20, None], using_trgrid=True)

    # Mdot Grid with 0.05 c sun, log(g)=3.67.
    model_lines_md05_3p67, model_names_md05_3p67, line_names_md05_3p67 = load_models(path_to_mdotgrid_05_3p67, indices_of_number=[20, None], using_trgrid=True)

    # Mdot Grid with 0.05 c sun, log(g)=3.92.
    model_lines_md05_3p92, model_names_md05_3p92, line_names_md05_3p92 = load_models(path_to_mdotgrid_05_3p92, indices_of_number=[20, None], using_trgrid=True)

    # Mdot Grid with 0.05 c sun, log(g)=4.17.
    model_lines_md05_4p17, model_names_md05_4p17, line_names_md05_4p17 = load_models(path_to_mdotgrid_05_4p17, indices_of_number=[20, None], using_trgrid=True)

    # Mdot Grid with 0.05 c sun, log(g)=4.42.
    model_lines_md05_4p42, model_names_md05_4p42, line_names_md05_4p42 = load_models(path_to_mdotgrid_05_4p42, indices_of_number=[20, None], using_trgrid=True)

    mdots_md05_3p42 = []
    for name in model_names_md05_3p42:
        # print(name[25:])
        mdots_md05_3p42.append(float(name[25:]))

    mdots_md05_3p67 = []
    for name in model_names_md05_3p67:
        # print(name[25:])
        mdots_md05_3p67.append(float(name[25:]))

    mdots_md05_3p92 = []
    for name in model_names_md05_3p92:
        # print(name[25:])
        mdots_md05_3p92.append(float(name[25:]))

    mdots_md05_4p17 = []
    for name in model_names_md05_4p17:
        # print(name[25:])
        mdots_md05_4p17.append(float(name[25:]))

    mdots_md05_4p42 = []
    for name in model_names_md05_4p42:
        # print(name[25:])
        mdots_md05_4p42.append(float(name[25:]))

    md05_3p42_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md05_3p42, model_names_md05_3p42, line_names_md05_3p42, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=3, xdata=np.array(mdots_md05_3p42), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)

    md05_3p67_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md05_3p67, model_names_md05_3p67, line_names_md05_3p67, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=4, xdata=np.array(mdots_md05_3p67), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)#

    md05_3p92_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md05_3p92, model_names_md05_3p92, line_names_md05_3p92, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=4, xdata=np.array(mdots_md05_3p92), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)

    md05_4p17_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md05_4p17, model_names_md05_4p17, line_names_md05_4p17, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=4, xdata=np.array(mdots_md05_4p17), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)

    md05_4p42_parms = chi_sq(data_lines1, uv_line_errs1, model_lines_md05_4p42, model_names_md05_4p42, line_names_md05_4p42, ["CIII1176", "CIV1550"],
           fit="upper_lim", using_renorm=True, data_names=data_names1, idx_right=3, xdata=np.array(mdots_md05_4p42), xlabel=r"$\dot{M}$ ($M_{\odot}\ {\rm yr}^{-1}$)",
           return_estimates=True, show=False)

    mdot_vs_l_05 = np.array([md05_3p42_parms, md05_3p67_parms, md05_3p92_parms, md05_4p17_parms, md05_4p42_parms])

    plt.close("all")
    mdot_fig, mdot_axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5), sharey=True)

    mdots = [mdot_vs_l_01, mdot_vs_l_03, mdot_vs_l_05]
    abun_labels = ["0.01", "0.03", "0.05"]
    # print(mdots)
    for i, ax in enumerate(mdot_axs):
        ax.errorbar(mdots[i][:,0] - 6, np.log10(mdots[i][:,1]), #xerr=mdots[i][:,-1]
                    xerr=mdots[i][:,-1],yerr=np.full_like(np.log10(mdots[i][:,1]), 0.1),
                    elinewidth=0.6,
                    fmt="o", c="k", uplims=True, label=r"$1\sigma$")
        ax.errorbar(mdots[i][:,0] - 6, np.log10(mdots[i][:,2]),
                    xerr=mdots[i][:,-1],yerr=np.full_like(np.log10(mdots[i][:,2]), 0.1),
                    elinewidth=0.6,
                    fmt="o", c="r", uplims=True, label=r"$3\sigma$")
        ax.plot(mdots[i][:,0] - 6, mdots[i][:,3], "-", c="gray", label="Vink et al.")
        ax.plot(mdots[i][:,0] - 6, mdots[i][:,4], "--", c="gray", label=r"Krti\v{c}ka and Kub\'{a}t")
        ax.plot(mdots[i][:,0] - 6, mdots[i][:,5], ":", c="gray", label="Bjorklund et al.")
        ax.set_xlabel(r"$\log\left(L/10^{6}L_{\odot}\right)$",size=19)
        ax.set_xlim(mdots[i][:,0][0] - 6.005, mdots[i][:,0][-1] - 5.995)
        ax.set_title(rf"[C] = {abun_labels[i]} C$_\odot$", size=19, y=1.02)
        ax.tick_params(which="major", direction="inout", top=True, right=True, length=8, labelsize=15)
        if i == 0:
            ax.set_ylabel(r"$\log\left(\dot{M}/M_{\odot}\ {\rm yr}^{-1}\right)$", size=19)
        if i == 0:
            # ax.legend(loc=(1.05,0.7), bbox_transform=ax.transAxes)
            ax.legend(loc=2, fontsize=12)
    plt.tight_layout()
    plt.show()


    # # TR Grid with 0.01 c sun, log(g)=3.42.
    # model_lines_tr01_3p42, model_names_tr01_3p42, line_names_tr01_3p42 = load_models(path_to_trgrid_01_3p42, indices_of_number=[5, 9], using_trgrid=True)

    # # TR Grid with 0.01 c sun, log(g)=3.67.
    # model_lines_tr01_3p67, model_names_tr01_3p67, line_names_tr01_3p67 = load_models(path_to_trgrid_01_3p67, indices_of_number=[5, 9], using_trgrid=True)

    # # TR Grid with 0.01 c sun, log(g)=3.92.
    # model_lines_tr01_3p92, model_names_tr01_3p92, line_names_tr01_3p92 = load_models(path_to_trgrid_01_3p92, indices_of_number=[5, 9], using_trgrid=True)

    # # TR Grid with 0.01 c sun, log(g)=4.17.
    # model_lines_tr01_4p17, model_names_tr01_4p17, line_names_tr01_4p17 = load_models(path_to_trgrid_01_4p17, indices_of_number=[5, 9], using_trgrid=True)

    # # TR Grid with 0.01 c sun, log(g)=4.42.
    # model_lines_tr01_4p42, model_names_tr01_4p42, line_names_tr01_4p42 = load_models(path_to_trgrid_01_4p42, indices_of_number=[5, 9], using_trgrid=True)

    # # TR Grid with 0.03 c sun, log(g)=3.42.
    # model_lines_tr03_3p42, model_names_tr03_3p42, line_names_tr03_3p42 = load_models(path_to_trgrid_03_3p42, indices_of_number=[5, 9], using_trgrid=True)

    # # TR Grid with 0.03 c sun, log(g)=3.67.
    # model_lines_tr03_3p67, model_names_tr03_3p67, line_names_tr03_3p67 = load_models(path_to_trgrid_03_3p67, indices_of_number=[5, 9], using_trgrid=True)

    # # TR Grid with 0.03 c sun, log(g)=3.92.
    # model_lines_tr03_3p92, model_names_tr03_3p92, line_names_tr03_3p92 = load_models(path_to_trgrid_03_3p92, indices_of_number=[5, 9], using_trgrid=True)

    # TR Grid with 0.03 c sun, log(g)=4.17.
    model_lines_tr03_4p17, model_names_tr03_4p17, line_names_tr03_4p17 = load_models(path_to_trgrid_03_4p17, indices_of_number=[5, 9], using_trgrid=True)

    # # TR Grid with 0.03 c sun, log(g)=4.42.
    # model_lines_tr03_4p42, model_names_tr03_4p42, line_names_tr03_4p42 = load_models(path_to_trgrid_03_4p42, indices_of_number=[5, 9], using_trgrid=True)

    # # TR Grid with 0.05 c sun, log(g)=3.42.
    # model_lines_tr05_3p42, model_names_tr05_3p42, line_names_tr05_3p42 = load_models(path_to_trgrid_05_3p42, indices_of_number=[5, 9], using_trgrid=True)

    # # TR Grid with 0.05 c sun, log(g)=3.67.
    # model_lines_tr05_3p67, model_names_tr05_3p67, line_names_tr05_3p67 = load_models(path_to_trgrid_05_3p67, indices_of_number=[5, 9], using_trgrid=True)

    # # TR Grid with 0.05 c sun, log(g)=3.92.
    # model_lines_tr05_3p92, model_names_tr05_3p92, line_names_tr05_3p92 = load_models(path_to_trgrid_05_3p92, indices_of_number=[5, 9], using_trgrid=True)

    # # TR Grid with 0.05 c sun, log(g)=4.17.
    # model_lines_tr05_4p17, model_names_tr05_4p17, line_names_tr05_4p17 = load_models(path_to_trgrid_05_4p17, indices_of_number=[5, 9], using_trgrid=True)

    # # TR Grid with 0.05 c sun, log(g)=4.42.
    # model_lines_tr05_4p42, model_names_tr05_4p42, line_names_tr05_4p42 = load_models(path_to_trgrid_05_4p42, indices_of_number=[5, 9], using_trgrid=True)


    # grid_temps = []
    # for name in model_names_tr03_3p67:
    #     grid_temps.append(float(name[10:-5]))

    # tr01_3p42_grid_temps = []
    # for name in model_names_tr01_3p42:
    #     tr01_3p42_grid_temps.append(float(name[10:-5]))

    # tr01_3p67_grid_temps = []
    # for name in model_names_tr01_3p67:
    #     tr01_3p67_grid_temps.append(float(name[10:-5]))

    # tr01_3p92_grid_temps = []
    # for name in model_names_tr01_3p92:
    #     tr01_3p92_grid_temps.append(float(name[10:-5]))

    # tr01_4p17_grid_temps = []
    # for name in model_names_tr01_4p17:
    #     tr01_4p17_grid_temps.append(float(name[10:-5]))

    # tr01_4p42_grid_temps = []
    # for name in model_names_tr01_4p42:
    #     tr01_4p42_grid_temps.append(float(name[10:-5]))

    # tr03_3p42_grid_temps = []
    # for name in model_names_tr03_3p42:
    #     tr03_3p42_grid_temps.append(float(name[10:-5]))

    # tr03_3p67_grid_temps = []
    # for name in model_names_tr03_3p67:
    #     tr03_3p67_grid_temps.append(float(name[10:-5]))

    # tr03_3p92_grid_temps = []
    # for name in model_names_tr03_3p92:
    #     tr03_3p92_grid_temps.append(float(name[10:-5]))

    tr03_4p17_grid_temps = []
    for name in model_names_tr03_4p17:
        tr03_4p17_grid_temps.append(float(name[10:-5]))

    # tr03_4p42_grid_temps = []
    # for name in model_names_tr03_4p42:
    #     tr03_4p42_grid_temps.append(float(name[10:-5]))

    # tr05_3p42_grid_temps = []
    # for name in model_names_tr05_3p42:
    #     tr05_3p42_grid_temps.append(float(name[10:-5]))

    # tr05_3p67_grid_temps = []
    # for name in model_names_tr05_3p67:
    #     tr05_3p67_grid_temps.append(float(name[10:-5]))

    # tr05_3p92_grid_temps = []
    # for name in model_names_tr05_3p92:
    #     tr05_3p92_grid_temps.append(float(name[10:-5]))

    # tr05_4p17_grid_temps = []
    # for name in model_names_tr05_4p17:
    #     tr05_4p17_grid_temps.append(float(name[10:-5]))

    # tr05_4p42_grid_temps = []
    # for name in model_names_tr05_4p42:
    #     tr05_4p42_grid_temps.append(float(name[10:-5]))





    # chi_sq(data_lines1, uv_line_errs1, model_lines_md01_3p67, model_names_md01_3p67, line_names_md01_3p67, ["CIII1176", "CIV1550"],
    #        fit=None, using_renorm=True, data_names=data_names1)#, idx_left=2, idx_right=2, xdata=np.array(md01), xlabel=r"$T_{\rm eff}$ (K)",
    #     #    using_renorm=True, data_names=data_names1)#, savename="example_gridsearch03.jpg")

    # chi_sq(data_lines1, uv_line_errs1, model_lines_md01_3p92, model_names_md01_3p92, line_names_md01_3p92, ["CIII1176", "CIV1550"],
    #        fit=None, using_renorm=True, data_names=data_names1)#, idx_left=2, idx_right=2, xdata=np.array(tr03_3p92_grid_temps), xlabel=r"$T_{\rm eff}$ (K)",
    #     #    using_renorm=True, data_names=data_names1)#, savename="example_gridsearch03.jpg")

    # chi_sq(data_lines1, uv_line_errs1, model_lines_md01_4p17, model_names_md01_4p17, line_names_md01_4p17, ["CIII1176", "CIV1550"],
    #        fit=None, using_renorm=True, data_names=data_names1)#, idx_left=1, idx_right=1, xdata=np.array(tr03_4p17_grid_temps), xlabel=r"$T_{\rm eff}$ (K)",
    #     #    using_renorm=True, data_names=data_names1)

    # chi_sq(data_lines1, uv_line_errs1, model_lines_md01_4p42, model_names_md01_4p42, line_names_md01_4p42, ["CIII1176", "CIV1550"],
    #     fit=None, using_renorm=True, data_names=data_names1)#, idx_left=1, idx_right=1, xdata=np.array(tr03_4p42_grid_temps), xlabel=r"$T_{\rm eff}$ (K)",
    #     # using_renorm=True, data_names=data_names1)

    # #  print(model_names_md01_3p42)
    # chi_sq(data_lines1, uv_line_errs1, model_lines_md03_3p42, model_names_md03_3p42, line_names_md03_3p42, ["CIII1176", "CIV1550"],
    #        fit=None, using_renorm=True, data_names=data_names1)#, idx_left=1, idx_right=1, xdata=np.array([]), xlabel=r"$T_{\rm eff}$ (K)",
    #     #    using_renorm=True, data_names=data_names1)

    # chi_sq(data_lines1, uv_line_errs1, model_lines_md03_3p67, model_names_md03_3p67, line_names_md03_3p67, ["CIII1176", "CIV1550"],
    #        fit=None, using_renorm=True, data_names=data_names1)#, idx_left=2, idx_right=2, xdata=np.array(md01), xlabel=r"$T_{\rm eff}$ (K)",
    #     #    using_renorm=True, data_names=data_names1)#, savename="example_gridsearch03.jpg")

    # chi_sq(data_lines1, uv_line_errs1, model_lines_tr03_3p92, model_names_tr03_3p92, line_names_tr03_3p92, ["CIII1176"],
    #        fit="CIs", using_renorm=True, data_names=data_names1, idx_left=2, idx_right=2, xdata=np.array(tr03_3p92_grid_temps), xlabel=r"$T_{\rm eff}$ (kK)", savename="example_gridsearch.jpg")

    chi_sq(data_lines1, uv_line_errs1, model_lines_tr03_4p17, model_names_tr03_4p17, line_names_tr03_4p17, ["CIII1176"],
           fit="CIs", using_renorm=True, data_names=data_names1, idx_left=2, idx_right=2, xdata=np.array(tr03_4p17_grid_temps), xlabel=r"$T_{\rm eff}$ (kK)")

    # chi_sq(data_lines1, uv_line_errs1, model_lines_md03_4p42, model_names_md03_4p42, line_names_md03_4p42, ["CIII1176", "CIV1550"],
    #     fit=None, using_renorm=True, data_names=data_names1)

