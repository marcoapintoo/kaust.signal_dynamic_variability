import os
import numpy as np
import scipy.io
import pickle
from utils.textgrid import TextGridExtractor
from signal_dynamic_variabiity import random_projection
from signal_dynamic_variabiity import tv_var
from analysis import conditionals
from analysis import plotter    
from analysis import asymptotic
import seaborn as sns

def extract_labels_movie():
    file_storage = "./datasets/scripts/word_marks.npz"
    if not os.path.exists(file_storage):
        extractor = TextGridExtractor(word_sampling_frequency=1/1.5)
        video_length = [
            325.5026875,
            343.5206875,
            331.5086875,
            331.5086875,
            280.524375,
            340.5176875
        ]
        print(":: Extracting labels")
        for n in range(1, 7):
            filename = "./datasets/scripts/MovieScript{0:02d}.TextGrid".format(n)
            print("::: Processing", filename)
            extractor.extract_from_file(filename=filename, maximum_length=video_length[n - 1])

        print("::: Categories:", extractor.categories)
        for name, word_vectors in extractor.script_values:
            print("::: Category {0} | {1} words | Last 10 words: {2} ".format(name, word_vectors.shape, word_vectors[-10:]))

        with open("./datasets/scripts/speech_non_speech.csv", "wt") as f:
            f.write("\n".join(extractor.words_in_category(category="speech_nonspeech", return_code=False)))

        np.savez_compressed(file_storage, **extractor.sequences_by_category_word)
    return np.load(file_storage)

"""
# Old definition
# Less useful
def filter_labels(labels, category):
    filtered_labels = {}
    for key, value in labels.items():
        if not key.startswith(category):
            continue
        filtered_labels[key[len(category) + 1:].strip()] = value
    return filtered_labels
"""
def filter_labels(labels, category):
    names = []
    values = 0
    for key, value in labels.items():
        if not key.startswith(category):
            continue
        keyname = key[len(category) + 1:].strip()
        values += len(names) * value
        names.append(keyname)
    return names, values

def filter_group_labels(labels, category):
    values = 0
    group_names = [key[len(category) + 1:].strip().split("_")[-1].strip() for key, value in labels.items() if key.startswith(category)]
    group_names = np.unique(group_names).tolist()
    for key, value in labels.items():
        if not key.startswith(category):
            continue
        keyname = key[len(category) + 1:].strip().split("_")[-1].strip() 
        values += group_names.index(keyname) * value
    return group_names, values

def load_dataset(codename):
    subject_dataset = scipy.io.loadmat("./datasets/sample_fmri/{0}.mat".format(codename))['mean_roi'].T
    return subject_dataset

def load_dataset_fieldnames():
    return [ "CG.L", "PP.L", "PT.L", "PoCG.L", "PreCG.L", "STG.L", "TTG.L", "TTS.L", "CG.R", "PP.R", "PT.R", "PoCG.R", "PreCG.R", "STG.R", "TTG.R", "TTS.R", ]

def load_dataset_values():
    userids = [
        6251, 6389, 6492, 6550,
        6551, 6617, 6643, 6648, 
        6737, 6743, 6756, 6769, #6781, 6791
    ]
    order_background = [
        ['song', 'jeff_song',],
        ['music', 'megan_music', 'sierra_music', 'jeff_megan_music', 'jeff_music'], 
        ['jeff_noise', 'megan_noise', 'noise', 'nathan_noise', 'student_noise', 'jeff_megan_noise'],
        ['silence', 'jeff_silence', 'mom_silence', 'olivia_silence', 'megan_silence',],
    ]
    locutors_vs_silence = [
        ['silence'],
        #['music', 'noise', 'song',],
        ['megan_music', 'megan_noise', 'megan_silence', ],
        ['jeff_megan_music', 'jeff_megan_noise',],
        ['jeff_song', 'jeff_music',
        'jeff_noise',
        'jeff_silence', ]
    ]
    main_labels_rows = [
        ['song',],
        ['music',], 
        ['noise',],
        ['silence',],
    ]
    main_labels = [
        ['music', 'song'],
        ['silence', 'noise'],
    ]
    return userids, order_background, locutors_vs_silence, main_labels_rows, main_labels

def get_empirical_asymptotic(signal_names, projector):
    cache_name = "_asymptotic_inverse_projection_{0}.pickle".format(projector.n_estimators)
    if not os.path.exists(cache_name):
        asymptotic_inverse_projection = asymptotic.AsymptoticMatrixInverseProjection(signal_names)
        plottable_objects = asymptotic_inverse_projection.fit(projector)
        pickle.dump([asymptotic_inverse_projection, plottable_objects], open(cache_name, "wb"))
    asymptotic_inverse_projection, plottable_objects = pickle.load(open(cache_name, "rb"))
    plottable_objects = sorted(plottable_objects, key=lambda o: o.hashcode)
    return asymptotic_inverse_projection, plottable_objects

def plot_empirical_asymptotic_matrices(plottable_objects, as_grid=False, plotbasepath="plots"):
    if len(plottable_objects) > 1000:
        print(":: Currently we cannot print more than 1000 objects")
        return
    if as_grid:
        plot = plotter.DistanceHistogramPlotter(basepath="./{0}/convergence_matrices".format(plotbasepath), filename_template="{codesubject}_convergence_matrix", tight_mode=True)
        plot.m = None
        plot.cmap = sns.diverging_palette(10, 220, sep=100, n=10, as_cmap=True)
        plot.default_title = "State {codesubject}"
        plot.plot_grid("ALL", [p.hashcode for p in plottable_objects], plottable_objects, cols=8)
        return
    for plottable_object in plottable_objects:
        plot = plotter.DistanceHistogramPlotter(basepath="./{0}/convergence_matrices".format(plotbasepath), filename_template="{codesubject}_convergence_matrix", tight_mode=False)
        plot.cmap = sns.diverging_palette(10, 220, sep=100, n=10, as_cmap=True)
        plot.default_title = "State {codesubject}"
        plot.m = None
        plot.plot(plottable_object.hashcode, plottable_object)


#
#
#
from utils.connectivity import fmri_var_to_pdc
# 
def plot_stimuli_grid_per_subject(userid, names, plotable_matrices, plotbasepath="plots"):
    plot = plotter.DistanceHistogramPlotter(basepath="./{0}/convergence_stimuli".format(plotbasepath), filename_template="{codesubject}_stimuli", tight_mode=True)
    plot.m = None
    #plot.cmap = "YlGnBu"
    plot.cmap = sns.diverging_palette(10, 220, sep=100, n=10, as_cmap=True)
    plot.default_title = "Stimuli {codesubject}"
    plot.plot_grid("{0}".format(userid), names, plotable_matrices, cols=4)

def get_average_matrix(key, empirical_conditional, asymptotic_inverse_projection, signal_names, as_pdc=False):
    matrix = 0
    for index, proportion in enumerate(empirical_conditional.get_distribution(key, normalized=True)):
        #if index == 0 or proportion == 0:
        if proportion == 0:
            #Index 0 means that there is no pattern associated
            continue
        if index not in asymptotic_inverse_projection.convergence_vectors:
            print("::: Error index", index, "not found")
            continue
        matrix += proportion * asymptotic_inverse_projection.convergence_vectors[index]
    if np.all(matrix == 0):
        print("## Possible error. Verify error 01. Key", key)
        matrix = np.zeros((asymptotic_inverse_projection.dimensions, asymptotic_inverse_projection.dimensions))
    #matrix = np.log10(matrix ** 2 + 1e-100)
    #matrix = tv_var.var_to_pdc(matrix.reshape(1, matrix.shape[1], -1))[0]
    plotable = asymptotic.MatrixPlotableObject(
        hashcode="",
        keynames=signal_names,
        distance_matrix=matrix if not as_pdc else fmri_var_to_pdc(matrix),
    )
    return plotable

def plot_subject_grid_per_stimuli(stimulus, convergence_matrices, plotbasepath="plots"):
    plot = plotter.DistanceHistogramPlotter(basepath="./{0}/convergence_stimuli".format(plotbasepath), filename_template="{codesubject}_stimuli", tight_mode=True)
    plot.m = None
    #plot.cmap = "YlGnBu"
    plot.cmap = sns.diverging_palette(10, 220, sep=100, n=10, as_cmap=True)
    plot.m = 0.7; plot.cmap = sns.diverging_palette(10, 220, sep=100, n=10, as_cmap=True)
    plot.default_title = "Subject {codesubject}"
    plot.plot_grid("{0}".format(stimulus), [a for a, b in convergence_matrices.items()], [b for a, b in convergence_matrices.items()], cols=4)

def plot_stimuli_mean_across_subjects(stimulus, convergence_matrices, signal_names, plotbasepath="plots"):
    mean_matrix = np.array([cv.distance_matrix for cv in convergence_matrices.values()]).mean(axis=0)
    plotable = asymptotic.MatrixPlotableObject(
        hashcode="",
        keynames=signal_names,
        distance_matrix=mean_matrix,
    )
    plot = plotter.DistanceHistogramPlotter(basepath="./{0}/convergence_stimuli".format(plotbasepath), filename_template="{codesubject}_mean_stimuli")
    plot.m = None
    #plot.cmap = "YlGnBu"
    plot.cmap = sns.diverging_palette(10, 220, sep=100, n=10, as_cmap=True)
    plot.m = 0.7; plot.cmap = sns.diverging_palette(10, 220, sep=100, n=10, as_cmap=True)
    plot.default_title = "Subject {codesubject}"
    plot.plot("{0}".format(stimulus), plotable)


def plot_empirical_distributions(userid, empirical_conditional, orders=[], plotbasepath="plots"):
    plot = plotter.HistogramPlotter(basepath="./{0}/histograms/{1}".format(plotbasepath, userid))
    plot.plot(userid, empirical_conditional)
    
    for basepath, groups, filename_template in orders:
        plotters = [
            plotter.DistanceHistogramPlotter(basepath="./{0}/{1}/histograms".format(plotbasepath, basepath), filename_template=filename_template, label_groups=groups),
            plotter.HistogramMatrixPlotter(basepath="./{0}/{1}/distance_matrix".format(plotbasepath, basepath), filename_template=filename_template, label_groups=groups, tight_mode=True),
        ]
        for plot in plotters:
            plot.plot(userid, empirical_conditional)

def get_plot_empirical_distributions(userid, convergence_matrices_users, empirical_conditional, asymptotic_inverse_projection, signal_names, plotbasepath="plots", as_pdc=False):
    names, plotable_matrices = [], []
    print("KEYNAMES:", sorted(empirical_conditional.keynames))
    for k, key in enumerate(sorted(empirical_conditional.keynames)):
        print("   ", userid, key, empirical_conditional.get_distribution(key, normalized=True)[:5].tolist())
        plotable = get_average_matrix(key, empirical_conditional, asymptotic_inverse_projection, signal_names, as_pdc=as_pdc)
        convergence_matrices_users.setdefault(key, {})
        convergence_matrices_users[key][userid] = plotable
        names.append(key)
        plotable_matrices.append(plotable)
    print("=")
    plot_stimuli_grid_per_subject(userid, names, plotable_matrices, plotbasepath=plotbasepath)


def plot_all_conditionals(empirical_conditionals, orders, rows, cols, plotbasepath="plots"):
    plotters = []
    for basepath, groups, filename_template in orders:
        plotters.append(
            plotter.DistanceHistogramPlotter(basepath="./{0}/{1}".format(plotbasepath, basepath), filename_template=filename_template, label_groups=groups, tight_mode=True)
        )
    for plot in plotters:
        plot.plot_grid("ALL", [a for a, b in empirical_conditionals], [b for a, b in empirical_conditionals], rows=rows, cols=cols)

def default_processing(n_estimators):
    print("=" * 100)
    print("=== Estimators:", n_estimators)
    print("=" * 100)
    userids, order_background, locutors_vs_silence, main_labels_rows, main_labels = load_dataset_values()
    convergence_matrices_users = {}

    signal_names = load_dataset_fieldnames()
    names, movie_labels = filter_labels(extract_labels_movie(), category="speech_nonspeech")
    grouped_names, grouped_movie_labels = filter_group_labels(extract_labels_movie(), category="speech_nonspeech")
    #projector = random_projection.RandomProjector(n_estimators=6)
    projector = random_projection.RandomProjector(n_estimators=n_estimators, random_state=0)
    tvprocess = tv_var.TimeVaryingVAR(window_size=100, step_size=1, window_type="boxcar")
    #print("bins=", np.arange(0.1, 2**projector.random_vector_count + 1.1, 1))
    
    plotbasepath = "plots/{0}".format(2**projector.random_vector_count)
    empirical_conditional_bins = np.arange(0.1, 2**projector.random_vector_count + 1.1, 1)
    #
    # Asymptotic matrices
    #
    asymptotic_inverse_projection, plottable_objects = get_empirical_asymptotic(signal_names, projector)
    # Plot individual matrices
    plot_empirical_asymptotic_matrices(plottable_objects, plotbasepath=plotbasepath)
    # Plot the grid
    plot_empirical_asymptotic_matrices(plottable_objects, as_grid=True, plotbasepath=plotbasepath)

    user_empirical_conditionals = []
    user_empirical_conditionals_grouped = []

    for userid in userids:
        print(":: User ID", userid)
        signal = load_dataset(userid)
        coefficients = tvprocess.process(signal)
        projected_coefficients = projector.project_sequence(coefficients)
        
        # All labels
        labels = movie_labels[: len(coefficients)] #Delete the last parts lost with the time-varying
        empirical_conditional = conditionals.EmpiricalConditionals(bins=empirical_conditional_bins)
        empirical_conditional.fit(projected_coefficients, labels, names)
        plot_empirical_distributions(userid, empirical_conditional, orders=[
            ("background", order_background, "{codesubject}_histogram_distance"),
            ("locutors_vs_silence", locutors_vs_silence, "{codesubject}_histogram_distance"),
        ], plotbasepath=plotbasepath)
        user_empirical_conditionals.append((userid, empirical_conditional))

        # Print empirical distributions
        get_plot_empirical_distributions(userid, convergence_matrices_users, empirical_conditional, asymptotic_inverse_projection, signal_names, plotbasepath=plotbasepath, as_pdc=True)

        # Grouped labels
        grouped_labels = grouped_movie_labels[: len(coefficients)] #Delete the last parts lost with the time-varying
        empirical_conditional_grouped = conditionals.EmpiricalConditionals(bins=empirical_conditional_bins)
        empirical_conditional_grouped.fit(projected_coefficients, grouped_labels, grouped_names)
        plot_empirical_distributions(userid, empirical_conditional_grouped, orders=[
            ("background_grouped", main_labels, "{codesubject}_histogram_matrix"),
            ("background_grouped", main_labels_rows, "{codesubject}_histogram_rows"),
        ], plotbasepath=plotbasepath)
        user_empirical_conditionals_grouped.append((userid, empirical_conditional_grouped))


    for stimulus, convergence_matrices in convergence_matrices_users.items():
        plot_subject_grid_per_stimuli(stimulus, convergence_matrices, plotbasepath=plotbasepath)
        plot_stimuli_mean_across_subjects(stimulus, convergence_matrices, signal_names, plotbasepath=plotbasepath)
    
    plot_all_conditionals(user_empirical_conditionals, rows=None, cols=4, orders=[
        ("background", order_background, "{codesubject}_histogram_distance"),
        ("locutors_vs_silence", locutors_vs_silence, "{codesubject}_histogram_distance"),
    ], plotbasepath=plotbasepath)
    
    plot_all_conditionals(user_empirical_conditionals_grouped, rows=None, cols=4, orders=[
        ("background_grouped", main_labels, "{codesubject}_histogram_matrix"),
    ], plotbasepath=plotbasepath)
    

if __name__ == "__main__":
    default_processing(n_estimators=5)

if __name__ == "__main__2":
    default_processing(n_estimators=2)
    default_processing(n_estimators=3)
    default_processing(n_estimators=4)
    default_processing(n_estimators=5)
    default_processing(n_estimators=6)
    default_processing(n_estimators=8)
    default_processing(n_estimators=9)
    default_processing(n_estimators=10)
