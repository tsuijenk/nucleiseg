import matplotlib.pyplot as pp
import numpy as np
import skimage.io as skio
import os

from merseg.model import State, AppearanceLikelihood, update_params, SelectedOpts, ShapePriors
import merseg.process_image as pi
import merseg.PerformanceAnalysis as pa

opts_all = SelectedOpts(True, True, 0.1, True, neighbourhood=20, mask_type=3)


def main(fn, sp, iters=1, opts=opts_all):
    np.random.seed(0)
    p_i = pi.ProcessImage()

    fore_probs, back_probs = p_i.compute_probs(fn)

    X = p_i.X
    Y = p_i.Y
    l = 0.0005
    l = l / 10000

    true_state = State(X, Y, sp)
    true_state.foreground = p_i.produce_binary_img(fn)
    save_image(X, Y, true_state, "truth.TIFF")

    beta = l
    log_like = AppearanceLikelihood(fore_probs, back_probs)
    state = State(X, Y, sp)

    save_image(X, Y, state, "init.TIFF")

    objs, inferred_state = update_params(beta, log_like, state, opts, iters=iters)

    p_i.draw_ellipse_overlay(objs)

    save_image(X, Y, state, "infer.TIFF")

    print(true_state.num_objects, state.num_objects)

    print("Rand Index: " + str(pa.rand_index(true_state.foreground_mask, state.foreground_mask)))
    print("Segmentation Accuracy " + str(pa.segmentation_accuracy(true_state.foreground_mask, state.foreground_mask)))


def save_image(X, Y, state, name):
    
    img = np.zeros((X, Y))
    
    img[state.foreground_mask] = 1
    
    skio.imsave(os.path.join(os.getcwd(), 'output', name), img, check_contrast = False)



if __name__ == "__main__":

    # SelectedOpts signature:
    # SelectedOpts((bool) mask, (bool) sample, (float 0-1) sample_size, (bool) move_centres,
    # (optional, int) neighbourhood=1, (optional, int) mask_type=1)

    opts_none = SelectedOpts(False, False, 1.0, False)

    opts_mask_only = SelectedOpts(True, False, 1.0, False, neighbourhood=0, mask_type=3)

    opts_sample_only = SelectedOpts(False, True, 0.1, False, neighbourhood=0, mask_type=3)

    opts_centres_only = SelectedOpts(False, False, 1.0, True, neighbourhood=1, mask_type=3)

    opts_centres_mask = SelectedOpts(True, False, 1.0, True, neighbourhood=1, mask_type=3)

    opts_centres_sample = SelectedOpts(False, True, 0.1, True, neighbourhood=1, mask_type=3)

    opts_mask_sample = SelectedOpts(True, True, 0.1, False, neighbourhood=1, mask_type=3)

    filename = os.path.join(os.getcwd(), 'data', 'clahe_nuclei.jpg')
    
    # filename = "../data/merFISH_01_001_05 .TIFF"
    # filename_nuclei = "../data/dna-34.png"
    # filename = "../data/merFISH_01_003_07.TIFF"

    s_p = ShapePriors(25, 80, 25, 60)
    
    s_p_nuclei = ShapePriors(30, 90, 30, 80)
    
    #main(filename, s_p, iters=500)

    # main(filename_nuclei, s_p_nuclei, iters=100)

    main(filename, s_p, iters=200, opts=opts_mask_sample)

    #import line_profiler

    #profiler = line_profiler.LineProfiler(
    #    update_params,
    #)

    #profiler.run("main(filename, s_p, iters=100)")

    #profiler.print_stats()
