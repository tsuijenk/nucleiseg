import matplotlib.pyplot as pp
import numpy as np
import scipy.stats as ss
import merseg.process_image as pi
import merseg.PerformanceAnalysis as pa

from merseg.model import Centre, State, Ellipse, AppearanceLikelihood, update_params, SelectedOpts, ShapePriors

opts_all = SelectedOpts(True, True, 0.1, True, neighbourhood=10, mask_type=1)


def main(iters=1, opts=opts_all):
    np.random.seed(0)
    sp = ShapePriors(5, 20, 5, 10)
    X = 100
    Y = 100
    V = X * Y
    l = 0.0005
    N = ss.poisson.rvs(l * V)
    cells = []
    true_state = State(X, Y, sp)
    for n in range(N):
        centre = Centre(
            np.random.randint(0, X),
            np.random.randint(0, Y)
        )
        shape = Ellipse(
            np.random.uniform(5, 20),
            np.random.uniform(5, 10),
            np.random.uniform(0, 2 * np.pi)
        )
        cells.append(true_state.create_object(centre, shape))
        true_state.add_object(cells[-1])

    img = np.zeros((X, Y))
    img[true_state.foreground_mask] = 1
    fig = pp.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap="gray")
    fig.savefig("truth.png", bbox_inches="tight")

    beta = l
    log_prob_fore = np.zeros((X, Y))
    log_prob_fore[true_state.foreground_mask] = np.log(0.99)
    log_prob_fore[~true_state.foreground_mask] = np.log(0.01)
    log_prob_back = np.zeros((X, Y))
    log_prob_back[true_state.foreground_mask] = np.log(0.01)
    log_prob_back[~true_state.foreground_mask] = np.log(0.99)
    log_like = AppearanceLikelihood(log_prob_back, log_prob_fore)
    state = State(X, Y, sp)

    img = np.zeros((X, Y))
    img[state.foreground_mask] = 1
    fig = pp.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap="gray")
    fig.savefig("init.png", bbox_inches="tight")

    objs, inferred_state = update_params(beta, log_like, state, opts, iters=iters)

    conv_img = np.zeros((X, Y, 3), dtype=np.uint8)
    conv_img[true_state.foreground_mask] = [255, 255, 255]

    pi.draw_ellipse_overlay_synthetic(objs, conv_img)

    img = np.zeros((X, Y))
    img[state.foreground_mask] = 1
    fig = pp.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap="gray")
    fig.savefig("infer.png", bbox_inches="tight")

    print(true_state.num_objects, state.num_objects)

    print("Rand Index: " + str(pa.segmentation_accuracy(true_state.foreground_mask, state.foreground_mask)))
    print("Segmentation Accuracy " + str(pa.segmentation_accuracy(true_state.foreground_mask, state.foreground_mask)))


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

    opts_no_neighbour = SelectedOpts(True, True, 0.1, False, neighbourhood=1, mask_type=1)

    # main(iters=500)

    import line_profiler

    from merseg.model import update_z, State, AppearanceLikelihood

    # profiler = line_profiler.LineProfiler(
    #     update_params,
    #     update_z,
    #     State.add_object,
    #     State.remove_object,
    #     State._clip_region,
    #     AppearanceLikelihood.log_p
    # )

    profiler = line_profiler.LineProfiler(
        update_params
    )

    profiler.run("main(iters=500)")

    profiler.print_stats()
