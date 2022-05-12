from collections import defaultdict

import numpy as np
import random
import skimage.draw
from numba import njit

from merseg.math_utils import discrete_rvs, log_normalize, log_poisson_pdf
import matplotlib.pyplot as pp


class AdaptVariance(object):
    def __init__(self):
        self.Ji = 5
        self.batch_size = 1
        self.major_var = 0
        self.minor_var = 0
        self.rot_var = 0
        self.all_dim_var = 0
        self.major_accept_rate = np.zeros(3)  # proposals, accepts, rate
        self.minor_accept_rate = np.zeros(3)
        self.rot_accept_rate = np.zeros(3)
        self.all_dim_accept_rate = np.zeros(3)

    def update_major(self, proposals, accept):
        self.major_accept_rate[0] += proposals
        self.major_accept_rate[1] += accept

    def update_minor(self, proposals, accept):
        self.minor_accept_rate[0] += proposals
        self.minor_accept_rate[1] += accept

    def update_rot(self, proposals, accept):
        self.rot_accept_rate[0] += proposals
        self.rot_accept_rate[1] += accept

    def update_all_dim(self, proposals, accept):
        self.all_dim_accept_rate[0] += proposals
        self.all_dim_accept_rate[1] += accept

    def check_rates(self, aspect):

        if aspect == "major":
            self.major_accept_rate[2] = self.major_accept_rate[1] / self.major_accept_rate[0]
            if self._batch_check(self.major_accept_rate[0]):
                self.major_var = self._check_for_updating(self.major_accept_rate[2], self.major_var)
        elif aspect == "minor":
            self.minor_accept_rate[2] = self.minor_accept_rate[1] / self.minor_accept_rate[0]
            if self._batch_check(self.minor_accept_rate[0]):
                self.minor_var = self._check_for_updating(self.minor_accept_rate[2], self.minor_var)
        elif aspect == "rot":
            self.rot_accept_rate[2] = self.rot_accept_rate[1] / self.rot_accept_rate[0]
            if self._batch_check(self.rot_accept_rate[0]):
                self.rot_var = self._check_for_updating(self.rot_accept_rate[2], self.rot_var)
        elif aspect == "all":
            self.all_dim_accept_rate[2] = self.all_dim_accept_rate[1] / self.all_dim_accept_rate[0]
            if self._batch_check(self.all_dim_accept_rate[0]):
                self.all_dim_var = self._check_for_updating_all_dim(self.all_dim_accept_rate[2], self.all_dim_var)

    def _batch_check(self, val):
        return np.floor(val / self.batch_size) % self.Ji == 0

    def _check_for_updating(self, val, var):
        delta_n = min(0.01, self.Ji**(-1/2))
        if val <= 0.435:
            # adapt down
            var -= delta_n
        elif val >= 0.445:
            # adapt up
            var += delta_n
        return var

    def _check_for_updating_all_dim(self, val, var):
        delta_n = min(0.01, self.Ji**(-1/2))
        var_1 = 0
        if val <= 0.23:
            # adapt down
            var_1 -= delta_n
        elif val >= 0.24:
            # adapt up
            var_1 += delta_n

        self.major_var += var_1
        self.minor_var += var_1
        self.rot_var += var_1
        return var + var_1

    # def _check_for_updating_all_dim(self, val, var):
    #     delta_n = min(0.01, self.Ji**(-1/2))
    #     if val <= 0.435:
    #         # adapt down
    #         var -= delta_n
    #     elif val >= 0.445:
    #         # adapt up
    #         var += delta_n
    #
    #     return var


class SelectedOpts(object):
    def __init__(self, mask, sample, sample_size, move_centres, neighbourhood=1, mask_type=1):
        self.mask = mask  # bool
        self.mask_type = mask_type  # int, 1 for single, 2 for double mask
        self.sample = sample  # bool
        self.sample_size = sample_size  # float from 0 to 1.0
        self.move_centres = move_centres  # bool
        self.neighbourhood = neighbourhood  # int

    def mask_test(self, state, x, y):
        if self.mask_type == 1:
            return not state.invground[x, y] > 0
        elif self.mask_type == 2:
            return not state.invground[x, y] > 0 or state.foreground[x, y] > 0
        else:
            return not state.invground[x, y] > 0 and not state.foreground[x, y] > 0


class ShapePriors(object):
    def __init__(self, major_prior_min, major_prior_max, minor_prior_min, minor_prior_max, rot_min=0):
        self.major_prior_min = major_prior_min
        self.major_prior_max = major_prior_max
        self.minor_prior_min = minor_prior_min
        self.minor_prior_max = minor_prior_max
        self.rot_min = rot_min

    def sample_major_prior(self):
        return np.random.uniform(self.major_prior_min, self.major_prior_max)

    def sample_minor_prior(self):
        return np.random.uniform(self.minor_prior_min, self.minor_prior_max)

    def sample_rot_prior(self):
        return np.random.uniform(self.rot_min, 2 * np.pi)


def show_img(state):
    img = np.zeros((state.x, state.y))
    img[state.foreground_mask] = 1
    fig = pp.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap="gray")
    fig.show()


def update_params(beta, log_like, state, opts, iters=1):
    x_1, y_1 = np.mgrid[:state.x:1, :state.y:1]
    positions = np.vstack([x_1.ravel(), y_1.ravel()]).T
    coords = positions

    rng = np.random.default_rng(seed=0)
    log_p_curr = log_like.log_p(state)

    for i in range(iters):
        if i % 10 == 0:
            print(f'iteration {i}: detected {state.num_objects} nuclei')
            show_img(state)

        if opts.sample:
            sample_size = int((state.x * state.y) * opts.sample_size)
            coords = rng.choice(positions, size=sample_size, replace=False)

        rng.shuffle(coords)

        for j in range(len(coords)):
            x = coords[j][0]
            y = coords[j][1]
            if opts.mask:
                if opts.mask_test(state, x, y):
                    log_p_curr = update_z(beta, log_like, x, y, state, opts, log_p_curr)
            else:
                log_p_curr = update_z(beta, log_like, x, y, state, opts, log_p_curr)

        state.reset_invground()
        shape_update = rng.choice(["step", "3D"], size=1, p=[.7, .3])[0]

        if shape_update == "step":
            log_p_curr = update_s_dim(log_like, state, log_p_curr)
        else:
            log_p_curr = update_s(log_like, state, log_p_curr)

    show_img(state)
    return state.objects_list, state


def plot_accept_rate(x):
    new_fig = pp.figure()
    ax = pp.axes()
    ax.plot(x)
    new_fig.show()


def update_shape_on_mask(state, obj, shape, log_p_curr, log_like):
    idxs, c = state.remove_object_foreground(obj)
    rem_log_p = log_like.log_p_shape_change(state, log_p_curr, idxs, c)
    obj.shape = shape
    obj.shape.changed = True
    idxs, c = state.add_object_foreground(obj)
    return log_like.log_p_shape_change(state, rem_log_p, idxs, c)


def update_shape(log_like, new_shape, obj, state, log_p_curr):
    old_shape1 = obj.shape
    log_p_old = log_p_curr
    log_p_new = update_shape_on_mask(state, obj, new_shape, log_p_curr, log_like)
    u = np.random.uniform()
    accept = 1
    if np.log(u) > log_p_new - log_p_old:
        log_p_curr = update_shape_on_mask(state, obj, old_shape1, log_p_new, log_like)
        accept = 0
    return log_p_curr, accept


def update_s(log_like, state, log_p_curr, iters=1):

    for i, obj in enumerate(state.objects_list):
        for _ in range(iters):
            new_shape = Ellipse(
                obj.shape.major + np.random.normal(0, np.exp(2 * obj.adapt.major_var)),
                obj.shape.minor + np.random.normal(0, np.exp(2 * obj.adapt.minor_var)),
                obj.shape.rotation + np.random.normal(0, np.exp(2 * obj.adapt.rot_var))
            )
            log_p_curr, iter_acpt = update_shape(log_like, new_shape, obj, state, log_p_curr)
            obj.adapt.update_all_dim(1, iter_acpt)
        obj.adapt.check_rates("all")

    return log_p_curr


def update_s_dim(log_like, state, log_p_curr, iters=1):

    for obj in state.objects_list:
        for _ in range(iters):
            # Major
            old_shape = obj.shape
            new_shape = Ellipse(
                old_shape.major + np.random.normal(0, np.exp(2 * obj.adapt.major_var)),
                old_shape.minor,
                old_shape.rotation
            )
            log_p_curr, iter_acpt = update_shape(log_like, new_shape, obj, state, log_p_curr)
            obj.adapt.update_major(1, iter_acpt)
            obj.adapt.check_rates("major")

            # Minor
            old_shape = obj.shape
            new_shape = Ellipse(
                old_shape.major,
                old_shape.minor + np.random.normal(0, np.exp(2 * obj.adapt.minor_var)),
                old_shape.rotation
            )
            log_p_curr, iter_acpt = update_shape(log_like, new_shape, obj, state, log_p_curr)
            obj.adapt.update_minor(1, iter_acpt)
            obj.adapt.check_rates("minor")

            # Rotation
            old_shape = obj.shape
            new_shape = Ellipse(
                old_shape.major,
                old_shape.minor,
                old_shape.rotation + np.random.normal(0, np.exp(2 * obj.adapt.rot_var))
            )
            log_p_curr, iter_acpt = update_shape(log_like, new_shape, obj, state, log_p_curr)
            obj.adapt.update_rot(1, iter_acpt)
            obj.adapt.check_rates("rot")

    return log_p_curr


def update_z(beta, log_like, x, y, state, opts, log_p_orig):
    sp = state.shape_prior
    log_p = np.zeros(3)
    log_p_rem = 0
    rem_obj = None
    log_p_curr = log_p_orig

    if len(state.objects[(x, y)]) > 0:
        rem_obj = random.choice(state.objects[(x, y)])
        idxs, c = state.remove_object(rem_obj)
        log_p_rem = log_like.log_p_shape_change(state, log_p_orig, idxs, c)
        log_p[0] = log_poisson_pdf(len(state.objects[(x, y)]), beta / state.D) + log_p_rem
        state.add_object(rem_obj)
    else:
        log_p[0] = -np.inf
        state.objects.pop((x, y))  # remove empty key

    log_p[1] = log_poisson_pdf(len(state.objects[(x, y)]), beta / state.D) + log_p_curr

    centre = Centre(
        x,
        y
    )
    shape = Ellipse(
        sp.sample_major_prior(),
        sp.sample_minor_prior(),
        sp.sample_rot_prior()
    )

    add_obj = state.create_object(centre, shape)
    idxs, c = state.add_object(add_obj)
    log_p_add = log_like.log_p_shape_change(state, log_p_curr, idxs, c)
    log_p[2] = log_poisson_pdf(len(state.objects[(x, y)]), beta / state.D) + log_p_add
    state.remove_object(add_obj)

    log_p = log_normalize(log_p)

    try:
        idx = discrete_rvs(np.exp(log_p))

    except ValueError:
        return log_p_curr

    if idx == 0:
        state.remove_object(rem_obj)
        return log_p_rem

    elif idx == 2:
        if opts.move_centres:
            log_p_add = check_neighbourhood(state, add_obj, log_like, opts.neighbourhood, log_p_curr)
        state.add_object(add_obj)
        return log_p_add

    elif idx == 1:
        state.add_object_inv(add_obj)
        return log_p_curr


def check_neighbourhood(state, obj, log_like, neighbourhood_layers, orig_log_p):
    idxs_orig, counter_orig = state.add_object_foreground(obj)
    orig_centre = obj.centre.to_tuple()
    best_centre = obj.centre.to_tuple()
    best_log_p = log_like.log_p_shape_change(state, orig_log_p, idxs_orig, counter_orig)

    state.remove_object_foreground(obj)
    cardinal_coords = np.array([[-1, 1], [0, 1], [1, 1], [-1, 0], [1, 0], [-1, -1], [0, -1], [1, -1]])
    for i in range(1, neighbourhood_layers + 1):
        for j in range(8):
            x_shift = cardinal_coords[j][0] * i
            y_shift = cardinal_coords[j][1] * i
            idxs, count = state.add_object_foreground_shift(obj, x_shift, y_shift)

            log_p_curr = log_like.log_p_shape_change(state, orig_log_p, idxs, count)

            state.remove_object_foreground_shift(obj, x_shift, y_shift)
            u = np.random.uniform()
            if not (np.log(u) > log_p_curr - best_log_p):
                best_log_p = log_p_curr
                best_centre = (orig_centre[0] + x_shift, orig_centre[1] + y_shift)

    if best_centre != orig_centre:
        obj.centre.x = best_centre[0]
        obj.centre.y = best_centre[1]
        obj.shape.changed = True

    return best_log_p


class Ellipse(object):

    def __init__(self, major, minor, rotation):
        self.major = major
        self.minor = minor
        self.rotation = rotation

    @property
    def major(self):
        return self._major

    @major.setter
    def major(self, x):
        self.changed = True
        self._major = x

    @property
    def minor(self):
        return self._minor

    @minor.setter
    def minor(self, x):
        self.changed = True
        self._minor = x

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, x):
        self.changed = True
        self._rotation = x


class Centre(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_tuple(self):
        return self.x, self.y


class Cell(object):

    def __init__(self, centre, shape, obj_id=None):
        self.centre = centre
        self.shape = shape
        self.id = obj_id
        self._region = skimage.draw.ellipse(
            self.centre.x,
            self.centre.y,
            self.shape.minor,
            self.shape.major,
            rotation=self.shape.rotation
        )
        self.shape.changed = False
        self.adapt = AdaptVariance()

    @property
    def region(self):
        if self.shape.changed:
            self._region = skimage.draw.ellipse(
                self.centre.x,
                self.centre.y,
                self.shape.minor,
                self.shape.major,
                rotation=self.shape.rotation
            )
            self.shape.changed = False

        return self._region


@njit
def update_foreground_track_idx_change(foreground, idxs, change_type):
    reg_size = len(idxs[0])
    changed_idxs = np.zeros((reg_size, 2), dtype=np.intc)
    counter = 0

    for i in range(reg_size):
        x = idxs[0][i]
        y = idxs[1][i]
        prev = foreground[x, y] > 0
        # now update foreground
        foreground[x, y] += change_type
        curr = foreground[x, y] > 0
        if curr != prev:
            # index has changed fore/back mode, add it to tracking arr and incr counter
            changed_idxs[counter] = np.array([x, y])
            counter += 1
    return changed_idxs, counter


class State(object):

    def __init__(self, x, y, shape_prior):
        self.objects = defaultdict(list)
        self.foreground = np.zeros((x, y), dtype=int)
        self.invground = np.zeros((x, y), dtype=int)
        self.x = x
        self.y = y
        self._obj_count = 0
        self.shape_prior = shape_prior

    @property
    def D(self):
        return self.x * self.y

    @property
    def foreground_mask(self):
        return self.foreground > 0

    @property
    def invground_mask(self):
        return self.invground > 0

    @property
    def num_objects(self):
        return sum([len(x) for x in self.objects.values()])

    @property
    def objects_list(self):
        result = []
        for x in self.objects.values():
            result.extend(x)
        return result

    def create_object(self, centre, shape):
        obj = Cell(centre, shape, obj_id=self._obj_count)
        self._obj_count += 1
        return obj

    def add_object(self, obj):
        self.objects[obj.centre.to_tuple()].append(obj)
        changed_idxs, counter = self.add_object_foreground(obj)
        return changed_idxs, counter

    def remove_object(self, obj):
        size = len(self.objects[obj.centre.to_tuple()])
        self.objects[obj.centre.to_tuple()].remove(obj)
        if size == 1:
            self.objects.pop(obj.centre.to_tuple())
        changed_idxs, counter = self.remove_object_foreground(obj)
        return changed_idxs, counter

    def add_object_foreground(self, obj):
        idxs = self._clip_region(obj.region)
        changed_idxs, counter = update_foreground_track_idx_change(self.foreground, idxs, 1)
        return changed_idxs, counter

    def remove_object_foreground(self, obj): #, naccept
        idxs = self._clip_region(obj.region)
        changed_idxs, counter = update_foreground_track_idx_change(self.foreground, idxs, -1)
        return changed_idxs, counter

    def add_object_inv(self, obj):
        self.invground[self._clip_region(obj.region)] += 1

    def remove_object_inv(self, obj):
        self.invground[self._clip_region(obj.region)] -= 1

    def reset_invground(self):
        self.invground = np.zeros((self.x, self.y), dtype=int)

    def _clip_region(self, region):
        x, y = region
        idxs = (x >= 0) & (x < self.x) & (y > 0) & (y < self.y)
        return x[idxs], y[idxs]

    def _clip_region_shift(self, region, x_shift, y_shift):
        x, y = region
        x = x + x_shift
        y = y + y_shift
        idxs = (x >= 0) & (x < self.x) & (y > 0) & (y < self.y)
        return x[idxs], y[idxs]

    def add_object_foreground_shift(self, obj, x_shift, y_shift):
        idxs = self._clip_region_shift(obj.region, x_shift, y_shift)
        changed_idxs, counter = update_foreground_track_idx_change(self.foreground, idxs, 1)
        return changed_idxs, counter

    def remove_object_foreground_shift(self, obj, x_shift, y_shift):
        self.foreground[self._clip_region_shift(obj.region, x_shift, y_shift)] -= 1


class AppearanceLikelihood(object):

    def __init__(self, log_prob_back, log_prob_fore):
        self.log_prob_back = log_prob_back
        self.log_prob_fore = log_prob_fore

    def log_p(self, state):
        return _log_p_appearance(self.log_prob_back, self.log_prob_fore, state.foreground)

    def log_p_shape_change(self, state, old_log_p, idxs, counter):
        return _log_p_appearance_shape_change(self.log_prob_back, self.log_prob_fore, state.foreground, old_log_p, idxs,
                                              counter)


@njit
def _log_p_appearance_shape_change(log_back, log_fore, fore, old_log_p, idxs, counter):
    log_p = old_log_p
    for i in range(counter):
        x = idxs[i][0]
        y = idxs[i][1]
        if fore[x, y] <= 0:
            log_p -= log_fore[x, y]
            log_p += log_back[x, y]
        else:
            log_p -= log_back[x, y]
            log_p += log_fore[x, y]
    return log_p


@njit
def _log_p_appearance(log_back, log_fore, fore):
    x, y = log_back.shape
    log_p = 0
    for i in range(x):
        for j in range(y):
            if fore[i, j] <= 0:
                log_p += log_back[i, j]
            else:
                log_p += log_fore[i, j]
    return log_p
