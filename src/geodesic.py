import torch
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)

"""
    This function will contain all of the geometric code
    including the control points, spline, derivatives etc.
"""

class SphericalCubicSpline():
    def __init__(self, t_points, nodes):

        # def update( self, t_points, nodes):
        assert t_points.shape[0] == nodes.shape[0]

        # Control points (to parameterize the interpolation) and nodes (points on the path)
        self.t_points = t_points
        self.nodes = nodes

        # computes the angles between two consecutive points
        cos_thetas = torch.sum(nodes[1:]* nodes[:-1], dim=-1)/(torch.norm(nodes[1:], dim=-1) * torch.norm(nodes[:-1], dim=-1))
        cos_thetas = torch.clip(cos_thetas, -1, 1)

        # thetas solves the radian angles between the points
        self.thetas = torch.arccos(cos_thetas)
        self.max_t = torch.max(t_points)
        self.min_t = torch.min(t_points)

        self.cubic_coeffs = natural_cubic_spline_coeffs(t_points, nodes)
        self.cubic_euclidean = NaturalCubicSpline(self.cubic_coeffs)

    # linear interpolation between two points
    def lerp(self, t_points, x0, x1):
        return torch.cat([(1 - t) * x0 + t * x1 for t in t_points], dim=0)

    # spherical interpolation (between points on a sphere)
    def slerp(self, x0, x1, theta, a):
        # with a in [0,1], return the point on the line between x0 and x1 at the angle theta
        return (torch.sin((1-a)*theta) * x0 + torch.sin(a*theta) * x1) / torch.sin(theta)

    def slerp_d(self, x0, x1, theta, a, interval):
        # derivative of slerp
        v = (-torch.cos((1-a)*theta)*x0*theta + torch.cos(a*theta)*x1*theta) / torch.sin(theta)
        return v * (1/interval)  # for piecewise normalisation

    def slerp_dd(self, x0, x1, theta, a, interval):
        # second derivative of slerp
        a = (-torch.sin((1-a)*theta)*x0*theta**2 - torch.sin(a*theta)*x1*theta**2) / torch.sin(theta)
        return a * (1/interval)**2  # for piecewise normalisation

    def slerp_dd_cubic(self, query_points):
        # get bounding fit values and indices
        t_interval, t_indexes = self.get_interval(query_points)
        t_in_fit_mask = []

        # loop through the interval, if the upper and lower indices are the same,
        # it means the t query is equal t one of the fitted values (i.e in the t_in_fit_mask)
        for i in range(t_interval.shape[0]):
            if t_indexes[i][0] == t_indexes[i][1]:
                t_in_fit_mask.append(True)
            else:
                t_in_fit_mask.append(False)

        t_in_fit_mask = torch.tensor(t_in_fit_mask)
        cubic_dd = self.cubic_euclidean.derivative(query_points, order=2)
        return cubic_dd

    def get_interval(self, query_points):
        # return the smallest fitted t greater than given, and largest fitted t smaller than given
        if torch.sum(query_points > self.max_t) > 0 or torch.sum(query_points < self.min_t) > 0:
            raise ValueError('Query is out of range of the fitted spline')

        # Query points determine the location of interpolation along the geodesic
        query_points_exp = query_points.unsqueeze(1)
        t_points_exp = self.t_points.unsqueeze(0)

        greater_than_bool = t_points_exp >= query_points_exp
        less_than_bool = t_points_exp <= query_points_exp

        # If the element in t_points is (greater) lower than the query, replace it with (-inf) inf
        filtered_greater = torch.where(greater_than_bool, t_points_exp, float('inf'))
        filtered_lower = torch.where(less_than_bool, t_points_exp, float('-inf'))

        # Pick the minimum (maximum) of the remaining points to find the next largest (smallest) values
        t_greater = filtered_greater.min(dim=1)[0]
        t_lower = filtered_lower.max(dim=1)[0]

        idx_greater = filtered_greater.argmin(dim=1)
        idx_lower = filtered_lower.argmax(dim=1)

        t_point_interval = torch.stack([t_lower, t_greater], dim=1)
        t_point_indexes = torch.stack([idx_lower, idx_greater], dim=1)

        return t_point_interval, t_point_indexes

    def evaluate(self, query_points):
        # Get the bracketing points and indices for each query point
        t_intervals, t_indexes = self.get_interval(query_points)
        interp_points =[]

        # Loop through the queries
        for i in range(t_intervals.shape[0]):
            t_idx_pair = t_indexes[i]

            # If any pair of indices are the same, the query is already on a control point
            if t_idx_pair[0] == t_idx_pair[1]:
                interp_points.append(self.nodes[t_idx_pair[0]])

            #Otherwise, compute slerp
            else:
                t0, t1 = t_intervals[i] # e.g 0.3 and 0.4 for a specific query
                param = (query_points[i] - t0) / (t1 - t0) # distance between intervals as a fraction

                start, end = self.nodes[t_idx_pair]

                angle = self.thetas[t_idx_pair[0]]

                interp_points.append(self.slerp(start, end, angle, param))

        return torch.stack(interp_points)

    def evaluate_derivative(self, query_points, derivative_function):
        diff_points = []
        t_intervals, t_indexes = self.get_interval(query_points)

        for i in range(t_intervals.shape[0]):
            t_idx_pair = t_indexes[i]
            t0, t1 = t_intervals[i]

            # Get bracket node values and indices for the query
            x0, x1 = self.nodes[t_idx_pair]
            idx0, idx1 = t_idx_pair

            # If any pair of indices are the same, the query is already on a control point
            if idx0 == idx1:

                # If it is the max control point, calculate instantaneous derivative w.r.t a at the max point
                if t0 == self.max_t:
                    v = derivative_function(self.nodes[idx0-1], x0, self.thetas[idx0-1], 1, (t0-self.t_points[idx0-1]))
                    diff_points.append(v)

                # If it is the min control point, calculate instantaneous derivative w.r.t a at the min point
                elif t0 == self.min_t:
                    v = derivative_function(x1, self.nodes[idx1+1], self.thetas[idx0], 0, (self.t_points[idx1+1]-t1))
                    diff_points.append(v)

                # Otherwise, we compute the derivative as a sum of terms from the left and the right nodes
                else:
                    v = 0.5*derivative_function(self.nodes[idx0-1], x0, self.thetas[idx0-1], 1, (t0-self.t_points[idx0-1])) \
                        + 0.5*derivative_function(x1, self.nodes[idx1+1], self.thetas[idx0], 0, (self.t_points[idx1+1]-t1))
                    diff_points.append(v)
            else:
                a = (query_points[i] - t0) / (t1 - t0)
                diff_points.append(derivative_function(x0, x1, self.thetas[idx0], a, (t1-t0)))

        return torch.stack(diff_points)

    def __call__(self, query_points, order=0):
        match order:
            case 0:
                return self.evaluate(query_points)
            case 1:
                return self.evaluate_derivative(query_points, self.slerp_d)
            case 2:
                return self.slerp_dd_cubic(query_points)
            case _:
                raise RuntimeError(f"Spline error: {order}-derivative not implemented!")

# To sample the points on the path to optimise
class BisectionSampler():
    def __init__(
        self,
        device,
        max_strength=3,  #'strength' corresponds to the number of control points (strength/precision of interpolation)
        bisect_interval=100,
        only_new_points=False,
        ):
        control_dict = {}
        middle_num = 0

        # course-to-fine optimisation (the number of control points increases throughout the optimsation)
        for i in range(1, max_strength + 1):
            middle_num += (middle_num + 1)
            t = torch.linspace(0, 1, middle_num + 2)
            control_dict[i] = t[1: -1]

        #control_dict {1: [0.5000],
        #              2: [0.2500, 0.5000, 0.7500],
        #              3: [0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750]}

        if only_new_points:
        # if this is true, the path only optimise the new inserted points
        # for example, after optimising [0.5], it will only optimise [0.25 , 0.75] not [0.25, 0.5, 0.75]
            control_dict_update = {}
            for i in range(1, max_strength+1):
                if i == 1:
                    control_dict_update[i] = control_dict[i]
                else:
                    t = control_dict[i]
                    t_prev = control_dict[i-1]
                    new_p = t[torch.isin(t, t_prev, invert=True)]
                    control_dict_update[i] = new_p
            control_dict = control_dict_update

        self.max_strength = max_strength
        self.control_dict = control_dict
        self.cur_strength = 1
        self.bisect_interval = bisect_interval

    def add_strength(self, it):
        if it is None:
            self.cur_strength += 1 # directly add strength without condition
        elif it >= self.cur_strength * self.bisect_interval:
            self.cur_strength += 1

    def get_query_points(self):
        # if the current strength is less than the max strength, return the control points for that level
        if self.cur_strength <= self.max_strength:
            return self.control_dict[self.cur_strength]
        else:
            return None

# Geometric functions
def norm_fix(tensor, norm_fix):
    # maintain the tensor norm as norm_fix
    return norm_fix * tensor / torch.norm(tensor)

def norm_fix_batch(batch, norm_fix):
    # maintain the b tensor xs norm as m
    if isinstance(norm_fix, float):
        norm_fix = torch.tensor([norm_fix] * batch.shape[0]).to(batch.device)
    return norm_fix[:,None] * batch / torch.norm(batch, dim=-1)[:, None]

def o_project(x, y):
    # project torch vector x onto the orthogonal complement of y
    y_hat = y / torch.norm(y)
    return x - torch.dot(x, y_hat) * y_hat

def o_project_batch(xs, ys):
    # project numpy batch vector x onto the orthogonal complement of y
    ys_norm = torch.norm(ys, dim=-1)
    ys_hat = ys / ys_norm[:, None]
    return xs - torch.sum(xs * ys_hat, dim=-1)[:,None] * ys_hat



