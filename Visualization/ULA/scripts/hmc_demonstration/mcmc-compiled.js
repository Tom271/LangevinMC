"use strict";

var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var MCSampler = function () {
    // both energy and position accept one argument - vector.
    // in the case of demonstration vector can have only one or two variables.

    function MCSampler(distribution, x_start, y_start) {
        _classCallCheck(this, MCSampler);

        this.distribution = distribution;
        this.positions = [];
        this.positions.push([x_start, y_start]);
        this.T = 1;
        this.random = new RandomGenerator(42);
    }

    _createClass(MCSampler, [{
        key: "energy",
        value: function energy(x) {
            return this.distribution.energy(x);
        }
    }, {
        key: "grad",
        value: function grad(x) {
            return this.distribution.gradient(x);
        }
    }, {
        key: "set_temperature",
        value: function set_temperature(T) {
            this.T = T;
        }
    }, {
        key: "set_algo",
        value: function set_algo(algo) {
            this.algo = algo;
        }
    }, {
        key: "to_3d_point",
        value: function to_3d_point(position) {
            return [position[0], position[1], this.energy(position)];
        }

        // metropolis - hastings

    }, {
       /// meat
        key: "generate_mh",
        value: function generate_mh() {
          if (this.algo == "ULA") {
                  var spread = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0.1;
                  var sqrtspread = Math.sqrt(2*spread);

                  var position_old = this.positions[this.positions.length - 1];
                  var g = this.grad(position_old);
                  var position_new = [position_old[0] - spread * g[0] / this.T + sqrtspread * this.random.random_normal(0, 1), position_old[1]  - spread * g[1] / this.T + sqrtspread * this.random.random_normal(0, 1)];

                  var result = void 0;
                  result = clone(position_new);

                  this.positions.push(result);

                  // return final position, candidate, and reject solution
                  return [this.to_3d_point(result), this.to_3d_point(position_new), false];
          }

          if (this.algo == "tULA") {
                  var spread = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0.1;
                  var sqrtspread = Math.sqrt(2*spread);

                  var position_old = this.positions[this.positions.length - 1];

                  // tamed gradient
                  var g = this.grad(position_old);
                  var normg = Math.sqrt(Math.pow(g[0], 2) + Math.pow(g[1], 2));
                  g = [g[0]/(1.0 + spread*normg), g[1]/(1.0 + spread*normg)];


                  var position_new = [position_old[0] - spread * g[0] / this.T + sqrtspread * this.random.random_normal(0, 1), position_old[1]  - spread * g[1] / this.T + sqrtspread * this.random.random_normal(0, 1)];

                  var result = void 0;
                  result = clone(position_new);

                  this.positions.push(result);

                  // return final position, candidate, and reject solution
                  return [this.to_3d_point(result), this.to_3d_point(position_new), false];
          }


          if (this.algo == "tULAc") {
                  var spread = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0.1;
                  var sqrtspread = Math.sqrt(2*spread);

                  var position_old = this.positions[this.positions.length - 1];

                  // tamed gradient
                  var g = this.grad(position_old);
                  g = [g[0]/(1.0 + spread*Math.abs(g[0])), g[1]/(1.0 + spread*Math.abs(g[1]))];

                  var position_new = [position_old[0] - spread * g[0] / this.T + sqrtspread * this.random.random_normal(0, 1), position_old[1]  - spread * g[1] / this.T + sqrtspread * this.random.random_normal(0, 1)];

                  var result = void 0;
                  result = clone(position_new);

                  this.positions.push(result);

                  // return final position, candidate, and reject solution
                  return [this.to_3d_point(result), this.to_3d_point(position_new), false];
          }


          if (this.algo == "LM") {
                  if (this.old_gaussian == undefined) {
                    this.old_gaussian = [this.random.random_normal(0, 1), this.random.random_normal(0, 1)];
                  }

                  var spread = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0.1;
                  var sqrtspread = Math.sqrt(0.5 * spread);

                  var position_old = this.positions[this.positions.length - 1];
                  var g = this.grad(position_old);
                  var new_gaussian = [this.random.random_normal(0, 1), this.random.random_normal(0, 1)];
                  var position_new = [position_old[0] - spread * g[0] / this.T + sqrtspread * (new_gaussian[0] + this.old_gaussian[0]), position_old[1]  - spread * g[1] / this.T + sqrtspread * (new_gaussian[1] + this.old_gaussian[1])];
                  this.old_gaussian = clone(new_gaussian);

                  var result = void 0;
                  result = clone(position_new);

                  this.positions.push(result);

                  // return final position, candidate, and reject solution
                  return [this.to_3d_point(result), this.to_3d_point(position_new), false];
          }

          if (this.algo == "tLM") {
                  if (this.old_gaussian == undefined) {
                    this.old_gaussian = [this.random.random_normal(0, 1), this.random.random_normal(0, 1)];
                  }

                  var spread = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0.1;
                  var sqrtspread = Math.sqrt(0.5 * spread);

                  var position_old = this.positions[this.positions.length - 1];
                  var g = this.grad(position_old);
                  var normg = Math.sqrt(Math.pow(g[0], 2) + Math.pow(g[1], 2));
                  g = [g[0]/(1.0 + spread*normg), g[1]/(1.0 + spread*normg)];
                  var new_gaussian = [this.random.random_normal(0, 1), this.random.random_normal(0, 1)];
                  var position_new = [position_old[0] - spread * g[0] / this.T + sqrtspread * (new_gaussian[0] + this.old_gaussian[0]), position_old[1]  - spread * g[1] / this.T + sqrtspread * (new_gaussian[1] + this.old_gaussian[1])];
                  this.old_gaussian = clone(new_gaussian);

                  var result = void 0;
                  result = clone(position_new);

                  this.positions.push(result);

                  // return final position, candidate, and reject solution
                  return [this.to_3d_point(result), this.to_3d_point(position_new), false];
          }

          if (this.algo == "RWM") {
                  var spread = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0.1;
                  var sqrtspread = Math.sqrt(2*spread);

                  var position_old = this.positions[this.positions.length - 1];
                  var position_new = [position_old[0]  + sqrtspread * this.random.random_normal(0, 1), position_old[1] + sqrtspread * this.random.random_normal(0, 1)];

                  var energy_old = this.energy(position_old);
                  var energy_new = this.energy(position_new);
                  var reject = this.random.random() > Math.exp((energy_old - energy_new) / this.T);

                  if (reject) {
                      result = clone(position_old);
                  } else {
                      result = clone(position_new);
                  }
                  this.positions.push(result);

                  // return final position, candidate, and reject solution
                  return [this.to_3d_point(result), this.to_3d_point(position_new), reject];
          }


          if (this.algo == "MALA") {

                  var spread = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0.1;
                  var sqrtspread = Math.sqrt(2*spread);

                  var position_old = this.positions[this.positions.length - 1];
                  var gx = this.grad(position_old);
                  var position_new = [position_old[0] - spread*gx[0]/this.T + sqrtspread * this.random.random_normal(0, 1),
                                      position_old[1] - spread*gx[1]/this.T + sqrtspread * this.random.random_normal(0, 1)];

                  var gy = this.grad(position_new);


                  var energy_old = this.energy(position_old);
                  var energy_new = this.energy(position_new);
                  var reject = this.random.random() > Math.exp((energy_old - energy_new
                                                      + 0.25*(
                                                        Math.pow(position_new[0] - position_old[0] + spread*gx[0], 2) + Math.pow(position_new[1] - position_old[1] + spread*gx[1], 2)
                                                        - Math.pow(position_old[0] - position_new[0] + spread*gy[0], 2) - Math.pow(position_old[1] - position_new[1] + spread*gy[1], 2)
                                                      )/spread  ) / this.T);

                  if (reject) {
                      result = clone(position_old);
                  } else {
                      result = clone(position_new);
                  }
                  this.positions.push(result);

                  // return final position, candidate, and reject solution
                  return [this.to_3d_point(result), this.to_3d_point(position_new), reject];
          }


          if (this.algo == "tMALA") {

                  var spread = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0.1;
                  var sqrtspread = Math.sqrt(2*spread);

                  var position_old = this.positions[this.positions.length - 1];
                  var gx = this.grad(position_old);
                  var normgx = Math.sqrt(Math.pow(gx[0], 2) + Math.pow(gx[1], 2));
                  gx = [gx[0]/(1.0 + spread*normgx), gx[1]/(1.0 + spread*normgx)];
                  var position_new = [position_old[0] - spread*gx[0]/this.T + sqrtspread * this.random.random_normal(0, 1),
                                      position_old[1] - spread*gx[1]/this.T + sqrtspread * this.random.random_normal(0, 1)];

                  var gy = this.grad(position_new);
                  var normgy = Math.sqrt(Math.pow(gy[0], 2) + Math.pow(gy[1], 2));
                  gy = [gy[0]/(1.0 + spread*normgy), gy[1]/(1.0 + spread*normgy)];

                  var energy_old = this.energy(position_old);
                  var energy_new = this.energy(position_new);
                  var reject = this.random.random() > Math.exp((energy_old - energy_new
                                                      + 0.25*(
                                                        Math.pow(position_new[0] - position_old[0] + spread*gx[0], 2) + Math.pow(position_new[1] - position_old[1] + spread*gx[1], 2)
                                                        - Math.pow(position_old[0] - position_new[0] + spread*gy[0], 2) - Math.pow(position_old[1] - position_new[1] + spread*gy[1], 2)
                                                      )/spread  ) / this.T);

                  if (reject) {
                      result = clone(position_old);
                  } else {
                      result = clone(position_new);
                  }
                  this.positions.push(result);

                  // return final position, candidate, and reject solution
                  return [this.to_3d_point(result), this.to_3d_point(position_new), reject];
          }



        }
    }, {
        key: "leapfrog_step",
        value: function leapfrog_step(position, momentum, epsilon) {
            var iteration_alpha = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : 1.;
            var mass = arguments.length > 4 && arguments[4] !== undefined ? arguments[4] : 1.;

            momentum = VectorUtils.multiply(momentum, Math.sqrt(iteration_alpha));
            position = VectorUtils.add_with_coeffs(position, momentum, 1., epsilon / 2. / mass);
            var gradient = this.grad(position);
            momentum = VectorUtils.add_with_coeffs(momentum, gradient, 1., -epsilon);
            position = VectorUtils.add_with_coeffs(position, momentum, 1., epsilon / 2. / mass);
            momentum = VectorUtils.multiply(momentum, Math.sqrt(iteration_alpha));
            return [position, momentum];
        }

        // hamiltonian monte-carlo
        // step_size = epsilon in Neal's paper
        // n_steps = L in Neal's paper
        // tempering_alpha - alpha from Neal's paper

    }, {
        key: "generate_hmc",
        value: function generate_hmc() {
            var _this = this;

            var _ref = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {},
                _ref$suppress_reject = _ref.suppress_reject,
                suppress_reject = _ref$suppress_reject === undefined ? false : _ref$suppress_reject,
                _ref$step_size = _ref.step_size,
                step_size = _ref$step_size === undefined ? 0.0513 : _ref$step_size,
                _ref$n_steps = _ref.n_steps,
                n_steps = _ref$n_steps === undefined ? 60 : _ref$n_steps,
                _ref$tempering_alpha = _ref.tempering_alpha,
                tempering_alpha = _ref$tempering_alpha === undefined ? 1. : _ref$tempering_alpha;

            var n_steps_ = Math.max(1, this.random.random_normal(n_steps, Math.sqrt(n_steps)));

            var _positions = _slicedToArray(this.positions[this.positions.length - 1], 2),
                x_old = _positions[0],
                y_old = _positions[1];

            var position = [x_old, y_old];
            var start_position = position;
            var momentum = [this.random.random_normal(0, 1) * Math.sqrt(this.T), this.random.random_normal(0, 1) * Math.sqrt(this.T)];
            var energy_old = this.energy(position) + VectorUtils.norm_squared(momentum) / 2.;

            var trajectory = [[position, momentum]];

            for (var iteration = 0; iteration < n_steps_; iteration++) {
                // alpha in the first half of trajectory, 1 / alpha in second, 1 in the middle
                var iteration_alpha = Math.pow(tempering_alpha, -Math.sign(2 * iteration + 1 - n_steps_));
                // console.log('alpha', iteration, iteration_alpha);

                var _leapfrog_step = this.leapfrog_step(position, momentum, step_size, iteration_alpha);

                var _leapfrog_step2 = _slicedToArray(_leapfrog_step, 2);

                position = _leapfrog_step2[0];
                momentum = _leapfrog_step2[1];

                trajectory.push([position, momentum]);
            }

            var energy_new = this.energy(position) + VectorUtils.norm_squared(momentum) / 2.;

            var reject = this.random.random() > Math.exp((energy_old - energy_new) / this.T);
            if (suppress_reject) {
                reject = false;
            }
            var result = void 0;
            if (reject) {
                result = start_position;
            } else {
                result = position;
            }
            this.positions.push(result);
            var trajectory_3d = trajectory.map(function (x) {
                return _this.to_3d_point(x[0]);
            });

            // return final position, candidate, and reject solution
            return [this.to_3d_point(result), this.to_3d_point(position), reject, trajectory_3d];
        }
    }, {
        key: "get_3d_trajectory",
        value: function get_3d_trajectory() {
            var _this2 = this;

            return this.positions.map(function (x) {
                return _this2.to_3d_point(x);
            });
        }
    }]);

    return MCSampler;
}();

//# sourceMappingURL=mcmc-compiled.js.map
