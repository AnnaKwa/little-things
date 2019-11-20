import numpy as np
from scipy.interpolate import interp1d
from .helpers import calc_physical_distance_per_pixel, extrapolate_v_outside_last_radius

SEC_PER_GYR = 3.15576e+16

class Galaxy:
    def __init__(
            self,
            distance_to_galaxy,
            deg_per_pixel,
            image_xdim,
            image_ydim,
            galaxy_name=None,
            vlos_2d_data=None,
            output_dir='output',
            luminosity=None,
            HI_mass=None,
            age=10. # Gyr
    ):
        self.galaxy_name = galaxy_name
        self.output_dir = output_dir
        self.vlos_2d_data = vlos_2d_data
        self.luminosity = luminosity
        self.HI_mass = HI_mass

        self.rate_constant = 1.5278827817856099e-26 * age * SEC_PER_GYR

        # TODO: write func to automatically read deg/pix and dim from fits header and remove these args
        self.deg_per_pixel = deg_per_pixel
        self.kpc_per_pixel = calc_physical_distance_per_pixel(distance_to_galaxy, self.deg_per_pixel)
        self.image_xdim, self.image_ydim = image_xdim, image_ydim


    def set_prior_bounds(
            self,
            cross_section_bounds=(2.999, 3.001),
            rate_bounds=(2, 1e4),
            sigma0_bounds=(2, 1e3),
            ml_bounds=(0.1, 10),
            ml_median=0.5,
            rmax_prior=False,
            vmax_prior=False,
            log10_rmax_spread=0.11,
            log10_c200_spread=0.11,
            abs_err_vel_factor=0.05,
            tophat_width=3.
    ):
        if self.luminosity and self.HI_mass:
            abs_err_vel = abs_err_vel_factor * ((0.5 * self.luminosity + self.HI_mass) * 1e9 / 50) ** 0.25
        else:
            raise ValueError('Need to set luminosity and HI mass for galaxy.')
        regularization_params = (abs_err_vel, 0.0, vmax_prior, 1.414)  # what are these??

        rho0_bounds = (rate_bounds[0] / (self.rate_constant * sigma0_bounds[0] * cross_section_bounds[0]),
                       rate_bounds[1] / (self.rate_constant * sigma0_bounds[1] * cross_section_bounds[1]))
        bounds = {
            'rho0': tuple(np.log10(rho0_bounds)),
            'sigma0': tuple(np.log10(sigma0_bounds)),
            'cross_section': cross_section_bounds,
            'ml': ml_bounds
        }
        prior_params = {
            'rmax_prior': rmax_prior,
            'log10_rmax_spread': log10_rmax_spread,
            'log10_c200_spread': log10_c200_spread,
            'ml_median': ml_median,
            'tophat_width': tophat_width
        }

        self.regularization_params = regularization_params
        self.bounds = bounds
        self.prior_params = prior_params


    def interpolate_baryonic_rotation_curve(
            self,
            baryon_type, # stellar/star/stars or gas
            rotation_curve_radii,
            rotation_curve_velocities
    ):
        """

        :param gas_rotation_curve_radii: [kpc]
        :param gas_rotation_curve_velocities: [km/s]
        :return:
        """

        rotation_curve_radii, rotation_curve_velocities = np.array([
            (r, v) for r, v in zip(rotation_curve_radii, rotation_curve_velocities)
                if (np.min(self.radii)<= r <= np.max(self.radii))
        ]).T
        interp_rotation = interp1d(rotation_curve_radii, rotation_curve_velocities)
        interp_radii = [
            r for r in self.radii
            if (np.min(rotation_curve_radii) <= r <= np.max(rotation_curve_radii))
        ]
        v_interp = list(interp_rotation(interp_radii))
        extrap_inside_radii = [r for r in self.radii if r < np.min(rotation_curve_radii)]
        extrap_outside_radii = [r for r in self.radii if r > np.max(rotation_curve_radii)]
        v_extrap_inside = [np.min(rotation_curve_velocities) for r in extrap_inside_radii]
        r_last, v_last = rotation_curve_radii[-1], rotation_curve_velocities[-1]
        v_extrap_outside = [extrapolate_v_outside_last_radius(r, r_last, v_last) for r in extrap_outside_radii]
        if baryon_type == 'gas':
            self.v_gas = np.array(v_extrap_inside + v_interp + v_extrap_outside)
        else:
            self.v_stellar = np.array(v_extrap_inside + v_interp + v_extrap_outside)

    def set_tilted_ring_parameters(
            self,
            v_systemic,
            radii,
            inclination,
            position_angle,
            x_pix_center,
            y_pix_center
    ):
        '''
        scalar (single value) inputs:
            v_systemic [km/s]
        the following inputs should be 1 dim lists or arrays of same length
            radii [kpc]
            inclination [deg]
            position_angle [deg]
            x_pix_center [pixel]
            y_pix_center [pixel]

        '''
        self.radii = radii
        self.v_systemic = v_systemic
        self.ring_parameters = {
            radius: {
                'inc': inc * (np.pi/360),
                'pos_ang': pos_ang * (np.pi/360),
                'x_center': x,
                'y_center': y
            }
            for radius, inc, pos_ang, x, y
                in zip(radii, inclination, position_angle, x_pix_center, y_pix_center)
        }

    def create_2d_velocity_field(
            self,
            radii,
            v_rot
    ):
        '''
        uses tilted ring model parameters to calculate velocity field
        using eqn 1-3 of 1709.02049 and v_rot from mass model

        it is easier to loop through polar coordinates and then map the v_los to the
        nearest x,y point

        returns 2d velocity field array
        '''
        v_field = np.zeros(shape=(self.image_ydim, self.image_xdim))
        for r, v in zip(radii, v_rot):
            for theta in np.linspace(0, 2.*np.pi, 1000):
                x, y, v_los = self._calc_v_los_at_r_theta(v, r, theta)
                if (self.image_xdim - 1 > x > 0 and y < self.image_ydim-1 and y>0):
                    arr_x, arr_y = int(np.round(x, 0)), int(np.round(y, 0))
                    try:
                        v_field[arr_y][arr_x] = v_los
                    except:
                        print (arr_x, arr_y, v_los)
        return v_field

    def _calc_v_los_at_r_theta(
            self,
            v_rot,
            r,
            theta
    ):
        inc = self.ring_parameters[r]['inc']
        x0 = self.ring_parameters[r]['x_center']
        y0 = self.ring_parameters[r]['y_center']
        x_from_galaxy_center, y_from_galaxy_center = self._convert_galaxy_to_observer_coords(r, theta)
        v_los = v_rot * np.cos(theta) * np.sin(inc) - self.v_systemic
        x = x0 + x_from_galaxy_center
        y = y0 + y_from_galaxy_center
        return (x, y, v_los)


    def _convert_galaxy_to_observer_coords(
            self,
            r,
            theta
    ):
        '''

        :param r: physical distance from center [kpc]
        :param theta: azimuthal measured CCW from major axis in plane of disk
        :return: x, y coords in observer frame after applying inc and position angle adjustment
        '''
        inc = self.ring_parameters[r]['inc']
        pos_ang = self.ring_parameters[r]['pos_ang']
        x_kpc = r * (np.sin(2*pos_ang) * np.sin(theta) - np.cos(2*pos_ang) * np.cos(theta) * np.cos(inc))
        y_kpc = r * (np.cos(2*pos_ang) * np.sin(theta) + np.sin(2*pos_ang) * np.cos(theta) * np.cos(inc))
        x_pix = x_kpc / self.kpc_per_pixel
        y_pix = y_kpc / self.kpc_per_pixel
        return (x_pix, y_pix)