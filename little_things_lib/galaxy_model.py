import numpy as np

from little_things_lib.helpers import calc_physical_distance_per_pixel

class GalaxyModel:
    def __init__(
            self,
            distance_to_galaxy,
            deg_per_pixel,
            image_xdim,
            image_ydim,
            galaxy_name=None,
            vlos_2d_data=None,
            output_dir='output',
    ):
        self.galaxy_name = galaxy_name
        self.output_dir = output_dir
        self.vlos_2d_data = vlos_2d_data

        # TODO: write func to automatically read deg/pix and dim from fits header and remove these args
        self.deg_per_pixel = deg_per_pixel
        self.kpc_per_pixel = calc_physical_distance_per_pixel(distance_to_galaxy, self.deg_per_pixel)
        self.image_xdim, self.image_ydim = image_xdim, image_ydim

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
            v_systemic
        the following inputs should be 1 dim lists or arrays of same length
            radii [pixels]
            inclination [deg]
            position_angle [deg]
            x_pix_center [pixel]
            y_pix_center [pixel]

        '''
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
                if (x < self.image_xdim-1  and y < self.image_ydim-1
                and x>0 and y>0):
                    arr_x, arr_y = int(np.round(x, 0)), int(np.round(y, 0))
                    v_field[arr_y][arr_x] = v_los
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
        x_kpc = r * (np.cos(pos_ang) * np.cos(theta) - np.sin(pos_ang) * np.sin(theta) * np.sin(inc))
        y_kpc = r * (np.sin(pos_ang) * np.cos(theta) + np.cos(pos_ang) * np.sin(theta) * np.sin(inc))
        x_pix = x_kpc / self.kpc_per_pixel
        y_pix = y_kpc / self.kpc_per_pixel
        return (x_pix, y_pix)