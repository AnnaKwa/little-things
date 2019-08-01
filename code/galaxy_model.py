import numpy as np

class GalaxyModel:
    def __init__(
            self,
            galaxy_name=None,
            fits_file=None,
            output_dir='output'):
        self.galaxy_name = galaxy_name
        self.output_dir = output_dir
        self.fits_file = fits_file

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
                'inc': inc,
                'pos_ang': pos_ang,
                'x_center': x,
                'y_center': y
            }
            for radius, inc, pos_ang, x, y
                in zip(radii, inclination, position_angle, x_pix_center, y_pix_center)
        }

    def create_2d_velocity_field(
            self,
            radii,
            v_rot):
        '''
        uses tilted ring model parameters to calculate velocity field
        using eqn 1-3 of 1709.02049 and v_rot from mass model

        returns 2d velocity field array
        '''
        pass
