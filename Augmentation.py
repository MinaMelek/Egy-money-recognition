# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 01:51:35 2017

@author: MinaMelek
"""

import Augmentor
direct='D:\\Education\\4th_communication\\GP\\JAVA\\Work\\D\\6'
n0=515
p = Augmentor.Pipeline(direct)
p.rotate(probability=1,max_left_rotation=20,max_right_rotation=20)
p.random_distortion(probability=0.5,grid_width=10, grid_height=10, magnitude=10)
p.sample(5*n0)
p = Augmentor.Pipeline(direct)
p.skew(probability=1,magnitude=0.7)
p.random_distortion(probability=0.5,grid_width=10, grid_height=10, magnitude=10)
p.sample(5*n0)
p = Augmentor.Pipeline(direct)
p.shear(probability=1,max_shear_left=10,max_shear_right=10)
p.random_distortion(probability=0.5,grid_width=10, grid_height=10, magnitude=10)
p.sample(3*n0)
p = Augmentor.Pipeline(direct)
p.rotate180(probability=0.5)
p.random_distortion(probability=1,grid_width=10, grid_height=10, magnitude=10)
p.sample(3*n0)
p = Augmentor.Pipeline(direct)
p.rotate(probability=0.6,max_left_rotation=20,max_right_rotation=20)
p.skew(probability=0.5,magnitude=0.7)
p.shear(probability=0.5,max_shear_left=10,max_shear_right=10)
p.random_distortion(probability=0.7,grid_width=10, grid_height=10, magnitude=10)
p.sample(3*n0)
#p.random_distortion(probability=1,grid_width=10, grid_height=10, magnitude=10)
#p.shear(probability=1,max_shear_left=20,max_shear_right=20)
#p.flip_top_bottom(probability=0.5)
#p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)