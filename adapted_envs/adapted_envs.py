import contextlib
import os
import tempfile

import numpy as np
import xml.etree.ElementTree as ET

import roboschool
from roboschool.gym_mujoco_walkers import (
    RoboschoolForwardWalkerMujocoXML, RoboschoolHalfCheetah, RoboschoolHopper, RoboschoolAnt, RoboschoolWalker2d
)

from sunblaze_envs.base import EnvBinarySuccessMixin
from sunblaze_envs.mujoco import RoboschoolXMLModifierMixin, RoboschoolTrackDistSuccessMixin

ROBOSCHOOL_ASSETS = os.path.join(roboschool.__path__[0], 'mujoco_assets')

class ModifiableRoboschoolHalfCheetah(RoboschoolHalfCheetah, RoboschoolTrackDistSuccessMixin):

    def reset(self, new=True):
        return super(ModifiableRoboschoolHalfCheetah, self).reset()

    @property
    def parameters(self):
        return {'id': self.spec.id, }

class AdaptedHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def set_power(self, new_power):
        self.power = new_power

    def set_gravity(self, new_gravity):
        self.gravity = new_gravity
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('option'):
                elem.set('gravity', str(self.gravity))
        
    def set_friction(self, new_friction):
        self.friction = new_friction
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def set_env_params(self, new_param_dict):
        self.set_power(new_param_dict['power'])
        self.set_gravity(new_param_dict['gravity'])
        self.set_friction(new_param_dict['friction'])
                
    def reset(self, new=True, new_param_dict=None):
        if new_param_dict is not None:
            set_env_params(new_param_dict)
        return super(AdaptedHalfCheetah, self).reset(new)

    @property
    def parameters(self):
        parameters = super(AdaptedHalfCheetah, self).parameters
        parameters.update({'power': self.power, 'gravity' : self.gravity, 'friction': self.friction})
        return parameters

class ModifiableRoboschoolAnt(RoboschoolAnt, RoboschoolTrackDistSuccessMixin):

    def reset(self, new=True):
        return super(ModifiableRoboschoolAnt, self).reset()

    @property
    def parameters(self):
        return {'id': self.spec.id, }

class AdaptedAnt(RoboschoolXMLModifierMixin, ModifiableRoboschoolAnt):
    def set_power(self, new_power):
        self.power = new_power

    def set_gravity(self, new_gravity):
        # Change density rather than gravity
        self.gravity = new_gravity
        self.density = 5.0 * (self.gravity / -9.81)
        with self.modify_xml('ant.xml') as tree:
            for elem in tree.iterfind('option'):
                elem.set('density', str(self.density))
    '''    
    def set_friction(self, new_friction):
        self.friction = new_friction
        with self.modify_xml('ant.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')
    '''

    def set_env_params(self, new_param_dict):
        self.set_power(new_param_dict['power'])
        self.set_gravity(new_param_dict['gravity'])
        #self.set_friction(new_param_dict['friction'])
                
    def reset(self, new=True, new_param_dict=None):
        if new_param_dict is not None:
            set_env_params(new_param_dict)
        return super(AdaptedAnt, self).reset(new)

    @property
    def parameters(self):
        parameters = super(AdaptedAnt, self).parameters
        parameters.update({'power': self.power, 'gravity' : self.gravity}) #, 'friction': self.friction})
        return parameters
