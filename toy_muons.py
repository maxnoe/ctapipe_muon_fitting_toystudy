#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from ctapipe.containers import ArrayEventContainer, DL1CameraContainer
from ctapipe.instrument import SubarrayDescription
import astropy.units as u
from ctapipe.image import ImageModifier

from ctapipe.image import ImageProcessor
from ctapipe.image.muon import MuonProcessor
from ctapipe.image.muon.intensity_fitter import image_prediction_no_units
from tqdm.auto import tqdm

from ctapipe.core import Tool, traits, Field, Container
from ctapipe.visualization import CameraDisplay
from ctapipe.io import HDF5TableWriter


class TrueParametersContainer(Container):
    default_prefix = "true"
    radius = Field(np.nan * u.deg)
    width = Field(np.nan * u.deg)
    fov_lon = Field(np.nan * u.deg)
    fov_lat =Field(np.nan * u.deg)
    impact_parameter = Field(np.nan * u.m)
    phi = Field(np.nan * u.m)
    optical_efficiency = Field(np.nan)


class ToyMuonFits(Tool):
    
    subarray_path = traits.Path(help="Path to read subarray from").tag(config=True)
    n_events = traits.Integer(10000).tag(config=True)
    output_path = traits.Path(help="Output path").tag(config=True)
    tel_id = traits.Integer(1).tag(config=True)
    noise = traits.Float(3).tag(config=True)
    seed = traits.Integer(0).tag(config=True)

    aliases = {
        ("s", "subarray"): "ToyMuonFits.subarray_path",
        ("o", "output"): "ToyMuonFits.output_path",
    }

    def setup(self):
        self.subarray = SubarrayDescription.read(self.subarray_path)
        self.telescope = self.subarray.tel[self.tel_id]
        self.optics = self.telescope.optics
        self.rng = np.random.default_rng(0)

        self.add_noise = ImageModifier(
            self.subarray,
            noise_level_bright_pixels=self.noise,
            noise_level_dim_pixels=self.noise,
        )
        self.image_processor = ImageProcessor(self.subarray)
        self.muon_processor = MuonProcessor(
            self.subarray,
            pedestal=np.sqrt(self.noise),
        )
        self.camera_geometry = self.muon_processor.geometries[self.tel_id]
        self.count = 0
        self.writer = self.enter_context(HDF5TableWriter(self.output_path, add_prefix=True))

    def start(self):
        for _ in tqdm(range(self.n_events)):
            event, true_parameters = self.generate_event()

            self.image_processor(event)
            self.muon_processor(event)

            muon = event.muon.tel[self.tel_id]
            self.writer.write(
                f"muon/tel_{self.tel_id:03d}",
                (true_parameters, muon.ring, muon.efficiency, muon.parameters),
            )


    def generate_event(self):
        image, parameters = self.generate_muon_image()

        image = self.add_noise(self.tel_id, image, rng=self.rng)

        event = ArrayEventContainer(count=self.count)
        event.dl1.tel[self.tel_id] = DL1CameraContainer(
            image=image,
        )

        self.count += 1
        return event, parameters


    def image_prediction(
        self,
        center_x,
        center_y,
        phi,
        impact_parameter,
        radius,
        ring_width,
    ):
        mirror_area = self.optics.mirror_area
        mirror_radius = np.sqrt(mirror_area / np.pi)
        hole_radius = self.muon_processor.intensity_fitter.hole_radius_m.tel[self.tel_id]

        return image_prediction_no_units(
            mirror_radius_m=mirror_radius.to_value(u.m),
            hole_radius_m=hole_radius,
            impact_parameter_m=impact_parameter.to_value(u.m),
            phi_rad=phi.to_value(u.rad),
            center_x_rad=center_x.to_value(u.rad),
            center_y_rad=center_y.to_value(u.rad),
            radius_rad=radius.to_value(u.rad),
            ring_width_rad=ring_width.to_value(u.rad),
            pixel_x_rad=self.camera_geometry.pix_x.to_value(u.rad),
            pixel_y_rad=self.camera_geometry.pix_y.to_value(u.rad),
            pixel_diameter_rad=self.camera_geometry.pixel_width.to_value(u.rad)[0],
        )


    def generate_muon_image(self):
        rng = self.rng
        mirror_radius = np.sqrt(self.optics.mirror_area / np.pi)
        max_angle = 0.5 * self.muon_processor.fov_radius[self.tel_id]
        offset = max_angle * np.sqrt(rng.uniform(0, 1))
        phi_cog = rng.uniform(0, 2 * np.pi)
        fov_lon = np.cos(phi_cog) * offset
        fov_lat = np.sin(phi_cog) * offset
        
        true_parameters = TrueParametersContainer(
            radius = rng.uniform(0.6, 1.2) * u.deg,
            width = rng.uniform(0.02, 0.1) * u.deg,
            fov_lon = fov_lon,
            fov_lat = fov_lat,
            impact_parameter = rng.uniform(0, 0.5 * mirror_radius.to_value(u.m)) * u.m,
            phi = rng.uniform(0, 360) * u.deg,
            optical_efficiency = rng.uniform(0.1, 0.3),
        )

        image = true_parameters.optical_efficiency * self.image_prediction(
            center_x=true_parameters.fov_lon,
            center_y=true_parameters.fov_lat,
            phi=true_parameters.phi,
            impact_parameter=true_parameters.impact_parameter,
            radius=true_parameters.radius,
            ring_width=true_parameters.width,
        )
        return image, true_parameters


if __name__ == "__main__":
    tool = ToyMuonFits().run()
