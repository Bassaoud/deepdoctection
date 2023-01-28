# -*- coding: utf-8 -*-
# File: layout.py

# Copyright 2021 Dr. Janis Meyer. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Module for layout pipeline component
"""
from typing import Optional

import numpy as np

from ..datapoint.image import Image
from ..extern.base import ObjectDetector, PdfMiner
from ..utils.detection_types import JsonDict
from ..utils.transform import PadTransform
from .base import PredictorPipelineComponent
from .registry import pipeline_component_registry


@pipeline_component_registry.register("ImageLayoutService")
class ImageLayoutService(PredictorPipelineComponent):
    """
    Pipeline component for determining the layout. Which layout blocks are determined depends on the Detector and thus
    usually on the data set on which the Detector was pre-trained. If the Detector has been trained on Publaynet, these
    are layouts such as text, title, table, list and figure. If the Detector has been trained on DocBank, these are
    rather Abstract, Author, Caption, Equation, Figure, Footer, List, Paragraph, Reference, Section, Table, Title.

    The component is usually at the beginning of the pipeline. Cropping of the layout blocks can be selected to simplify
    further processing.

    **Example**

            d_items = TPFrcnnDetector(item_config_path, item_weights_path, {"1": "ROW", "2": "COLUMNS"})
            item_component = ImageLayoutService(d_items)
    """

    def __init__(
        self,
        layout_detector: ObjectDetector,
        to_image: bool = False,
        crop_image: bool = False,
        padder: Optional[PadTransform] = None
    ):
        """
        :param layout_detector: object detector
        :param to_image: Generate an image for each detected block, e.g. populate `ImageAnnotation.image`. Useful,
                         if you want to process only some blocks in a subsequent pipeline component.
        :param crop_image: Do not only populate `ImageAnnotation.image` but also crop the detected block according
                           to its bounding box and populate the resulting sub image to
                           `ImageAnnotation.image.image`.
        """
        self.to_image = to_image
        self.crop_image = crop_image
        self.padder = padder
        super().__init__(self._get_name(layout_detector.name), layout_detector)

    def serve(self, dp: Image) -> None:
        if dp.image is None:
            raise ValueError("image cannot be None")
        np_image = dp.image
        if self.padder:
            np_image = self.padder.apply_image(np_image)
        detect_result_list = self.predictor.predict(np_image)  # type: ignore
        if self.padder and  detect_result_list:
            boxes = np.array([detect_result.box for detect_result in detect_result_list])
            boxes_orig = self.padder.inverse_apply_coords(boxes)
            for idx, detect_result in enumerate(detect_result_list):
                detect_result.box = boxes_orig[idx,:].tolist()

        for detect_result in detect_result_list:
            self.dp_manager.set_image_annotation(detect_result, to_image=self.to_image, crop_image=self.crop_image)

    def get_meta_annotation(self) -> JsonDict:
        assert isinstance(self.predictor, (ObjectDetector, PdfMiner))
        return dict(
            [
                ("image_annotations", self.predictor.possible_categories()),
                ("sub_categories", {}),
                ("relationships", {}),
                ("summaries", []),
            ]
        )

    @staticmethod
    def _get_name(predictor_name: str) -> str:
        return f"image_{predictor_name}"

    def clone(self) -> "PredictorPipelineComponent":
        predictor = self.predictor.clone()
        padder_clone = None
        if self.padder:
            padder_clone = self.padder.clone()
        if not isinstance(predictor, ObjectDetector):
            raise ValueError(f"predictor must be of type ObjectDetector, but is of type {type(predictor)}")
        return self.__class__(predictor, self.to_image, self.crop_image, padder_clone)
