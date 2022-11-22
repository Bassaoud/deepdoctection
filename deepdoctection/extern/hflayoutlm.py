# -*- coding: utf-8 -*-
# File: hflayoutlm.py

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
HF Layoutlm model for diverse downstream tasks.
"""
from abc import ABC
from copy import copy
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Union
from collections import defaultdict

from ..utils.detection_types import Requirement, JsonDict
from ..utils.file_utils import (
    get_pytorch_requirement,
    get_transformers_requirement,
    pytorch_available,
    transformers_available,
)
from ..utils.settings import (
    BioTag,
    ObjectTypes,
    TokenClasses,
    TypeOrStr,
    get_type,
    token_class_tag_to_token_class_with_tag,
    token_class_with_tag_to_token_class_and_tag,
)
from .base import LMSequenceClassifier, LMTokenClassifier, SequenceClassResult, TokenClassResult
from .pt.ptutils import set_torch_auto_device

if pytorch_available():
    import torch
    import torch.nn.functional as F
    from torch import Tensor  # pylint: disable=W0611

if transformers_available():
    from transformers import (
        LayoutLMForSequenceClassification,
        LayoutLMForTokenClassification,
        PretrainedConfig,
        LayoutLMv2Config,
        LayoutLMv2ForTokenClassification
    )


def predict_token_classes(
    uuids: List[List[str]],
    input_ids: "Tensor",
    attention_mask: "Tensor",
    token_type_ids: "Tensor",
    boxes: "Tensor",
    tokens: List[List[str]],
    model: "LayoutLMForTokenClassification",
    images: Optional[List["Tensor"]] = None,
) -> List[TokenClassResult]:
    """
    :param uuids: A list of uuids that correspond to a word that induces the resulting token
    :param input_ids: Token converted to ids to be taken from LayoutLMTokenizer
    :param attention_mask: The associated attention masks from padded sequences taken from LayoutLMTokenizer
    :param token_type_ids: Torch tensor of token type ids taken from LayoutLMTokenizer
    :param boxes: Torch tensor of bounding boxes of type 'xyxy'
    :param tokens: List of original tokens taken from LayoutLMTokenizer
    :param model: layoutlm model for token classification
    :param images: A list of torch image tensors or None
    :return: A list of TokenClassResults
    """
    if images is None:
        outputs = model(input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, token_type_ids=token_type_ids)
    else:
        outputs = model(
            input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, token_type_ids=token_type_ids, image=images
        )
    soft_max = F.softmax(outputs.logits, dim=2)
    score = torch.max(soft_max, dim=2)[0].tolist()
    token_class_predictions_ = outputs.logits.argmax(-1).tolist()
    input_ids_list_ = input_ids.tolist()

    all_results = defaultdict(list)
    for idx, uuid_list in enumerate(uuids):
        for pos, token in enumerate(uuid_list):
            all_results[token].append((input_ids_list_[idx][pos],
                                       token_class_predictions_[idx][pos],
                                       tokens[idx][pos],
                                       score[idx][pos]))
    all_token_classes = []
    for uuid, res in all_results.items():
        res.sort(key=lambda x: x[3], reverse=True)
        output = res[0]
        all_token_classes.append(TokenClassResult(uuid=uuid,
                                                  token_id=output[0],
                                                  class_id=output[1],
                                                  token=output[2],
                                                  score=output[3]))
    return all_token_classes


def predict_sequence_classes(
    input_ids: "Tensor",
    attention_mask: "Tensor",
    token_type_ids: "Tensor",
    boxes: "Tensor",
    model: "LayoutLMForSequenceClassification",
) -> SequenceClassResult:
    """
    :param input_ids: Token converted to ids to be taken from LayoutLMTokenizer
    :param attention_mask: The associated attention masks from padded sequences taken from LayoutLMTokenizer
    :param token_type_ids: Torch tensor of token type ids taken from LayoutLMTokenizer
    :param boxes: Torch tensor of bounding boxes of type 'xyxy'
    :param model: layoutlm model for token classification
    :return: SequenceClassResult
    """

    outputs = model(input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, token_type_ids=token_type_ids)
    score = torch.max(F.softmax(outputs.logits)).tolist()
    sequence_class_predictions = outputs.logits.argmax(-1).squeeze().tolist()

    return SequenceClassResult(class_id=sequence_class_predictions, score=float(score))  # type: ignore


class HFLayoutLmTokenClassifierBase(LMTokenClassifier, ABC):
    """
    A wrapper class for :class:`transformers.LayoutLMForTokenClassification` to use within a pipeline component.
    Check https://huggingface.co/docs/transformers/model_doc/layoutlm for documentation of the model itself.
    Note that this model is equipped with a head that is only useful when classifying tokens. For sequence
    classification and other things please use another model of the family.

    **Example**

        .. code-block:: python

            # setting up compulsory ocr service
            tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
            tess = TesseractOcrDetector(tesseract_config_path)
            ocr_service = TextExtractionService(tess)

            # hf tokenizer and token classifier
            tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
            layoutlm = HFLayoutLmTokenClassifier("path/to/config.json","path/to/model.bin",
                                                  categories= ['B-answer', 'B-header', 'B-question', 'E-answer',
                                                               'E-header', 'E-question', 'I-answer', 'I-header',
                                                               'I-question', 'O', 'S-answer', 'S-header',
                                                               'S-question'])

            # token classification service
            layoutlm_service = LMTokenClassifierService(tokenizer,layoutlm, image_to_layoutlm_features)

            pipe = DoctectionPipe(pipeline_component_list=[ocr_service,layoutlm_service])

            path = "path/to/some/form"
            df = pipe.analyze(path=path)

            for dp in df:
                ...
    """
    model: Union[LayoutLMForTokenClassification, LayoutLMv2ForTokenClassification]

    def __init__(
        self,
        path_config_json: str,
        path_weights: str,
        categories_semantics: Optional[Sequence[TypeOrStr]] = None,
        categories_bio: Optional[Sequence[TypeOrStr]] = None,
        categories: Optional[Mapping[str, TypeOrStr]] = None,
        device: Optional[Literal["cpu", "cuda"]] = None,
    ):
        """
        :param categories_semantics: A dict with key (indices) and values (category names) for NER semantics, i.e. the
                                     entities self. To be consistent with detectors use only values >0. Conversion will
                                     be done internally.
        :param categories_bio: A dict with key (indices) and values (category names) for NER tags (i.e. BIO). To be
                               consistent with detectors use only values>0. Conversion will be done internally.
        :param categories: If you have a pre-trained model you can pass a complete dict of NER categories
        """

        self.name = "_".join(Path(path_weights).parts[-3:])
        if categories is None:
            if categories_semantics is None:
                raise ValueError("If categories is None then categories_semantics cannot be None")
            if categories_bio is None:
                raise ValueError("If categories is None then categories_bio cannot be None")

        self.path_config = path_config_json
        self.path_weights = path_weights
        self.categories_semantics = (
            [get_type(cat_sem) for cat_sem in categories_semantics] if categories_semantics is not None else []
        )
        self.categories_bio = [get_type(cat_bio) for cat_bio in categories_bio] if categories_bio is not None else []
        if categories:
            self.categories = copy(categories)  # type: ignore
        else:
            self.categories = self._categories_orig_to_categories(
                self.categories_semantics, self.categories_bio  # type: ignore
            )
        if device is not None:
            self.device = device
        else:
            self.device = set_torch_auto_device()
        self.model.to(self.device)

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_pytorch_requirement(), get_transformers_requirement()]

    @staticmethod
    def _categories_orig_to_categories(
        categories_semantics: List[TokenClasses], categories_bio: List[BioTag]
    ) -> Dict[str, ObjectTypes]:
        categories_list = sorted(
            {
                token_class_tag_to_token_class_with_tag(token, tag)
                for token in categories_semantics
                for tag in categories_bio
            }
        )
        return {str(k): v for k, v in enumerate(categories_list, 1)}

    def _map_category_names(self, token_results: List[TokenClassResult]) -> List[TokenClassResult]:
        for result in token_results:
            result.class_name = self.categories[str(result.class_id + 1)]
            output = token_class_with_tag_to_token_class_and_tag(result.class_name)
            if output is not None:
                token_class, tag = output
                result.semantic_name = token_class
                result.bio_tag = tag
            result.class_id += 1
        return token_results

    def _validate_encodings(self, **encodings: Union[List[List[str]], "torch.Tensor"]):
        image_ids = encodings.get("image_ids")
        ann_ids = encodings.get("ann_ids")
        input_ids = encodings.get("input_ids")
        attention_mask = encodings.get("attention_mask")
        token_type_ids = encodings.get("token_type_ids")
        boxes = encodings.get("bbox")
        tokens = encodings.get("tokens")

        assert isinstance(ann_ids, list), type(ann_ids)
        if len(set(image_ids)) > 1:
            raise ValueError("HFLayoutLmTokenClassifier accepts for inference only one image.")
        assert isinstance(input_ids, torch.Tensor), type(input_ids)
        assert isinstance(attention_mask, torch.Tensor), type(attention_mask)
        assert isinstance(token_type_ids, torch.Tensor), type(token_type_ids)
        assert isinstance(boxes, torch.Tensor), type(boxes)
        assert isinstance(tokens, list), type(tokens)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        boxes = boxes.to(self.device)
        return image_ids, ann_ids, input_ids, attention_mask, token_type_ids,  boxes, tokens

    def clone(self) -> "HFLayoutLmTokenClassifierBase":
        return self.__class__(
            self.path_config,
            self.path_weights,
            self.categories_semantics,
            self.categories_bio,
            self.categories,
            self.device,
        )


class HFLayoutLmTokenClassifier(HFLayoutLmTokenClassifierBase):
    def __init__(
        self,
        path_config_json: str,
        path_weights: str,
        categories_semantics: Optional[Sequence[TypeOrStr]] = None,
        categories_bio: Optional[Sequence[TypeOrStr]] = None,
        categories: Optional[Mapping[str, TypeOrStr]] = None,
        device: Optional[Literal["cpu", "cuda"]] = None,
    ):
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config_json)
        self.model = LayoutLMForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )
        super().__init__(path_config_json, path_weights, categories_semantics, categories_bio, categories, device)

    def predict(self, **encodings: Union[List[List[str]], "torch.Tensor"]) -> List[TokenClassResult]:
        """
        Launch inference on LayoutLm for token classification. Pass the following arguments

        :param input_ids: Token converted to ids to be taken from LayoutLMTokenizer
        :param attention_mask: The associated attention masks from padded sequences taken from LayoutLMTokenizer
        :param token_type_ids: Torch tensor of token type ids taken from LayoutLMTokenizer
        :param boxes: Torch tensor of bounding boxes of type 'xyxy'
        :param tokens: List of original tokens taken from LayoutLMTokenizer

        :return: A list of TokenClassResults
        """

        image_ids, ann_ids, \
            input_ids, attention_mask, token_type_ids, boxes, tokens = self._validate_encodings(**encodings)

        results = predict_token_classes(
            ann_ids, input_ids, attention_mask, token_type_ids, boxes, tokens, self.model, None
        )

        return self._map_category_names(results)


class HFLayoutLmv2TokenClassifier(HFLayoutLmTokenClassifierBase):
    def __init__(
            self,
            path_config_json: str,
            path_weights: str,
            categories_semantics: Optional[Sequence[TypeOrStr]] = None,
            categories_bio: Optional[Sequence[TypeOrStr]] = None,
            categories: Optional[Mapping[str, TypeOrStr]] = None,
            device: Optional[Literal["cpu", "cuda"]] = None,
    ):
        config = LayoutLMv2Config.from_pretrained(pretrained_model_name_or_path=path_config_json)
        self.model = LayoutLMv2ForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )
        super().__init__(path_config_json, path_weights, categories_semantics, categories_bio, categories, device)

    def predict(self, **encodings: Union[List[List[str]], "torch.Tensor"]) -> List[TokenClassResult]:
        """
        Launch inference on LayoutLm for token classification. Pass the following arguments

        :param input_ids: Token converted to ids to be taken from LayoutLMTokenizer
        :param attention_mask: The associated attention masks from padded sequences taken from LayoutLMTokenizer
        :param token_type_ids: Torch tensor of token type ids taken from LayoutLMTokenizer
        :param boxes: Torch tensor of bounding boxes of type 'xyxy'
        :param tokens: List of original tokens taken from LayoutLMTokenizer

        :return: A list of TokenClassResults
        """

        image_ids, ann_ids, \
            input_ids, attention_mask, token_type_ids, boxes, tokens = self._validate_encodings(**encodings)

        images = encodings.get("images")
        images = images.to(self.device)
        results = predict_token_classes(
            ann_ids, input_ids, attention_mask, token_type_ids, boxes, tokens, self.model, images
        )

        return self._map_category_names(results)

    @staticmethod
    def default_arguments_for_input_mapping() -> JsonDict:
        """
        Add some default arguments that might be necessary when preparing a sample. Overwrite this method
        for some custom setting.
        """
        return {"image_width": 224,
                "image_height": 224}


class HFLayoutLmSequenceClassifier(LMSequenceClassifier):
    """
    A wrapper class for :class:`transformers.LayoutLMForSequenceClassification` to use within a pipeline component.
    Check https://huggingface.co/docs/transformers/model_doc/layoutlm for documentation of the model itself.
    Note that this model is equipped with a head that is only useful for classifying the input sequence. For token
    classification and other things please use another model of the family.

    **Example**

        .. code-block:: python

            # setting up compulsory ocr service
            tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
            tess = TesseractOcrDetector(tesseract_config_path)
            ocr_service = TextExtractionService(tess)

            # hf tokenizer and token classifier
            tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
            layoutlm = HFLayoutLmSequenceClassifier("path/to/config.json","path/to/model.bin",
                                                  categories=["handwritten", "presentation", "resume"])

            # token classification service
            layoutlm_service = LMSequenceClassifierService(tokenizer,layoutlm, image_to_layoutlm_features)

            pipe = DoctectionPipe(pipeline_component_list=[ocr_service,layoutlm_service])

            path = "path/to/some/form"
            df = pipe.analyze(path=path)

            for dp in df:
                ...
    """

    def __init__(
        self,
        path_config_json: str,
        path_weights: str,
        categories: Mapping[str, TypeOrStr],
        device: Optional[Literal["cpu", "cuda"]] = None,
    ):
        self.name = "_".join(Path(path_weights).parts[-3:])
        self.path_config = path_config_json
        self.path_weights = path_weights
        self.categories = copy(categories)  # type: ignore
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config_json)
        self.model = LayoutLMForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )
        if device is not None:
            self.device = device
        else:
            self.device = set_torch_auto_device()
        self.model.to(self.device)

    def predict(self, **encodings: Union[List[List[str]], "torch.Tensor"]) -> SequenceClassResult:

        input_ids = encodings.get("input_ids")
        attention_mask = encodings.get("attention_mask")
        token_type_ids = encodings.get("token_type_ids")
        boxes = encodings.get("bbox")

        assert isinstance(input_ids, torch.Tensor), type(input_ids)
        assert isinstance(attention_mask, torch.Tensor), type(attention_mask)
        assert isinstance(token_type_ids, torch.Tensor), type(token_type_ids)
        assert isinstance(boxes, torch.Tensor), type(boxes)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        boxes = boxes.to(self.device)

        result = predict_sequence_classes(
            input_ids,
            attention_mask,
            token_type_ids,
            boxes,
            self.model,
        )

        result.class_id += 1
        result.class_name = self.categories[str(result.class_id)]
        return result

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_pytorch_requirement(), get_transformers_requirement()]

    def clone(self) -> "HFLayoutLmSequenceClassifier":
        return self.__class__(self.path_config, self.path_weights, self.categories, self.device)
