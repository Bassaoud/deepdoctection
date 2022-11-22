# -*- coding: utf-8 -*-
# File: tokenclass.py

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
Module for token classification pipeline
"""
from copy import copy
import inspect
from typing import Any, Callable, List, Optional, Literal

from ..datapoint.image import Image
from ..extern.base import LMSequenceClassifier, LMTokenClassifier
from ..mapper.laylmstruct import LayoutLMFeatures
from ..utils.detection_types import JsonDict
from ..utils.settings import BioTag, LayoutType, PageType, TokenClasses, WordType
from .base import LanguageModelPipelineComponent
from .registry import pipeline_component_registry


@pipeline_component_registry.register("LMTokenClassifierService")
class LMTokenClassifierService(LanguageModelPipelineComponent):
    """
    Pipeline component for token classification

    **Example**

        .. code-block:: python

            # setting up compulsory ocr service
            tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
            tess = TesseractOcrDetector(tesseract_config_path)
            ocr_service = TextExtractionService(tess)

            # hf tokenizer and token classifier
            tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
            layoutlm = HFLayoutLmTokenClassifier(categories= ['B-answer', 'B-header', 'B-question', 'E-answer',
                                                               'E-header', 'E-question', 'I-answer', 'I-header',
                                                               'I-question', 'O', 'S-answer', 'S-header', 'S-question'])

            # token classification service
            layoutlm_service = LMTokenClassifierService(tokenizer,layoutlm,image_to_layoutlm)

            pipe = DoctectionPipe(pipeline_component_list=[ocr_service,layoutlm_service])

            path = "path/to/some/form"
            df = pipe.analyze(path=path)

            for dp in df:
                ...
    """

    def __init__(
        self,
        tokenizer: Any,
        language_model: LMTokenClassifier,
        mapping_to_lm_input_func: Callable[..., Callable[[Image], Optional[LayoutLMFeatures]]],
        padding: Literal["max_length", "do_not_pad", "longest"] = "max_length",
        truncation: bool = True,
        return_overflowing_tokens: bool = False,
        use_other_as_default_category: bool = False,
        sliding_window_stride: int = 0
    ) -> None:
        """
        :param tokenizer: Token classifier, typing allows currently anything. This will be changed in the future
        :param language_model: language model token classifier
        :param mapping_to_lm_input_func: Function mapping image to layout language model features
        :param padding: A padding strategy to be passed to the tokenizer. Must bei either `max_length, longest` or
                        `do_not_pad`.
        :param truncation: If "True" will truncate to a maximum length specified with the argument max_length or to the
                           maximum acceptable input length for the model if that argument is not provided. This will
                           truncate token by token, removing a token from the longest sequence in the pair if a pair of
                           sequences (or a batch of pairs) is provided.
                           If `False` then no truncation (i.e., can output batch with sequence lengths greater than the
                           model maximum admissible input size).
        :param return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows the overflowing tokens
                           can be returned as an additional batch element. Not that in this case, the number of input
                           batch samples will be smaller than the output batch samples.
        :param use_other_as_default_category: When predicting token classes, it might be possible that some words might
                                              not get sent to the model because they are categorized as not eligible
                                              token (e.g. empty string). If set to `True` it will assign all words
                                              without token the `BioTag.outside` token.
        """
        self.language_model = language_model
        self.padding = padding
        self.truncation = truncation
        self.return_overflowing_tokens = return_overflowing_tokens
        self.use_other_as_default_category = use_other_as_default_category
        self.sliding_window_stride = sliding_window_stride
        if self.use_other_as_default_category:
            categories_name_as_key = {val: key for key, val in self.language_model.categories.items()}
            self.other_name_as_key = {BioTag.outside: categories_name_as_key[BioTag.outside]}
        super().__init__(self._get_name(), tokenizer, mapping_to_lm_input_func)

        parameters = inspect.signature(mapping_to_lm_input_func).parameters
        self.required_kwargs = {"tokenizer": self.tokenizer,
                                "padding": self.padding,
                                "truncation": self.truncation,
                                "return_overflowing_tokens": self.return_overflowing_tokens,
                                "return_tensors": "pt",
                                "sliding_window_stride": self.sliding_window_stride}
        self.required_kwargs.update(self.language_model.default_arguments_for_input_mapping())
        for kwarg in self.required_kwargs:
            if kwarg not in parameters:
                raise TypeError(f"{mapping_to_lm_input_func} requires argument {kwarg}")

    def serve(self, dp: Image) -> None:
        lm_input = self.mapping_to_lm_input_func(**self.required_kwargs)(dp)
        if lm_input is None:
            return
        lm_output = self.language_model.predict(**lm_input)

        # turn to word level predictions and remove all special tokens
        lm_output = [
            token
            for token in lm_output
            if token.token_id
            not in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]
            and not token.token.startswith("##")
        ]

        words_populated: List[str] = []
        for token in lm_output:
            if token.uuid not in words_populated:
                self.dp_manager.set_category_annotation(token.semantic_name, None, WordType.token_class, token.uuid)
                self.dp_manager.set_category_annotation(token.bio_tag, None, WordType.tag, token.uuid)
                self.dp_manager.set_category_annotation(
                    token.class_name, token.class_id, WordType.token_tag, token.uuid
                )
                words_populated.append(token.uuid)

        if self.use_other_as_default_category:
            word_anns = dp.get_annotation(LayoutType.word)
            for word in word_anns:
                if WordType.token_class not in word.sub_categories:
                    self.dp_manager.set_category_annotation(
                        TokenClasses.other, None, WordType.token_class, word.annotation_id
                    )
                if WordType.tag not in word.sub_categories:
                    self.dp_manager.set_category_annotation(BioTag.outside, None, WordType.tag, word.annotation_id)
                if WordType.token_tag not in word.sub_categories:
                    self.dp_manager.set_category_annotation(
                        BioTag.outside,
                        self.other_name_as_key[BioTag.outside],
                        WordType.token_tag,
                        word.annotation_id,
                    )

    def clone(self) -> "LMTokenClassifierService":
        return self.__class__(
            copy(self.tokenizer),
            self.language_model.clone(),
            copy(self.mapping_to_lm_input_func),
            self.padding,
            self.truncation,
            self.return_overflowing_tokens,
            self.use_other_as_default_category,
        )

    def get_meta_annotation(self) -> JsonDict:
        return dict(
            [
                ("image_annotations", []),
                ("sub_categories", {LayoutType.word: {WordType.token_class, WordType.tag, WordType.token_tag}}),
                ("relationships", {}),
                ("summaries", []),
            ]
        )

    def _get_name(self) -> str:
        return f"lm_token_class_{self.language_model.name}"


@pipeline_component_registry.register("LMSequenceClassifierService")
class LMSequenceClassifierService(LanguageModelPipelineComponent):
    """
    Pipeline component for sequence classification

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
        tokenizer: Any,
        language_model: LMSequenceClassifier,
        mapping_to_lm_input_func: Callable[..., Callable[[Image], Optional[LayoutLMFeatures]]],
        padding: Literal["max_length", "do_not_pad", "longest"] = "max_length",
        truncation: bool = True,
        return_overflowing_tokens: bool = False,
    ) -> None:
        """
        :param tokenizer: Tokenizer, typing allows currently anything. This will be changed in the future
        :param language_model: language model sequence classifier
        :param mapping_to_lm_input_func: Function mapping image to layout language model features
        :param padding: A padding strategy to be passed to the tokenizer. Must bei either `max_length, longest` or
                        `do_not_pad`.
        :param truncation: If "True" will truncate to a maximum length specified with the argument max_length or to the
                           maximum acceptable input length for the model if that argument is not provided. This will
                           truncate token by token, removing a token from the longest sequence in the pair if a pair of
                           sequences (or a batch of pairs) is provided.
                           If `False` then no truncation (i.e., can output batch with sequence lengths greater than the
                           model maximum admissible input size).
        :param return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows the overflowing tokens
                           can be returned as an additional batch element. Not that in this case, the number of input
                           batch samples will be smaller than the output batch samples.
        """
        self.language_model = language_model
        parameters = inspect.signature(mapping_to_lm_input_func).parameters
        required_kwargs = {"tokenizer", "padding", "truncation", "return_overflowing_tokens",
                           "return_tensors"}
        for kwarg in required_kwargs:
            if kwarg not in parameters:
                raise TypeError(f"{mapping_to_lm_input_func} requires argument {kwarg}")
        self.padding = padding
        self.truncation = truncation
        self.return_overflowing_tokens = return_overflowing_tokens
        super().__init__(self._get_name(), tokenizer, mapping_to_lm_input_func)

    def serve(self, dp: Image) -> None:
        lm_input = self.mapping_to_lm_input_func(tokenizer=self.tokenizer,
                                                 padding=self.padding,
                                                 truncation=self.truncation,
                                                 return_overflowing_tokens=self.return_overflowing_tokens,
                                                 return_tensors="pt")(dp)
        if lm_input is None:
            return
        lm_output = self.language_model.predict(**lm_input)
        self.dp_manager.set_summary_annotation(
            PageType.document_type, lm_output.class_name, lm_output.class_id, None, lm_output.score
        )

    def clone(self) -> "LMSequenceClassifierService":
        return self.__class__(copy(self.tokenizer),
                              self.language_model.clone(),
                              copy(self.mapping_to_lm_input_func),
                              self.padding,
                              self.truncation,
                              self.return_overflowing_tokens)

    def get_meta_annotation(self) -> JsonDict:
        return dict(
            [
                ("image_annotations", []),
                ("sub_categories", {}),
                ("relationships", {}),
                ("summaries", [PageType.document_type]),
            ]
        )

    def _get_name(self) -> str:
        return f"lm_sequence_class_{self.language_model.name}"
