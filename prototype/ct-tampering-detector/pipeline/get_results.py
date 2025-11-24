# # pipeline/get_results.py
# """
# Tamper Detection Pipeline for CT Scan Volumes.

# This module orchestrates the preprocessing, classification, and localization
# of tampering in CT scan slices. It supports detection of real vs. tampered
# volumes and, for tampered volumes, sub-classification into injected or removed
# tampering with localization of affected regions.
# """

# import logging
# from typing import List, Dict, Any, Tuple, Optional
# from dataclasses import dataclass

# from pipeline.classifiers import TamperClassifiers  # Fixed import name
# from pipeline.localize import Localize
# from pipeline.preProces.preProcess import preprocess  # Assuming this is the correct path
# from pipeline.types.types import Types

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# @dataclass
# class ClassificationResult:
#     """Result from tamper classification."""
#     status: int
#     classification: str
#     confidence: float
#     affected_filenames: List[str]
#     error: Optional[Exception] = None


# @dataclass
# class LocalizationResult:
#     """Result from tamper localization."""
#     filename: str
#     coords: List[Tuple[float, float]]  # List of (x, y) coordinates for bounding boxes
#     heatmap: Any  # Placeholder for heatmap data (e.g., numpy array or encoded string)


# class TamperPipeline:
#     """
#     Main pipeline for analyzing CT scan volumes for tampering.

#     Orchestrates preprocessing, binary classification (real vs. tampered),
#     sub-classification (injected vs. removed), and localization.
#     """

#     def __init__(self):
#         self.types = Types()
#         self.classifier: Optional[TamperClassifiers] = None
#         self.localizer: Optional[Localize] = None

#     def _classify(self, files: List[str], num_files: int, classifier_type: int) -> ClassificationResult:
#         """
#         Perform classification using the provided classifier type.

#         Args:
#             files: List of preprocessed file paths.
#             num_files: Number of files.
#             classifier_type: Type of classifier (1 for real/fake, 2 for injected/removed).

#         Returns:
#             ClassificationResult with status, classification, confidence, affected filenames, and optional error.
#         """
#         try:
#             self.classifier = TamperClassifiers(files, num_files, type=classifier_type)
#             status, res, affected_fnames, error = self.classifier.get_results()
#             if error:
#                 raise error

#             if status not in [200, 201, 206] or not isinstance(res, tuple) or len(res) < 2:
#                 raise ValueError(f"Invalid classification result: status={status}, res={res}")

#             classification_type, confidence = res[0], res[1]
#             return ClassificationResult(
#                 status=status,
#                 classification=classification_type,
#                 confidence=confidence,
#                 affected_filenames=affected_fnames
#             )
#         except Exception as e:
#             logger.error(f"Classification failed for type {classifier_type}: {e}")
#             return ClassificationResult(status=500, classification=None, confidence=0.0, affected_filenames=[], error=e)

#     def _localize(self, files: List[str], num_files: int, affected_fnames: List[str], localizer_type: int) -> List[LocalizationResult]:
#         """
#         Perform localization on affected filenames using the provided localizer type.

#         Args:
#             files: List of preprocessed file paths.
#             num_files: Number of files.
#             affected_fnames: Filenames to localize.
#             localizer_type: Type of localizer (1 for injected, 2 for removed).

#         Returns:
#             List of LocalizationResult objects.
#         """
#         if not affected_fnames:
#             return []

#         try:
#             self.localizer = Localize(files, num_files, affected_fnames, type=localizer_type)
#             reports = self.localizer.get_results()
#             if not isinstance(reports, list):
#                 raise ValueError(f"Invalid localization reports: {reports}")

#             localization_results = []
#             for report in reports:
#                 if not isinstance(report, dict):
#                     logger.warning(f"Skipping invalid report: {report}")
#                     continue

#                 result = LocalizationResult(
#                     filename=report.get("fname", ""),
#                     coords=report.get("coords", []),
#                     heatmap=report.get("heatmap", None)
#                 )
#                 if result.filename:  # Only add non-empty results
#                     localization_results.append(result)
#             return localization_results
#         except Exception as e:
#             logger.error(f"Localization failed for type {localizer_type}: {e}")
#             return []

#     def analyze_volume(self, sorted_files_list: List[str]) -> Tuple[int, Dict[str, Any]]:
#         """
#         Analyze a volume of CT slices for tampering.

#         Steps:
#         1. Preprocess the file list.
#         2. Classify as real or tampered.
#         3. If tampered, sub-classify into injected/removed and localize affected slices.

#         Args:
#             sorted_files_list: List of file paths to CT slices.

#         Returns:
#             Tuple of (status_code, result_dict) where result_dict contains classification
#             details, confidence, and optional localization data.
#         """
#         if not sorted_files_list:
#             return 400, {"error": "No files provided"}

#         try:
#             logger.info(f"Starting analysis on {len(sorted_files_list)} files")
#             preprocessed_files = preprocess(sorted_files_list)
#             if len(preprocessed_files) != len(sorted_files_list):
#                 logger.warning("Preprocessing reduced file count")

#             # Step 1: Binary classification (real vs. tampered)
#             binary_result = self._classify(preprocessed_files, len(preprocessed_files), classifier_type=1)
#             if binary_result.error:
#                 return 500, {"error": "Classification failed", "details": str(binary_result.error)}

#             if binary_result.classification == self.types.type1:  # Real
#                 return 200, {
#                     "classification": "Real",
#                     "confidence": binary_result.confidence,
#                 }

#             elif binary_result.classification == self.types.type2:  # Tampered
#                 # Step 2: Sub-classification (injected vs. removed)
#                 sub_result = self._classify(preprocessed_files, len(preprocessed_files), classifier_type=2)
#                 if sub_result.error:
#                     logger.warning("Sub-classification failed, returning partial results")
#                     return 206, {
#                         "classification": "Tampered",
#                         "sub_classification": None,
#                         "classification_confidence": binary_result.confidence,
#                         "sub_classification_confidence": None,
#                         "affected_slices": binary_result.affected_filenames,
#                     }

#                 # Step 3: Localization
#                 injected_fnames = sub_result.affected_filenames[0] if len(sub_result.affected_filenames) >= 2 else []
#                 removed_fnames = sub_result.affected_filenames[1] if len(sub_result.affected_filenames) >= 2 else []

#                 if len(sub_result.affected_filenames) != 2:
#                     logger.warning("Mismatch in sub-classification filenames shape")

#                 injected_localizations = self._localize(preprocessed_files, len(preprocessed_files), injected_fnames, localizer_type=1)
#                 removed_localizations = self._localize(preprocessed_files, len(preprocessed_files), removed_fnames, localizer_type=2)

#                 # Build detailed result
#                 detailed_data = {
#                     "total_count": len(preprocessed_files),
#                     "total_tampered": len(binary_result.affected_filenames),
#                     "tamper_confidence": binary_result.confidence,
#                     "sub_classification_confidence": sub_result.confidence,
#                     "tampered_samples_data": {
#                         "total_injected": len(injected_localizations),
#                         "total_removed": len(removed_localizations),
#                         "tampered_slices": binary_result.affected_filenames,
#                         "injected_localization": [
#                             {
#                                 "fname": loc.filename,
#                                 "coords": loc.coords,
#                                 "heatmap": loc.heatmap,
#                             }
#                             for loc in injected_localizations
#                         ],
#                         "removed_localization": [
#                             {
#                                 "fname": loc.filename,
#                                 "coords": loc.coords,
#                                 "heatmap": loc.heatmap,
#                             }
#                             for loc in removed_localizations
#                         ],
#                     },
#                 }

#                 return 200, {
#                     "classification": "Tampered",
#                     "classification_confidence": binary_result.confidence,
#                     "data": detailed_data,
#                 }

#             else:
#                 raise ValueError(f"Unknown classification: {binary_result.classification}")

#         except Exception as e:
#             logger.error(f"Pipeline failed: {e}")
#             return 500, {"error": "Pipeline execution failed", "details": str(e)}