# pipeline/get_results.py
"""
Tamper Detection Pipeline for CT Scan Volumes - Enhanced with Multi-Channel Real/Fake Classifier

This module orchestrates the preprocessing, classification, and localization
of tampering in CT scan slices. It supports detection of real vs. tampered
volumes and, for tampered volumes, sub-classification into injected or removed
tampering with localization of affected regions.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from pipeline.config.configs import ModelConfig
from pipeline.classifiers import TamperClassifiers
from pipeline.localize import Localize
from pipeline.preProces.preProcess import preprocess
from pipeline.types.types import Types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Enhanced result from tamper classification."""
    status: int
    classification: str
    confidence: float
    affected_filenames: List[str]
    slice_details: Optional[List[Dict]] = None
    volume_statistics: Optional[Dict] = None
    error: Optional[Exception] = None


@dataclass
class LocalizationResult:
    """Result from tamper localization."""
    filename: str
    coords: List[Tuple[float, float]]  # List of (x, y) coordinates for bounding boxes
    heatmap: Any  # Placeholder for heatmap data (e.g., numpy array or encoded string)


class TamperPipeline:
    """
    Enhanced pipeline for analyzing CT scan volumes for tampering.

    Orchestrates preprocessing, binary classification (real vs. tampered),
    sub-classification (injected vs. removed), and localization with
    multi-channel input support.
    """

    def __init__(self):
        self.types = Types()
        self.classifier: Optional[TamperClassifiers] = None
        self.localizer: Optional[Localize] = None
        self.config = ModelConfig() 

    def _classify(self, files: List[Dict], num_files: int, classifier_type: int) -> ClassificationResult:
        """
        Perform enhanced classification using the provided classifier type.

        Args:
            files: List of preprocessed file dicts with multi-channel data.
            num_files: Number of files.
            classifier_type: Type of classifier (1 for real/fake, 2 for injected/removed).

        Returns:
            ClassificationResult with comprehensive results.
        """
        try:
            self.classifier = TamperClassifiers(files, num_files, type=classifier_type)
            status, res, affected_fnames, error = self.classifier.get_results()
            
            if error:
                raise error

            if status not in [200, 201, 206]:
                raise ValueError(f"Classification failed with status: {status}")

            # Enhanced result parsing for real/fake classifier
            if classifier_type == 1:
                if isinstance(res, tuple) and len(res) >= 2:
                    classification_type, confidence = res[0], res[1]
                    
                    # Get additional details from the classifier object
                    slice_details = getattr(self.classifier.obj, 'last_slice_details', None)
                    volume_stats = getattr(self.classifier.obj, 'last_volume_stats', None)
                    
                    return ClassificationResult(
                        status=status,
                        classification=classification_type,
                        confidence=confidence,
                        affected_filenames=affected_fnames,
                        slice_details=slice_details,
                        volume_statistics=volume_stats
                    )
                else:
                    raise ValueError(f"Invalid real/fake classification result format: {res}")
                    
            elif classifier_type == 2:
                # For injected/removed classifier
                if isinstance(res, tuple) and len(res) >= 2:
                    classification_type, confidence = res[0], res[1]
                    return ClassificationResult(
                        status=status,
                        classification=classification_type,
                        confidence=confidence,
                        affected_filenames=affected_fnames
                    )
                else:
                    raise ValueError(f"Invalid injected/removed classification result format: {res}")
            else:
                raise ValueError(f"Unknown classifier type: {classifier_type}")
                
        except Exception as e:
            logger.error(f"Classification failed for type {classifier_type}: {e}")
            return ClassificationResult(
                status=500, 
                classification=None, 
                confidence=0.0, 
                affected_filenames=[], 
                error=e
            )

    def _localize(self, files: List[Dict], num_files: int, affected_fnames: List[str], localizer_type: int) -> List[LocalizationResult]:
        """
        Perform localization on affected filenames using the provided localizer type.

        Args:
            files: List of preprocessed file dicts.
            num_files: Number of files.
            affected_fnames: Filenames to localize.
            localizer_type: Type of localizer (1 for injected, 2 for removed).

        Returns:
            List of LocalizationResult objects.
        """
        if not affected_fnames:
            return []

        try:
            self.localizer = Localize(files, num_files, affected_fnames, type=localizer_type)
            reports = self.localizer.get_results()
            
            if not isinstance(reports, list):
                raise ValueError(f"Invalid localization reports: {reports}")

            localization_results = []
            for report in reports:
                if not isinstance(report, dict):
                    logger.warning(f"Skipping invalid report: {report}")
                    continue

                result = LocalizationResult(
                    filename=report.get("fname", ""),
                    coords=report.get("coords", []),
                    heatmap=report.get("heatmap", None)
                )
                if result.filename:  # Only add non-empty results
                    localization_results.append(result)
            print(localization_results)
                    
            logger.info(f"Localization completed: {len(localization_results)} results for type {localizer_type}")
            return localization_results
            
        except Exception as e:
            logger.error(f"Localization failed for type {localizer_type}: {e}")
            return []

    def _handle_real_case(self, binary_result: ClassificationResult) -> Tuple[int, Dict[str, Any]]:
        """
        Handle the case where volume is classified as Real.
        
        Args:
            binary_result: Results from binary classification.
            
        Returns:
            Formatted response for real volume.
        """
        return 200, {
            "classification": "Real",
            "confidence": binary_result.confidence,
            "slice_statistics": binary_result.volume_statistics or {},
            "slice_details": binary_result.slice_details or [],
            "message": "Volume classified as authentic with high confidence"
        }

    def _handle_tampered_case(self, preprocessed_files: List[Dict], binary_result: ClassificationResult) -> Tuple[int, Dict[str, Any]]:
        """
        Handle the case where volume is classified as Tampered.
        
        Args:
            preprocessed_files: Preprocessed file data.
            binary_result: Results from binary classification.
            
        Returns:
            Formatted response for tampered volume with sub-classification and localization.
        """
        try:
            # Step 2: Sub-classification (injected vs. removed)
            sub_result = self._classify(preprocessed_files, len(preprocessed_files), classifier_type=2)
            
            if sub_result.error:
                logger.warning("Sub-classification failed, returning partial results")
                return 206, {
                    "classification": "Tampered",
                    "classification_confidence": binary_result.confidence,
                    "sub_classification": None,
                    "sub_classification_confidence": None,
                    "affected_slices": binary_result.affected_filenames,
                    "slice_statistics": binary_result.volume_statistics or {},
                    "slice_details": binary_result.slice_details or [],
                    "warning": "Sub-classification failed, only binary classification available"
                }

            # Step 3: Localization based on sub-classification
            # Extract affected filenames for injected and removed
            injected_fnames = []
            removed_fnames = []
            
            if len(sub_result.affected_filenames) >= 2:
                injected_fnames = sub_result.affected_filenames[0]  # First element for injected
                removed_fnames = sub_result.affected_filenames[1]   # Second element for removed
            else:
                logger.warning("Unexpected affected_filenames structure in sub-classification")
                # Fallback: use binary affected filenames for both
                injected_fnames = binary_result.affected_filenames
                removed_fnames = binary_result.affected_filenames

            # Perform localization
            injected_localizations = self._localize(
                preprocessed_files, len(preprocessed_files), injected_fnames, localizer_type=1
            )
            removed_localizations = self._localize(
                preprocessed_files, len(preprocessed_files), removed_fnames, localizer_type=2
            )

            # Build comprehensive result
            detailed_data = {
                "total_slices": len(preprocessed_files),
                "tampered_slices_count": len(binary_result.affected_filenames),
                "binary_confidence": binary_result.confidence,
                "sub_classification_confidence": sub_result.confidence,
                "tamper_analysis": {
                    "injected": {
                        "count": len(injected_localizations),
                        "localized_slices": [
                            {
                                "filename": loc.filename,
                                "bounding_boxes": loc.coords,
                                "heatmap_available": loc.heatmap is not None
                            }
                            for loc in injected_localizations
                        ]
                    },
                    "removed": {
                        "count": len(removed_localizations),
                        "localized_slices": [
                            {
                                "filename": loc.filename,
                                "bounding_boxes": loc.coords,
                                "heatmap_available": loc.heatmap is not None
                            }
                            for loc in removed_localizations
                        ]
                    }
                },
                "slice_statistics": binary_result.volume_statistics or {},
                "detailed_slices": binary_result.slice_details or []
            }

            return 200, {
                "classification": "Tampered",
                "sub_classification": sub_result.classification,
                "confidence": {
                    "binary": binary_result.confidence,
                    "sub_classification": sub_result.confidence
                },
                "data": detailed_data,
                "message": f"Volume classified as tampered with {sub_result.classification} sub-type"
            }

        except Exception as e:
            logger.error(f"Error in tampered case handling: {e}")
            return 206, {
                "classification": "Tampered",
                "classification_confidence": binary_result.confidence,
                "sub_classification": None,
                "sub_classification_confidence": None,
                "affected_slices": binary_result.affected_filenames,
                "slice_statistics": binary_result.volume_statistics or {},
                "slice_details": binary_result.slice_details or [],
                "error": "Sub-classification or localization failed",
                "details": str(e)
            }

    def analyze_volume(self, sorted_files_list: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        """
        Enhanced analysis of a volume of CT slices for tampering with multi-channel support.

        Steps:
        1. Multi-channel preprocessing (CT, ROI, FFT)
        2. Enhanced binary classification (real vs. tampered) with slice-level details
        3. If tampered: sub-classification into injected/removed and localization

        Args:
            sorted_files_list: List of dicts with keys:
                - "fname": filename
                - "data": numpy array of CT slice data

        Returns:
            Tuple of (status_code, result_dict) containing:
            - For Real volumes: classification, confidence, slice statistics
            - For Tampered volumes: comprehensive analysis with localization data
        """
        if not sorted_files_list:
            return 400, {
                "error": "No files provided",
                "message": "Please provide a list of CT slice files for analysis"
            }

        try:
            logger.info(f"Starting enhanced multi-channel analysis on {len(sorted_files_list)} slices")
            
            # Validate input format
            for i, item in enumerate(sorted_files_list):
                if not isinstance(item, dict) or 'fname' not in item or 'data' not in item:
                    return 400, {
                        "error": "Invalid input format",
                        "message": f"Item {i} missing 'fname' or 'data' key. Expected dict with 'fname' and 'data' keys."
                    }
                if not isinstance(item['data'], np.ndarray):
                    return 400, {
                        "error": "Invalid data type",
                        "message": f"Item {i} data must be a numpy array"
                    }

            # Step 1: Multi-channel preprocessing (CT, ROI, FFT generation)
            logger.info("Starting multi-channel preprocessing...")
            preprocessed_files = preprocess(sorted_files_list)
            
            if not preprocessed_files:
                return 500, {
                    "error": "Preprocessing failed",
                    "message": "No valid files could be processed after preprocessing"
                }
                
            if len(preprocessed_files) != len(sorted_files_list):
                logger.warning(f"Preprocessing filtered files: {len(sorted_files_list)} -> {len(preprocessed_files)}")

            logger.info(f"Multi-channel preprocessing completed: {len(preprocessed_files)} files")

            # Step 2: Enhanced binary classification (real vs. tampered)
            logger.info("Starting real/fake classification with multi-channel model...")
            binary_result = self._classify(preprocessed_files, len(preprocessed_files), classifier_type=1)
            
            if binary_result.error:
                logger.error(f"Binary classification failed: {binary_result.error}")
                return 500, {
                    "error": "Classification failed",
                    "details": str(binary_result.error),
                    "message": "Real/Fake classification could not be completed"
                }

            logger.info(f"Binary classification result: {binary_result.classification} "
                       f"(confidence: {binary_result.confidence:.3f})")

            # Route based on classification result
            if binary_result.classification.upper() == self.types.type1:  # Real
                logger.info("Volume classified as Real - returning results")
                return self._handle_real_case(binary_result)

            elif binary_result.classification.upper() == self.types.type2:  # Tampered
                logger.info("Volume classified as Tampered - starting sub-classification and localization")
                return self._handle_tampered_case(preprocessed_files, binary_result)

            else:
                logger.error(f"Unknown classification result: {binary_result.classification}")
                return 500, {
                    "error": "Unknown classification result",
                    "message": f"Received unexpected classification: {binary_result.classification}",
                    "received_classification": binary_result.classification
                }

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return 500, {
                "error": "Pipeline execution failed",
                "details": str(e),
                "message": "An unexpected error occurred during analysis"
            }

    def get_detailed_slice_analysis(self, sorted_files_list: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        """
        Get detailed slice-by-slice analysis for debugging and detailed reporting.
        
        Args:
            sorted_files_list: List of dicts with "fname" and "data"
            
        Returns:
            Detailed analysis with per-slice probabilities and predictions
        """
        if not sorted_files_list:
            return 400, {"error": "No files provided"}

        try:
            # Preprocess files
            preprocessed_files = preprocess(sorted_files_list)
            if not preprocessed_files:
                return 500, {"error": "Preprocessing failed"}

            # Run classification to get slice details
            binary_result = self._classify(preprocessed_files, len(preprocessed_files), classifier_type=1)
            
            if binary_result.error:
                return 500, {"error": "Classification failed", "details": str(binary_result.error)}

            # Build detailed response
            detailed_response = {
                "volume_summary": {
                    "classification": binary_result.classification,
                    "confidence": binary_result.confidence,
                    "total_slices": len(preprocessed_files),
                    "tampered_slices": len(binary_result.affected_filenames)
                },
                "slice_details": binary_result.slice_details or [],
                "statistics": binary_result.volume_statistics or {}
            }

            return 200, detailed_response

        except Exception as e:
            logger.error(f"Detailed analysis failed: {e}")
            return 500, {"error": "Detailed analysis failed", "details": str(e)}


# Utility function for quick analysis
def create_tamper_pipeline() -> TamperPipeline:
    """
    Factory function to create and return a configured TamperPipeline instance.
    
    Returns:
        Configured TamperPipeline instance
    """
    return TamperPipeline()


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the enhanced pipeline
    def example_usage():
        pipeline = TamperPipeline()
        
        # Example input data (you would replace this with actual data)
        example_slices = [
            {
                "fname": "slice_001.dcm",
                "data": np.random.rand(512, 512).astype(np.float32)  # Example CT slice
            },
            {
                "fname": "slice_002.dcm", 
                "data": np.random.rand(512, 512).astype(np.float32)
            }
            # ... more slices
        ]
        
        # Analyze volume
        status, results = pipeline.analyze_volume(example_slices)
        
        print(f"Status: {status}")
        print("Results:")
        print(results)
        
        # For detailed analysis
        if status == 200 and results.get("classification") == "Tampered":
            print("\nDetailed tamper analysis available in 'data' field")
        
    # Run example
    example_usage()