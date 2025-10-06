"""
Model testing and backtesting module for evaluating different AI models.
Provides comprehensive testing framework for comparing model performance.
"""

import time
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .model_providers import ModelProviderFactory, BaseLLMProvider
from .rag_chain import GISRAGChain
from .retriever import GISDocumentRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTester:
    """
    Comprehensive model testing and evaluation framework.
    """

    def __init__(self, retriever: GISDocumentRetriever, test_data_file: Optional[str] = None):
        """
        Initialize the model tester.

        Args:
            retriever: Document retriever instance
            test_data_file: Path to test questions file (optional)
        """
        self.retriever = retriever
        self.test_data = []
        self.results = []

        # Load test questions
        if test_data_file and Path(test_data_file).exists():
            self.load_test_data(test_data_file)
        else:
            self.create_default_test_data()

    def load_test_data(self, file_path: str) -> None:
        """
        Load test questions from JSON file.

        Args:
            file_path: Path to test data file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.test_data = json.load(f)
            logger.info(f"Loaded {len(self.test_data)} test questions from {file_path}")
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            self.create_default_test_data()

    def create_default_test_data(self) -> None:
        """Create default test questions for GIS/Remote Sensing."""
        self.test_data = [
            {
                "id": 1,
                "question": "What is NDVI and how is it calculated from satellite imagery?",
                "category": "Vegetation Analysis",
                "difficulty": "Easy",
                "expected_keywords": ["NDVI", "Normalized Difference Vegetation Index", "NIR", "Red band", "vegetation"]
            },
            {
                "id": 2,
                "question": "What are the main differences between Sentinel-2 and Landsat-8 satellites?",
                "category": "Satellite Platforms",
                "difficulty": "Medium",
                "expected_keywords": ["Sentinel-2", "Landsat-8", "resolution", "spectral bands", "revisit time"]
            },
            {
                "id": 3,
                "question": "Explain the process of atmospheric correction in remote sensing.",
                "category": "Image Processing",
                "difficulty": "Hard",
                "expected_keywords": ["atmospheric correction", "scattering", "absorption", "aerosol", "water vapor"]
            },
            {
                "id": 4,
                "question": "How is supervised classification different from unsupervised classification?",
                "category": "Classification",
                "difficulty": "Medium",
                "expected_keywords": ["supervised", "unsupervised", "training data", "algorithms", "accuracy assessment"]
            },
            {
                "id": 5,
                "question": "What are the applications of LiDAR in forestry and urban planning?",
                "category": "LiDAR Applications",
                "difficulty": "Medium",
                "expected_keywords": ["LiDAR", "forestry", "urban planning", "3D modeling", "canopy height"]
            },
            {
                "id": 6,
                "question": "How does radar imagery differ from optical imagery?",
                "category": "Remote Sensing Basics",
                "difficulty": "Easy",
                "expected_keywords": ["radar", "optical", "active sensor", "passive sensor", "weather", "day/night"]
            },
            {
                "id": 7,
                "question": "What is the difference between map projection and coordinate system?",
                "category": "GIS Basics",
                "difficulty": "Easy",
                "expected_keywords": ["projection", "coordinate system", "latitude", "longitude", "distortion"]
            },
            {
                "id": 8,
                "question": "Explain change detection techniques in remote sensing.",
                "category": "Change Detection",
                "difficulty": "Hard",
                "expected_keywords": ["change detection", "time series", "classification", "thresholding", "accuracy"]
            }
        ]
        logger.info(f"Created {len(self.test_data)} default test questions")

    def test_model(
        self,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        test_subset: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Test a specific model on the test dataset.

        Args:
            model_name: Name of the model to test
            temperature: Model temperature
            max_tokens: Maximum tokens
            test_subset: List of test question IDs to run (optional)

        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing model: {model_name}")

        # Initialize model
        try:
            provider = ModelProviderFactory.create_provider(model_name, temperature, max_tokens)
            if not provider.is_available():
                return {
                    "model_name": model_name,
                    "error": "Model not available",
                    "test_results": []
                }
        except Exception as e:
            return {
                "model_name": model_name,
                "error": str(e),
                "test_results": []
            }

        # Initialize RAG chain
        rag_chain = GISRAGChain(
            retriever=self.retriever,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider
        )

        # Select test questions
        if test_subset:
            test_questions = [q for q in self.test_data if q["id"] in test_subset]
        else:
            test_questions = self.test_data

        test_results = []
        total_start_time = time.time()

        for question in test_questions:
            logger.info(f"Testing question {question['id']}: {question['question'][:50]}...")

            try:
                # Generate response
                start_time = time.time()
                result = rag_chain.generate_response(
                    query=question["question"],
                    top_k=3,
                    include_sources=True
                )
                response_time = time.time() - start_time

                # Evaluate response
                evaluation = self.evaluate_response(question, result)

                test_result = {
                    "question_id": question["id"],
                    "question": question["question"],
                    "category": question["category"],
                    "difficulty": question["difficulty"],
                    "response": result["response"],
                    "response_time": response_time,
                    "sources_used": len(result["sources"]),
                    "token_usage": result["token_usage"],
                    "evaluation": evaluation,
                    "model_info": result.get("model_provider", "Unknown"),
                    "timestamp": result["timestamp"]
                }

                test_results.append(test_result)

            except Exception as e:
                logger.error(f"Error testing question {question['id']}: {str(e)}")
                test_results.append({
                    "question_id": question["id"],
                    "question": question["question"],
                    "category": question["category"],
                    "difficulty": question["difficulty"],
                    "error": str(e),
                    "response_time": 0,
                    "sources_used": 0,
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "evaluation": {"relevance_score": 0, "completeness_score": 0, "accuracy_score": 0}
                })

        total_test_time = time.time() - total_start_time

        # Calculate summary statistics
        successful_tests = [r for r in test_results if "error" not in r]
        summary = self.calculate_summary_statistics(test_results, total_test_time)

        return {
            "model_name": model_name,
            "test_time": total_test_time,
            "total_questions": len(test_questions),
            "successful_tests": len(successful_tests),
            "summary": summary,
            "test_results": test_results
        }

    def evaluate_response(self, question: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model response quality.

        Args:
            question: Test question data
            result: Model result

        Returns:
            Dictionary with evaluation scores
        """
        response = result["response"].lower()
        expected_keywords = question.get("expected_keywords", [])

        # Relevance score (0-1)
        keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in response)
        relevance_score = min(keyword_matches / len(expected_keywords), 1.0) if expected_keywords else 0.5

        # Completeness score (0-1)
        response_length = len(response.split())
        if response_length < 50:
            completeness_score = 0.3
        elif response_length < 100:
            completeness_score = 0.6
        elif response_length < 200:
            completeness_score = 0.8
        else:
            completeness_score = 1.0

        # Source usage score (0-1)
        sources_count = len(result.get("sources", []))
        source_score = min(sources_count / 3.0, 1.0)  # Ideal is 3 sources

        # Technical accuracy score (0-1)
        technical_terms = ["calculate", "formula", "method", "technique", "algorithm", "resolution", "wavelength"]
        technical_matches = sum(1 for term in technical_terms if term in response)
        accuracy_score = min(technical_matches / 3.0, 1.0)

        # Overall score
        overall_score = (relevance_score * 0.4 + completeness_score * 0.3 + source_score * 0.2 + accuracy_score * 0.1)

        return {
            "relevance_score": relevance_score,
            "completeness_score": completeness_score,
            "source_score": source_score,
            "accuracy_score": accuracy_score,
            "overall_score": overall_score,
            "keyword_matches": keyword_matches,
            "total_keywords": len(expected_keywords)
        }

    def calculate_summary_statistics(self, test_results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """
        Calculate summary statistics for test results.

        Args:
            test_results: List of test results
            total_time: Total test time

        Returns:
            Dictionary with summary statistics
        """
        successful_tests = [r for r in test_results if "error" not in r]

        if not successful_tests:
            return {
                "success_rate": 0,
                "avg_response_time": 0,
                "avg_relevance_score": 0,
                "avg_completeness_score": 0,
                "avg_source_score": 0,
                "avg_accuracy_score": 0,
                "avg_overall_score": 0,
                "total_tokens_used": 0,
                "tests_per_minute": 0
            }

        # Calculate averages
        avg_response_time = sum(r["response_time"] for r in successful_tests) / len(successful_tests)
        avg_relevance_score = sum(r["evaluation"]["relevance_score"] for r in successful_tests) / len(successful_tests)
        avg_completeness_score = sum(r["evaluation"]["completeness_score"] for r in successful_tests) / len(successful_tests)
        avg_source_score = sum(r["evaluation"]["source_score"] for r in successful_tests) / len(successful_tests)
        avg_accuracy_score = sum(r["evaluation"]["accuracy_score"] for r in successful_tests) / len(successful_tests)
        avg_overall_score = sum(r["evaluation"]["overall_score"] for r in successful_tests) / len(successful_tests)

        total_tokens = sum(r["token_usage"]["total_tokens"] for r in successful_tests)
        tests_per_minute = len(successful_tests) / (total_time / 60) if total_time > 0 else 0

        # Performance by category
        category_stats = {}
        for result in successful_tests:
            category = result["category"]
            if category not in category_stats:
                category_stats[category] = {"scores": [], "count": 0}
            category_stats[category]["scores"].append(result["evaluation"]["overall_score"])
            category_stats[category]["count"] += 1

        for category in category_stats:
            scores = category_stats[category]["scores"]
            category_stats[category]["avg_score"] = sum(scores) / len(scores) if scores else 0

        return {
            "success_rate": len(successful_tests) / len(test_results),
            "avg_response_time": avg_response_time,
            "avg_relevance_score": avg_relevance_score,
            "avg_completeness_score": avg_completeness_score,
            "avg_source_score": avg_source_score,
            "avg_accuracy_score": avg_accuracy_score,
            "avg_overall_score": avg_overall_score,
            "total_tokens_used": total_tokens,
            "tests_per_minute": tests_per_minute,
            "category_performance": category_stats
        }

    def compare_models(
        self,
        model_names: List[str],
        temperature: float = 0.1,
        max_tokens: int = 1000,
        test_subset: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same test dataset.

        Args:
            model_names: List of model names to compare
            temperature: Model temperature
            max_tokens: Maximum tokens
            test_subset: List of test question IDs to run (optional)

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing models: {model_names}")

        comparison_results = {}
        comparison_start_time = time.time()

        for model_name in model_names:
            logger.info(f"Testing {model_name}...")
            result = self.test_model(model_name, temperature, max_tokens, test_subset)
            comparison_results[model_name] = result

        total_comparison_time = time.time() - comparison_start_time

        # Create comparison summary
        comparison_summary = self.create_comparison_summary(comparison_results)

        return {
            "models_tested": model_names,
            "total_comparison_time": total_comparison_time,
            "comparison_summary": comparison_summary,
            "detailed_results": comparison_results
        }

    def create_comparison_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary comparison of model performance.

        Args:
            comparison_results: Results from model comparison

        Returns:
            Dictionary with comparison summary
        """
        summary_data = []

        for model_name, result in comparison_results.items():
            if "error" not in result:
                summary = result["summary"]
                summary_data.append({
                    "model": model_name,
                    "success_rate": summary["success_rate"],
                    "avg_response_time": summary["avg_response_time"],
                    "avg_overall_score": summary["avg_overall_score"],
                    "avg_relevance_score": summary["avg_relevance_score"],
                    "avg_completeness_score": summary["avg_completeness_score"],
                    "avg_source_score": summary["avg_source_score"],
                    "avg_accuracy_score": summary["avg_accuracy_score"],
                    "tests_per_minute": summary["tests_per_minute"],
                    "total_tokens_used": summary["total_tokens_used"]
                })

        if not summary_data:
            return {"error": "No successful model tests to compare"}

        # Sort by overall score
        summary_data.sort(key=lambda x: x["avg_overall_score"], reverse=True)

        # Find best performing model for each metric
        best_models = {}
        metrics = ["success_rate", "avg_overall_score", "avg_relevance_score", "avg_completeness_score",
                  "avg_source_score", "avg_accuracy_score", "tests_per_minute"]

        for metric in metrics:
            best_model = min(summary_data, key=lambda x: -x[metric] if metric != "avg_response_time" else x[metric])
            best_models[f"best_{metric}"] = best_model["model"]

        return {
            "ranking": summary_data,
            "best_models": best_models,
            "total_models_compared": len(summary_data)
        }

    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save test results to file.

        Args:
            results: Test results to save
            filename: Output filename (optional)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/model_test_results_{timestamp}.json"

        output_path = Path(filename)
        output_path.parent.mkdir(exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return ""

    def generate_report(self, results: Dict[str, Any], output_dir: str = "logs") -> str:
        """
        Generate a comprehensive test report.

        Args:
            results: Test results
            output_dir: Output directory for report

        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(output_dir) / f"model_test_report_{timestamp}.html"

        try:
            html_content = self.create_html_report(results)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Report generated: {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return ""

    def create_html_report(self, results: Dict[str, Any]) -> str:
        """
        Create HTML report from test results.

        Args:
            results: Test results

        Returns:
            HTML content
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }
                .best { background-color: #e8f5e8; }
                .summary { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
        """

        if "comparison_summary" in results:
            # Comparison report
            html += "<h1>Model Comparison Report</h1>"
            summary = results["comparison_summary"]

            if "ranking" in summary:
                html += "<h2>Model Ranking</h2>"
                html += "<table>"
                html += "<tr><th>Rank</th><th>Model</th><th>Overall Score</th><th>Response Time</th><th>Success Rate</th></tr>"
                for i, model in enumerate(summary["ranking"], 1):
                    html += f"<tr><td>{i}</td><td>{model['model']}</td><td>{model['avg_overall_score']:.3f}</td><td>{model['avg_response_time']:.2f}s</td><td>{model['success_rate']:.2%}</td></tr>"
                html += "</table>"

        html += """
        </body>
        </html>
        """

        return html


# Example usage and testing functions
def run_comprehensive_test(retriever: GISDocumentRetriever) -> Dict[str, Any]:
    """
    Run a comprehensive test of all available models.

    Args:
        retriever: Document retriever instance

    Returns:
        Comprehensive test results
    """
    tester = ModelTester(retriever)

    # Get available models
    available_models = ModelProviderFactory.get_available_models()
    working_models = [name for name, info in available_models.items() if info.get("available", False)]

    if not working_models:
        logger.warning("No working models found for testing")
        return {"error": "No working models available"}

    logger.info(f"Testing {len(working_models)} models: {working_models}")

    # Run comparison
    results = tester.compare_models(working_models)

    # Save results
    tester.save_results(results)
    tester.generate_report(results)

    return results


if __name__ == "__main__":
    # Example usage
    retriever = GISDocumentRetriever()
    retriever.initialize_or_load()

    # Run tests
    results = run_comprehensive_test(retriever)

    print("Test Results Summary:")
    if "comparison_summary" in results:
        summary = results["comparison_summary"]
        if "ranking" in summary:
            print("\nModel Ranking:")
            for i, model in enumerate(summary["ranking"], 1):
                print(f"{i}. {model['model']}: Score {model['avg_overall_score']:.3f}")