"""Synthetic large dataset generator for Verifily benchmarking.

Generates deterministic datasets of various sizes for performance testing.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional


# Fixed seed for determinism
DEFAULT_SEED = 42

# SFT (Supervised Fine-Tuning) schema templates
SFT_TEMPLATES = [
    {
        "instruction": "Explain {concept} in simple terms.",
        "input": "",
        "output": "{concept} is a fundamental concept that refers to {explanation}.",
    },
    {
        "instruction": "Write a {style} about {topic}.",
        "input": "",
        "output": "Once upon a time, there was a {topic} that {action}. The end.",
    },
    {
        "instruction": "Translate the following to {language}:",
        "input": "Hello, how are you?",
        "output": "{translation}",
    },
    {
        "instruction": "Summarize the key points about {subject}.",
        "input": "",
        "output": "The main points about {subject} are: 1) {point1}, 2) {point2}, 3) {point3}.",
    },
    {
        "instruction": "What are the advantages and disadvantages of {topic}?",
        "input": "",
        "output": "Advantages: {advantages}. Disadvantages: {disadvantages}.",
    },
]

# QA (Question-Answering) schema templates
QA_TEMPLATES = [
    {
        "question": "What is {concept}?",
        "answer": "{concept} is {definition}.",
        "context": "{context}",
    },
    {
        "question": "How does {process} work?",
        "answer": "{process} works by {explanation}.",
        "context": "{context}",
    },
    {
        "question": "Why is {topic} important?",
        "answer": "{topic} is important because {reason}.",
        "context": "{context}",
    },
    {
        "question": "When did {event} happen?",
        "answer": "{event} happened in {year}.",
        "context": "{context}",
    },
    {
        "question": "Who invented {invention}?",
        "answer": "{invention} was invented by {inventor}.",
        "context": "{context}",
    },
]

# Fill-in values for templates
CONCEPTS = [
    "machine learning", "neural networks", "deep learning", "natural language processing",
    "computer vision", "reinforcement learning", "transformers", "attention mechanisms",
    "gradient descent", "backpropagation", "convolutional networks", "recurrent networks",
    "generative models", "autoencoders", "GANs", "diffusion models",
]

TOPICS = [
    "artificial intelligence", "robotics", "autonomous vehicles", "smart assistants",
    "recommendation systems", "fraud detection", "medical diagnosis", "climate modeling",
    "financial forecasting", "supply chain optimization", "content moderation",
]

LANGUAGES = ["Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Arabic"]

STYLES = ["short story", "poem", "blog post", "technical article", "tweet"]

SUBJECTS = [
    "quantum computing", "blockchain technology", "5G networks", "edge computing",
    "cloud architecture", "microservices", "serverless computing", "containerization",
    "DevOps practices", "Agile methodology", "API design", "database optimization",
]

ACTIONS = [
    "changed the world forever", "revolutionized the industry", "solved complex problems",
    "enabled new possibilities", "transformed daily life", "created opportunities",
]

TRANSLATIONS = [
    "Hola, ¿cómo estás?", "Bonjour, comment allez-vous?", "Hallo, wie geht es Ihnen?",
    "你好，你好吗？", "こんにちは、お元気ですか？", "안녕하세요, 어떻게 지내세요?",
]

DEFINITIONS = [
    "a method for teaching computers to learn from data",
    "an approach to solving complex problems efficiently",
    "a framework for organizing and processing information",
    "a technique for extracting insights from large datasets",
]

EXPLANATIONS = [
    "processing inputs through multiple layers of computation",
    "iteratively improving performance through feedback",
    "identifying patterns in historical data",
    "optimizing parameters to minimize error",
]

CONTEXTS = [
    "In the field of computer science, this concept has gained significant attention.",
    "Recent advances in technology have made this approach more practical.",
    "Researchers have been studying this topic for several decades.",
    "Industry applications have demonstrated the value of this methodology.",
]

EVENTS = [
    "the invention of the transistor", "the launch of the first satellite",
    "the creation of the internet", "the development of the first compiler",
    "the introduction of the smartphone", "the release of the first AI system",
]

YEARS = ["1947", "1957", "1969", "1983", "2007", "2012"]

INVENTIONS = [
    "the telephone", "the light bulb", "the airplane", "the computer",
    "the world wide web", "the transistor", "the laser",
]

INVENTORS = [
    "Alexander Graham Bell", "Thomas Edison", "the Wright brothers",
    "Charles Babbage", "Tim Berners-Lee", "John Bardeen",
]


def generate_sft_record(rng: random.Random, index: int) -> dict:
    """Generate a single SFT record."""
    template = rng.choice(SFT_TEMPLATES)
    
    concept = rng.choice(CONCEPTS)
    topic = rng.choice(TOPICS)
    style = rng.choice(STYLES)
    language = rng.choice(LANGUAGES)
    subject = rng.choice(SUBJECTS)
    action = rng.choice(ACTIONS)
    translation = rng.choice(TRANSLATIONS)
    explanation = rng.choice(EXPLANATIONS)
    
    record = {}
    for key, value in template.items():
        record[key] = value.format(
            concept=concept,
            topic=topic,
            style=style,
            language=language,
            subject=subject,
            action=action,
            translation=translation,
            explanation=explanation,
            point1=f"point A about {subject}",
            point2=f"point B about {subject}",
            point3=f"point C about {subject}",
            advantages=f"advantage 1 of {topic}, advantage 2",
            disadvantages=f"disadvantage 1 of {topic}, disadvantage 2",
        )
    
    # Add metadata
    record["metadata"] = {
        "index": index,
        "schema_version": "1.0",
        "source": "synthetic",
    }
    
    return record


def generate_qa_record(rng: random.Random, index: int) -> dict:
    """Generate a single QA record."""
    template = rng.choice(QA_TEMPLATES)
    
    concept = rng.choice(CONCEPTS)
    process = rng.choice(["training", "inference", "optimization", "evaluation"])
    topic = rng.choice(TOPICS)
    event = rng.choice(EVENTS)
    year = rng.choice(YEARS)
    invention = rng.choice(INVENTIONS)
    inventor = rng.choice(INVENTORS)
    definition = rng.choice(DEFINITIONS)
    explanation = rng.choice(EXPLANATIONS)
    context = rng.choice(CONTEXTS)
    reason = f"it enables {rng.choice(['efficiency', 'accuracy', 'scalability'])}"
    
    record = {}
    for key, value in template.items():
        record[key] = value.format(
            concept=concept,
            process=process,
            topic=topic,
            event=event,
            year=year,
            invention=invention,
            inventor=inventor,
            definition=definition,
            explanation=explanation,
            context=context,
            reason=reason,
        )
    
    # Add metadata
    record["metadata"] = {
        "index": index,
        "schema_version": "1.0",
        "source": "synthetic",
    }
    
    return record


def generate_dataset(
    output_path: Union[str, Path],
    num_rows: int,
    schema: str = "sft",
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> Path:
    """Generate a synthetic dataset.
    
    Args:
        output_path: Path for output JSONL file
        num_rows: Number of rows to generate
        schema: Schema type ("sft" or "qa")
        seed: Random seed for determinism
        verbose: Print progress
        
    Returns:
        Path to generated file
        
    Example:
        >>> generate_dataset("data/train.jsonl", num_rows=10000, schema="sft")
        Path('data/train.jsonl')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize RNG with seed for determinism
    rng = random.Random(seed)
    
    # Select generator based on schema
    if schema == "sft":
        generate_record = generate_sft_record
    elif schema == "qa":
        generate_record = generate_qa_record
    else:
        raise ValueError(f"Unknown schema: {schema}. Use 'sft' or 'qa'.")
    
    # Generate records
    if verbose:
        print(f"Generating {num_rows:,} {schema.upper()} records to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_rows):
            record = generate_record(rng, i)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            if verbose and (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1:,} rows...")
    
    # Get file stats
    file_size = output_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    if verbose:
        print(f"Done! Generated {num_rows:,} rows ({file_size_mb:.1f} MB)")
    
    return output_path


def generate_benchmark_suite(
    output_dir: Union[str, Path],
    sizes: Optional[List[int]] = None,
    schemas: Optional[List[str]] = None,
    seed: int = DEFAULT_SEED,
) -> List[Path]:
    """Generate a suite of benchmark datasets.
    
    Args:
        output_dir: Directory for output files
        sizes: List of row counts to generate
        schemas: List of schemas to generate
        seed: Random seed
        
    Returns:
        List of generated file paths
        
    Example:
        >>> generate_benchmark_suite("./benchmark_data", sizes=[1000, 10000, 50000])
        [Path('benchmark_data/sft_1k.jsonl'), ...]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if sizes is None:
        sizes = [1000, 10000, 50000]
    
    if schemas is None:
        schemas = ["sft", "qa"]
    
    generated_files = []
    
    print(f"Generating benchmark suite in {output_dir}")
    print(f"Sizes: {sizes}")
    print(f"Schemas: {schemas}")
    print()
    
    for schema in schemas:
        for size in sizes:
            size_label = f"{size // 1000}k" if size >= 1000 else f"{size}"
            filename = f"{schema}_{size_label}.jsonl"
            output_path = output_dir / filename
            
            generate_dataset(
                output_path=output_path,
                num_rows=size,
                schema=schema,
                seed=seed,
                verbose=True,
            )
            
            generated_files.append(output_path)
            print()
    
    print(f"Generated {len(generated_files)} datasets")
    return generated_files


def main():
    """CLI entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets for Verifily benchmarking"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file or directory",
    )
    parser.add_argument(
        "--rows",
        "-n",
        type=int,
        default=1000,
        help="Number of rows to generate (default: 1000)",
    )
    parser.add_argument(
        "--schema",
        "-s",
        type=str,
        default="sft",
        choices=["sft", "qa"],
        help="Schema type (default: sft)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for determinism (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Generate full benchmark suite (multiple sizes)",
    )
    
    args = parser.parse_args()
    
    if args.suite:
        generate_benchmark_suite(
            output_dir=args.output,
            seed=args.seed,
        )
    else:
        generate_dataset(
            output_path=args.output,
            num_rows=args.rows,
            schema=args.schema,
            seed=args.seed,
            verbose=True,
        )


if __name__ == "__main__":
    main()
