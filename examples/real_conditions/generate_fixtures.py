#!/usr/bin/env python3
"""Generate deterministic fixture data for the real_conditions example.

Run once to create all data/ and runs/ directories.
Seeded for reproducibility — output is identical across runs.
"""

import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

SEED = 42
random.seed(SEED)

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
RUNS_DIR = BASE / "runs"


# ── Training data (60 SFT/QA rows) ─────────────────────────────

TRAIN_ROWS = [
    {"instruction": "What is the capital of France?", "output": "The capital of France is Paris.", "input": "", "tags": {"source": "human", "domain": "geography"}},
    {"instruction": "Explain photosynthesis in one sentence.", "output": "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen.", "input": "", "tags": {"source": "human", "domain": "biology"}},
    {"instruction": "What is the boiling point of water at sea level?", "output": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    {"instruction": "Name the largest planet in our solar system.", "output": "Jupiter is the largest planet in our solar system.", "input": "", "tags": {"source": "human", "domain": "astronomy"}},
    {"instruction": "What is the chemical formula for table salt?", "output": "The chemical formula for table salt is NaCl (sodium chloride).", "input": "", "tags": {"source": "human", "domain": "chemistry"}},
    {"instruction": "Who wrote Romeo and Juliet?", "output": "William Shakespeare wrote Romeo and Juliet.", "input": "", "tags": {"source": "human", "domain": "literature"}},
    {"instruction": "What is the speed of light in vacuum?", "output": "The speed of light in vacuum is approximately 299,792,458 meters per second.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    {"instruction": "Define machine learning.", "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.", "input": "", "tags": {"source": "human", "domain": "cs"}},
    {"instruction": "What is the Pythagorean theorem?", "output": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a^2 + b^2 = c^2.", "input": "", "tags": {"source": "human", "domain": "math"}},
    {"instruction": "Explain the difference between DNA and RNA.", "output": "DNA is double-stranded and contains deoxyribose sugar with thymine, while RNA is single-stranded with ribose sugar and uracil instead of thymine.", "input": "", "tags": {"source": "human", "domain": "biology"}},
    {"instruction": "What is gradient descent?", "output": "Gradient descent is an optimization algorithm that iteratively adjusts parameters in the direction of steepest decrease of a loss function.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "Describe the water cycle.", "output": "The water cycle involves evaporation from surface water, condensation into clouds, precipitation as rain or snow, and collection in bodies of water.", "input": "", "tags": {"source": "human", "domain": "geography"}},
    {"instruction": "What is Newton's first law of motion?", "output": "Newton's first law states that an object at rest stays at rest, and an object in motion stays in motion, unless acted upon by an external force.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    {"instruction": "What is a neural network?", "output": "A neural network is a computing system inspired by biological neural networks, consisting of interconnected nodes organized in layers that process information.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "Explain the concept of supply and demand.", "output": "Supply and demand is an economic model where price is determined by the relationship between the quantity of a good available and the desire of buyers for it.", "input": "", "tags": {"source": "human", "domain": "economics"}},
    {"instruction": "What is the mitochondria's function?", "output": "The mitochondria are the powerhouses of the cell, responsible for producing adenosine triphosphate (ATP) through cellular respiration.", "input": "", "tags": {"source": "human", "domain": "biology"}},
    {"instruction": "Define overfitting in machine learning.", "output": "Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on unseen data.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the greenhouse effect?", "output": "The greenhouse effect is the process where certain gases in Earth's atmosphere trap heat from the sun, warming the planet's surface.", "input": "", "tags": {"source": "human", "domain": "geography"}},
    {"instruction": "Explain what a hash function does.", "output": "A hash function maps input data of arbitrary size to a fixed-size output, producing a deterministic digest that is infeasible to reverse.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What causes tides?", "output": "Tides are caused primarily by the gravitational pull of the Moon and Sun on Earth's oceans.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    {"instruction": "Describe the structure of an atom.", "output": "An atom consists of a nucleus containing protons and neutrons, surrounded by electrons orbiting in shells.", "input": "", "tags": {"source": "human", "domain": "chemistry"}},
    {"instruction": "What is cross-validation?", "output": "Cross-validation is a technique for evaluating model performance by partitioning data into training and test folds, training on each fold, and averaging results.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "Explain the theory of evolution.", "output": "Evolution is the process by which species change over time through natural selection, where organisms with favorable traits are more likely to survive and reproduce.", "input": "", "tags": {"source": "human", "domain": "biology"}},
    {"instruction": "What is a transformer model?", "output": "A transformer is a deep learning architecture that uses self-attention mechanisms to process sequential data in parallel, enabling efficient training on large datasets.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the quadratic formula?", "output": "The quadratic formula is x = (-b ± sqrt(b^2 - 4ac)) / 2a, used to find roots of ax^2 + bx + c = 0.", "input": "", "tags": {"source": "human", "domain": "math"}},
    {"instruction": "Describe how vaccines work.", "output": "Vaccines introduce a weakened or inactive form of a pathogen to stimulate the immune system to produce antibodies without causing the disease.", "input": "", "tags": {"source": "human", "domain": "biology"}},
    {"instruction": "What is regularization in machine learning?", "output": "Regularization adds a penalty term to the loss function to prevent model parameters from becoming too large, reducing overfitting.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "Explain plate tectonics.", "output": "Plate tectonics describes the movement of Earth's lithospheric plates on the asthenosphere, causing earthquakes, volcanic activity, and mountain formation.", "input": "", "tags": {"source": "human", "domain": "geography"}},
    {"instruction": "What is the difference between supervised and unsupervised learning?", "output": "Supervised learning uses labeled data to train models for prediction, while unsupervised learning finds patterns in unlabeled data.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is entropy in thermodynamics?", "output": "Entropy is a measure of disorder in a system; the second law of thermodynamics states that entropy in an isolated system always increases.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    {"instruction": "Explain the concept of recursion.", "output": "Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller subproblems until a base case is reached.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the Doppler effect?", "output": "The Doppler effect is the change in frequency of a wave relative to an observer moving relative to the source of the wave.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    {"instruction": "Define precision and recall.", "output": "Precision is the fraction of predicted positives that are truly positive. Recall is the fraction of actual positives that are correctly predicted.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the Fibonacci sequence?", "output": "The Fibonacci sequence is a series where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21, and so on.", "input": "", "tags": {"source": "human", "domain": "math"}},
    {"instruction": "Explain how antibiotics work.", "output": "Antibiotics kill or inhibit the growth of bacteria by targeting bacterial cell walls, protein synthesis, or DNA replication mechanisms.", "input": "", "tags": {"source": "human", "domain": "biology"}},
    {"instruction": "What is backpropagation?", "output": "Backpropagation is an algorithm for training neural networks that computes gradients of the loss function with respect to each weight by applying the chain rule.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What causes earthquakes?", "output": "Earthquakes are caused by the sudden release of energy in the Earth's crust, usually due to the movement of tectonic plates along fault lines.", "input": "", "tags": {"source": "human", "domain": "geography"}},
    {"instruction": "Explain the concept of a p-value.", "output": "A p-value is the probability of obtaining results at least as extreme as the observed results, assuming the null hypothesis is true.", "input": "", "tags": {"source": "human", "domain": "math"}},
    {"instruction": "What is a convolutional neural network?", "output": "A CNN is a deep learning model that uses convolutional layers to automatically learn spatial features from grid-like data such as images.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "Describe the process of mitosis.", "output": "Mitosis is cell division where a single cell divides to produce two identical daughter cells, involving prophase, metaphase, anaphase, and telophase.", "input": "", "tags": {"source": "human", "domain": "biology"}},
    {"instruction": "What is the halting problem?", "output": "The halting problem asks whether a given program will eventually halt or run forever; Alan Turing proved it is undecidable in general.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "Explain Ohm's law.", "output": "Ohm's law states that the current through a conductor is directly proportional to the voltage across it: V = IR.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    {"instruction": "What is the central limit theorem?", "output": "The central limit theorem states that the distribution of sample means approaches a normal distribution as the sample size increases, regardless of the population distribution.", "input": "", "tags": {"source": "human", "domain": "math"}},
    {"instruction": "Define batch normalization.", "output": "Batch normalization normalizes layer inputs across a mini-batch, stabilizing and accelerating deep network training by reducing internal covariate shift.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the pH scale?", "output": "The pH scale measures the acidity or basicity of a solution, ranging from 0 (most acidic) to 14 (most basic), with 7 being neutral.", "input": "", "tags": {"source": "human", "domain": "chemistry"}},
    {"instruction": "Explain the CAP theorem.", "output": "The CAP theorem states that a distributed system cannot simultaneously guarantee consistency, availability, and partition tolerance; at most two can be achieved.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is natural selection?", "output": "Natural selection is the process where organisms with traits better suited to their environment tend to survive and reproduce more, passing those traits to offspring.", "input": "", "tags": {"source": "human", "domain": "biology"}},
    {"instruction": "Describe the OSI model.", "output": "The OSI model is a seven-layer framework for network communication: Physical, Data Link, Network, Transport, Session, Presentation, and Application.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is Avogadro's number?", "output": "Avogadro's number is approximately 6.022 x 10^23, representing the number of atoms or molecules in one mole of a substance.", "input": "", "tags": {"source": "human", "domain": "chemistry"}},
    {"instruction": "Explain the bias-variance tradeoff.", "output": "The bias-variance tradeoff describes the balance between a model's ability to fit training data (low bias) and its generalization to unseen data (low variance).", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the Heisenberg uncertainty principle?", "output": "The Heisenberg uncertainty principle states that it is impossible to simultaneously know both the exact position and momentum of a particle.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    {"instruction": "Define transfer learning.", "output": "Transfer learning is a technique where a model trained on one task is reused as the starting point for a model on a different but related task.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the Krebs cycle?", "output": "The Krebs cycle is a series of chemical reactions in cellular respiration that generates energy through the oxidation of acetyl-CoA derived from carbohydrates, fats, and proteins.", "input": "", "tags": {"source": "human", "domain": "biology"}},
    {"instruction": "Explain what an API is.", "output": "An API (Application Programming Interface) is a set of protocols and tools that allows different software applications to communicate with each other.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the Coriolis effect?", "output": "The Coriolis effect is the deflection of moving objects caused by Earth's rotation, influencing weather patterns and ocean currents.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    {"instruction": "Describe the difference between TCP and UDP.", "output": "TCP provides reliable, ordered delivery of data with error checking, while UDP is faster but unreliable, with no guarantee of delivery or order.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is Le Chatelier's principle?", "output": "Le Chatelier's principle states that if a system at equilibrium is disturbed, it will shift to counteract the disturbance and restore equilibrium.", "input": "", "tags": {"source": "human", "domain": "chemistry"}},
    {"instruction": "Define the term 'epoch' in deep learning.", "output": "An epoch is one complete pass through the entire training dataset during the training of a neural network.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the electromagnetic spectrum?", "output": "The electromagnetic spectrum is the range of all types of electromagnetic radiation, from radio waves to gamma rays, ordered by wavelength and frequency.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    {"instruction": "Explain dropout regularization.", "output": "Dropout randomly deactivates a fraction of neurons during training, forcing the network to learn redundant representations and reducing overfitting.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
]

assert len(TRAIN_ROWS) == 60, f"Expected 60 train rows, got {len(TRAIN_ROWS)}"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _row_text(row: dict) -> str:
    return f"{row.get('instruction', '')} {row.get('output', '')} {row.get('input', '')}".strip()


# ── Clean eval (fully disjoint) ─────────────────────────────────

EVAL_CLEAN = [
    {"instruction": "What is the speed of sound in air?", "output": "The speed of sound in air is approximately 343 meters per second at room temperature.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    {"instruction": "Explain the concept of a black hole.", "output": "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape from it.", "input": "", "tags": {"source": "human", "domain": "astronomy"}},
    {"instruction": "What is the Richter scale?", "output": "The Richter scale is a logarithmic scale used to measure the magnitude of earthquakes.", "input": "", "tags": {"source": "human", "domain": "geography"}},
    {"instruction": "Describe the function of white blood cells.", "output": "White blood cells are part of the immune system and help the body fight infections and diseases.", "input": "", "tags": {"source": "human", "domain": "biology"}},
    {"instruction": "What is Bayes' theorem?", "output": "Bayes' theorem describes the probability of an event based on prior knowledge of conditions related to the event: P(A|B) = P(B|A)P(A)/P(B).", "input": "", "tags": {"source": "human", "domain": "math"}},
    {"instruction": "Explain the concept of a database index.", "output": "A database index is a data structure that improves the speed of data retrieval operations by providing quick lookup paths.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the difference between weather and climate?", "output": "Weather refers to short-term atmospheric conditions, while climate describes long-term average weather patterns over a region.", "input": "", "tags": {"source": "human", "domain": "geography"}},
    {"instruction": "Define the concept of a stack in programming.", "output": "A stack is a linear data structure that follows Last In First Out (LIFO) principle, where elements are added and removed from the top.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the function of the liver?", "output": "The liver detoxifies chemicals, metabolizes drugs, produces bile for digestion, and stores vitamins and minerals.", "input": "", "tags": {"source": "human", "domain": "biology"}},
    {"instruction": "Explain how a compiler works.", "output": "A compiler translates source code written in a high-level programming language into machine code that a computer processor can execute.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the law of conservation of energy?", "output": "The law of conservation of energy states that energy cannot be created or destroyed, only transformed from one form to another.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    {"instruction": "Describe the concept of a linked list.", "output": "A linked list is a linear data structure where each element points to the next, allowing dynamic memory allocation and efficient insertions.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is photovoltaic effect?", "output": "The photovoltaic effect is the creation of voltage in a material upon exposure to light, used in solar cells to convert sunlight into electricity.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    {"instruction": "Explain the concept of containerization.", "output": "Containerization packages software with its dependencies into isolated containers that run consistently across different computing environments.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the structure of the human heart?", "output": "The human heart has four chambers: two atria that receive blood and two ventricles that pump blood to the lungs and body.", "input": "", "tags": {"source": "human", "domain": "biology"}},
    {"instruction": "Define the term 'polymorphism' in OOP.", "output": "Polymorphism allows objects of different classes to be treated as instances of a common parent class, enabling flexible code design.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the ozone layer?", "output": "The ozone layer is a region of Earth's stratosphere that absorbs most of the Sun's ultraviolet radiation, protecting life on Earth.", "input": "", "tags": {"source": "human", "domain": "geography"}},
    {"instruction": "Explain MapReduce.", "output": "MapReduce is a programming model for processing large datasets in parallel: the Map step processes key-value pairs, and the Reduce step aggregates results.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    {"instruction": "What is the difference between DNA replication and transcription?", "output": "DNA replication copies the entire DNA molecule, while transcription copies a specific gene segment into messenger RNA for protein synthesis.", "input": "", "tags": {"source": "human", "domain": "biology"}},
    {"instruction": "Define the term 'idempotent' in computing.", "output": "An operation is idempotent if performing it multiple times produces the same result as performing it once.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
]

assert len(EVAL_CLEAN) == 20


# ── Exact leak eval (copies from train) ─────────────────────────

def make_exact_leak_eval():
    """Create eval set with exact copies from train (>= 15% overlap)."""
    # Take 4 exact copies from train (4/20 = 20% overlap)
    exact_indices = [0, 5, 12, 24]  # deterministic selection
    leaked = [dict(TRAIN_ROWS[i]) for i in exact_indices]

    # Fill remaining 16 with unique eval rows
    unique = [
        {"instruction": "What is the atomic number of carbon?", "output": "The atomic number of carbon is 6.", "input": "", "tags": {"source": "human", "domain": "chemistry"}},
        {"instruction": "Explain the concept of inflation.", "output": "Inflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power.", "input": "", "tags": {"source": "human", "domain": "economics"}},
        {"instruction": "What is a binary search tree?", "output": "A binary search tree is a data structure where each node has at most two children, with left values less than parent and right values greater.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        {"instruction": "Describe the nitrogen cycle.", "output": "The nitrogen cycle converts atmospheric nitrogen into usable forms through fixation, nitrification, assimilation, and denitrification.", "input": "", "tags": {"source": "human", "domain": "biology"}},
        {"instruction": "What is the difference between HTTP and HTTPS?", "output": "HTTPS adds TLS/SSL encryption to HTTP, securing data transmission between client and server.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        {"instruction": "Explain the concept of entropy in information theory.", "output": "In information theory, entropy measures the average amount of information or uncertainty in a set of possible messages.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        {"instruction": "What is the function of hemoglobin?", "output": "Hemoglobin is a protein in red blood cells that binds to oxygen and transports it from the lungs to tissues throughout the body.", "input": "", "tags": {"source": "human", "domain": "biology"}},
        {"instruction": "Define the Liskov substitution principle.", "output": "The Liskov substitution principle states that objects of a superclass should be replaceable with objects of a subclass without breaking the program.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        {"instruction": "What is nuclear fission?", "output": "Nuclear fission is a reaction in which a heavy atomic nucleus splits into two lighter nuclei, releasing a large amount of energy.", "input": "", "tags": {"source": "human", "domain": "physics"}},
        {"instruction": "Explain the concept of a message queue.", "output": "A message queue is a communication method where messages are stored in a queue until the receiving application processes them.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        {"instruction": "What is the role of enzymes?", "output": "Enzymes are biological catalysts that speed up chemical reactions in living organisms by lowering the activation energy.", "input": "", "tags": {"source": "human", "domain": "biology"}},
        {"instruction": "Describe the concept of sharding.", "output": "Sharding is a database architecture pattern that distributes data across multiple database instances to improve scalability.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        {"instruction": "What is the Bernoulli principle?", "output": "Bernoulli's principle states that an increase in the speed of a fluid occurs simultaneously with a decrease in pressure.", "input": "", "tags": {"source": "human", "domain": "physics"}},
        {"instruction": "Explain eventual consistency.", "output": "Eventual consistency is a consistency model where, given enough time without new updates, all replicas converge to the same state.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        {"instruction": "What is CRISPR?", "output": "CRISPR is a gene-editing technology that allows scientists to modify DNA sequences precisely by cutting and replacing specific genetic material.", "input": "", "tags": {"source": "human", "domain": "biology"}},
        {"instruction": "Define the concept of a bloom filter.", "output": "A Bloom filter is a space-efficient probabilistic data structure that tests whether an element is a member of a set, with possible false positives but no false negatives.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    ]

    return leaked + unique


# ── Near leak eval (paraphrased from train) ─────────────────────

def make_near_leak_eval():
    """Create eval set with paraphrased near-duplicates from train (>= 30% overlap)."""
    # Take 7 rows from train and make near-duplicates (7/20 = 35% overlap)
    # Strategy: keep most of the original text, change only a few words so
    # character 3-gram Jaccard stays above 0.70 but rows are not exact matches.
    paraphrased = [
        # Near-duplicate of train[0]: "What is the capital of France?"
        {"instruction": "What is the capital of France?", "output": "The capital of France is Paris, the largest city in France.", "input": "", "tags": {"source": "synthetic", "domain": "geography"}},
        # Near-duplicate of train[1]: "Explain photosynthesis in one sentence."
        {"instruction": "Explain photosynthesis in a single sentence.", "output": "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen molecules.", "input": "", "tags": {"source": "synthetic", "domain": "biology"}},
        # Near-duplicate of train[3]: "Name the largest planet in our solar system."
        {"instruction": "Name the largest planet in our solar system.", "output": "Jupiter is the largest and most massive planet in our solar system.", "input": "", "tags": {"source": "synthetic", "domain": "astronomy"}},
        # Near-duplicate of train[7]: "Define machine learning."
        {"instruction": "Define machine learning.", "output": "Machine learning is a subset of artificial intelligence that enables computer systems to learn and improve from experience without being explicitly programmed to do so.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        # Near-duplicate of train[8]: "What is the Pythagorean theorem?"
        {"instruction": "What is the Pythagorean theorem?", "output": "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a^2 + b^2 = c^2.", "input": "", "tags": {"source": "synthetic", "domain": "math"}},
        # Near-duplicate of train[13]: "What is a neural network?"
        {"instruction": "What is a neural network?", "output": "A neural network is a computing system inspired by biological neural networks, consisting of interconnected nodes organized in layers that process and learn from information.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        # Near-duplicate of train[16]: "Define overfitting in machine learning."
        {"instruction": "Define overfitting in machine learning.", "output": "Overfitting occurs when a model learns the training data too well, including noise and outliers, and performs poorly on new unseen data.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
    ]

    # Fill remaining 13 with unique rows
    unique = [
        {"instruction": "What is the significance of the Rosetta Stone?", "output": "The Rosetta Stone enabled scholars to decipher Egyptian hieroglyphics by providing the same text in three scripts.", "input": "", "tags": {"source": "human", "domain": "history"}},
        {"instruction": "Explain how a transistor works.", "output": "A transistor is a semiconductor device that amplifies or switches electronic signals by controlling current flow between two terminals.", "input": "", "tags": {"source": "human", "domain": "physics"}},
        {"instruction": "What is the Turing test?", "output": "The Turing test evaluates whether a machine can exhibit intelligent behavior indistinguishable from a human in conversation.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        {"instruction": "Describe the carbon cycle.", "output": "The carbon cycle is the process by which carbon moves through Earth's atmosphere, oceans, soil, and living organisms.", "input": "", "tags": {"source": "human", "domain": "geography"}},
        {"instruction": "What is a deadlock in operating systems?", "output": "A deadlock occurs when two or more processes are blocked forever, each waiting for a resource held by the other.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        {"instruction": "Explain the concept of half-life.", "output": "Half-life is the time required for half the atoms of a radioactive substance to decay.", "input": "", "tags": {"source": "human", "domain": "physics"}},
        {"instruction": "What is the function of the kidneys?", "output": "The kidneys filter blood to remove waste products, regulate fluid balance, and maintain electrolyte levels.", "input": "", "tags": {"source": "human", "domain": "biology"}},
        {"instruction": "Define the SOLID principles.", "output": "SOLID is an acronym for five design principles: Single Responsibility, Open-Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        {"instruction": "What is the Hubble constant?", "output": "The Hubble constant is the rate at which the universe is expanding, currently estimated at about 70 km/s per megaparsec.", "input": "", "tags": {"source": "human", "domain": "astronomy"}},
        {"instruction": "Explain consensus algorithms.", "output": "Consensus algorithms enable distributed systems to agree on a single data value, ensuring reliability despite node failures. Examples include Raft and Paxos.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        {"instruction": "What is the difference between mitosis and meiosis?", "output": "Mitosis produces two identical diploid cells for growth, while meiosis produces four unique haploid cells for reproduction.", "input": "", "tags": {"source": "human", "domain": "biology"}},
        {"instruction": "Describe the concept of a hash table.", "output": "A hash table stores key-value pairs using a hash function to compute indices, providing average O(1) lookup time.", "input": "", "tags": {"source": "synthetic", "domain": "cs"}},
        {"instruction": "What is the standard model of particle physics?", "output": "The Standard Model describes the fundamental particles and forces of the universe, including quarks, leptons, and gauge bosons.", "input": "", "tags": {"source": "human", "domain": "physics"}},
    ]

    return paraphrased + unique


# ── Run directories ─────────────────────────────────────────────

def make_run(name, f1, exact_match, status="completed", seed=42,
             started_offset=0):
    """Create a synthetic run directory with all contract files."""
    run_dir = RUNS_DIR / name
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    ts_base = "2026-02-08T10:00:00Z"

    # config.yaml
    config = {
        "task": "sft",
        "base_model": "google/flan-t5-base",
        "seed": seed,
        "data_paths": {
            "train": "data/train.jsonl",
            "test": "data/eval_clean.jsonl",
        },
        "training": {
            "num_epochs": 3,
            "batch_size": 8,
            "learning_rate": 0.0002,
            "max_seq_length": 512,
        },
        "lora": {
            "enabled": True,
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
        },
        "output": {
            "dir": f"runs/{name}",
            "save_adapter_only": True,
        },
    }

    import yaml
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # environment.json
    env = {
        "python": "3.11.7",
        "torch": "2.2.0",
        "transformers": "4.38.0",
        "peft": "0.9.0",
        "platform": "Darwin",
        "arch": "arm64",
        "device": "mps",
    }
    with open(run_dir / "environment.json", "w") as f:
        json.dump(env, f, indent=2)

    # eval/eval_results.json
    eval_results = {
        "run_id": name,
        "test_data_path": "data/eval_clean.jsonl",
        "num_examples": 20,
        "overall": {
            "exact_match": exact_match,
            "f1": f1,
        },
        "slices": {
            "source": {
                "human": {"n": 12, "exact_match": exact_match + 0.01, "f1": f1 + 0.01},
                "synthetic": {"n": 8, "exact_match": exact_match - 0.01, "f1": f1 - 0.01},
            },
            "domain": {
                "cs": {"n": 6, "exact_match": exact_match - 0.02, "f1": f1 - 0.02},
                "biology": {"n": 5, "exact_match": exact_match + 0.02, "f1": f1 + 0.02},
                "physics": {"n": 4, "exact_match": exact_match, "f1": f1},
                "other": {"n": 5, "exact_match": exact_match + 0.01, "f1": f1 + 0.01},
            },
        },
        "hard_examples": [],
        "eval_duration_seconds": 2.3 + random.random(),
    }
    with open(eval_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # run_meta.json
    meta = {
        "run_id": name,
        "status": status,
        "task": "sft",
        "base_model": "google/flan-t5-base",
        "dataset_version": "train_v1",
        "compute_mode": "local",
        "device": "mps",
        "started_at": ts_base,
        "completed_at": ts_base,
        "duration_seconds": 120.0 + random.random() * 60,
        "metrics": {"train_loss": 0.45 + random.random() * 0.1},
        "artifact_path": f"runs/{name}",
        "config_hash": f"sha256:{_sha(json.dumps(config, sort_keys=True))}",
        "data_hash": f"sha256:{_sha('train_data_v1')}",
        "environment_hash": f"sha256:{_sha(json.dumps(env, sort_keys=True))}",
        "reproducibility_hash": f"sha256:{_sha(f'{name}_{seed}')}",
        "seed": seed,
        "run_name": name,
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # hashes.json — compute real hashes of the files we just wrote
    files_to_hash = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name not in ("hashes.json", "manifest.json"):
            rel = str(p.relative_to(run_dir))
            files_to_hash[rel] = _sha(p.read_text())

    chain_parts = "|".join(f"{k}={v}" for k, v in sorted(files_to_hash.items()))
    chain_hash = _sha(chain_parts)

    hashes = {
        "files": files_to_hash,
        "chain_hash": chain_hash,
    }
    with open(run_dir / "hashes.json", "w") as f:
        json.dump(hashes, f, indent=2)


# ── verifily.yaml (pipeline config) ────────────────────────────

def make_pipeline_configs():
    """Create pipeline config files for different scenarios."""
    # Main config — leaked eval, should DONT_SHIP
    leaked_config = {
        "run_dir": "runs/run_01_good",
        "train_data": "data/train.jsonl",
        "eval_data": "data/eval_leaked_exact.jsonl",
        "baseline_run": "runs/run_01_good",
        "ship_if": {
            "min_f1": 0.65,
            "min_exact_match": 0.50,
            "max_f1_regression": 0.03,
            "max_pii_hits": 0,
        },
    }

    # Clean config — should SHIP
    clean_config = {
        "run_dir": "runs/run_01_good",
        "train_data": "data/train.jsonl",
        "eval_data": "data/eval_clean.jsonl",
        "baseline_run": "runs/run_01_good",
        "ship_if": {
            "min_f1": 0.65,
            "min_exact_match": 0.50,
            "max_f1_regression": 0.03,
            "max_pii_hits": 0,
        },
    }

    import yaml
    with open(BASE / "verifily.yaml", "w") as f:
        yaml.dump(leaked_config, f, default_flow_style=False, sort_keys=False)

    with open(BASE / "verifily_clean.yaml", "w") as f:
        yaml.dump(clean_config, f, default_flow_style=False, sort_keys=False)


# ── Main ────────────────────────────────────────────────────────

def main():
    print("Generating real_conditions fixtures (seed=42)...\n")

    # Data
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, rows):
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {path.relative_to(BASE)}: {len(rows)} rows")

    write_jsonl(DATA_DIR / "train.jsonl", TRAIN_ROWS)
    write_jsonl(DATA_DIR / "eval_clean.jsonl", EVAL_CLEAN)
    write_jsonl(DATA_DIR / "eval_leaked_exact.jsonl", make_exact_leak_eval())
    write_jsonl(DATA_DIR / "eval_leaked_near.jsonl", make_near_leak_eval())

    # Runs
    #   run_01_good: stable baseline f1=0.7139
    #   run_02_good: slightly better f1=0.7201
    #   run_03_regression: regression f1=0.6650 (drop = 0.0551 > 0.02 threshold)
    #   run_04_recovery: recovery f1=0.7050
    print("\nCreating run directories:")
    make_run("run_01_good",       f1=0.7139, exact_match=0.5945, seed=42)
    print("  runs/run_01_good: f1=0.7139 (baseline)")
    make_run("run_02_good",       f1=0.7201, exact_match=0.6010, seed=43)
    print("  runs/run_02_good: f1=0.7201 (stable/improved)")
    make_run("run_03_regression", f1=0.6650, exact_match=0.5400, seed=44)
    print("  runs/run_03_regression: f1=0.6650 (REGRESSION)")
    make_run("run_04_recovery",   f1=0.7050, exact_match=0.5850, seed=45)
    print("  runs/run_04_recovery: f1=0.7050 (recovery)")

    # Pipeline configs
    make_pipeline_configs()
    print("\n  verifily.yaml (leaked eval → DONT_SHIP)")
    print("  verifily_clean.yaml (clean eval → SHIP)")

    print("\nDone! All fixtures generated.\n")


if __name__ == "__main__":
    main()
