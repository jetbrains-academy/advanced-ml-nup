# Advanced Machine Learning Course (NUP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive course on advanced machine learning techniques. 
This course provides hands-on experience with various unsupervised learning algorithms and other methods
through practical assignments with automated testing.

## Course Overview

This assignment covers advanced topics in machine learning, with a particular focus on:
- Data Aggregation and Processing
- Clustering Algorithms
- Expectation-Maximization (EM) Algorithm
- Semi-supervised Learning
- Topic Modeling

## Course Structure

The course is organized into several modules under the Unsupervised learning section:

1. **Intro**: Introduction to unsupervised learning concepts
2. **AggregationPoints10**: Techniques for data aggregation including:
   - Majority voting
   - Dawid-Skene method
3. **ClusterizationPoints20**: Advanced clustering algorithms
4. **EMforDSPoints30**: Expectation-Maximization algorithm for Data Science
5. **SemisupervisedPoints20**: Semi-supervised learning techniques
6. **TopicModelingPoints20**: Topic modeling and text analysis

## Usage

Each module contains:
- Task description (task.md)
- Implementation file (task.py or main.py)
- Test suite (tests/test_task.py)
- Additional resources when needed

To complete a task:
1. Read the task description in the respective module's task.md
2. Implement the required functionality in the implementation file
3. Run the tests to verify your solution

## Testing

The course uses unittest framework for automated testing. To run tests for a specific task:
```bash
python -m unittest Unsupervised/ModuleName/tests/test_task.py
```

## Author

Aleksandr Avdiushenko
