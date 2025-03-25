# Deep Learning Journey - Day 18: Exponential Weighted Average (EWA)

## Introduction
Exponential Weighted Average (EWA) is a technique used to smooth data points in a sequence by giving more weight to recent observations while exponentially decreasing the weight of older ones. This is widely used in deep learning and optimization algorithms to stabilize updates and improve convergence.

## Why Use Exponential Weighted Average?
- **Reduces noise**: Helps in smoothing out fluctuations in data.
- **Captures trends**: Provides a better understanding of the underlying patterns.
- **Enhances stability**: Used in optimization algorithms like Adam to stabilize weight updates.
- **Efficient memory usage**: Unlike simple moving averages, it does not require storing a fixed number of past values.

## What Was Implemented in This Notebook?
1. **Loaded and visualized temperature data**: Used the `DailyDelhiClimateTest.csv` dataset, focusing on date and mean temperature columns.
2. **Applied Exponential Weighted Average (EWA)**: Used the `ewm()` function with an alpha value of 0.9 to calculate the smoothed mean temperature.
3. **Visualized the results**:
   - Plotted raw temperature data points using red scatter points.
   - Plotted the EWA-smoothed curve in blue to observe the smoothing effect.

## Key Observations
- The EWA line is smoother than the original temperature data, reducing short-term fluctuations.
- The higher the alpha value (closer to 1), the more weight is given to recent data points.
- EWA helps in better trend identification while filtering out noise.

## Conclusion
Exponential Weighted Average is an important technique in both time series analysis and deep learning optimization. Understanding and implementing EWA helps in improving model stability and capturing meaningful patterns in noisy datasets.
