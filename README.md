# AI Job Market Analysis & Insights (2025–2026)

A comprehensive data analysis and interactive dashboard built with **Streamlit** to explore the evolving landscape of **Artificial Intelligence careers**.  
This project analyzes salaries, required skills, experience levels, and global demand trends for AI-related roles.

---

## Table of Contents

- Overview
- Key Features
- Dataset Description
- Installation & Setup
- Technologies Used
- Key Insights
- Author

---

## 🔍 Overview

This project performs **Exploratory Data Analysis (EDA)** on a dataset containing AI-related job postings.

The goal is to understand trends in the AI job market and answer questions such as:

- Which AI roles offer the highest average salaries?
- How does **years of experience** affect salary levels?
- Which **countries and industries** dominate the AI job market?
- What skills are most frequently required in AI jobs?
- Are **LLM-related roles** becoming more common?

The results are presented through an **interactive Streamlit dashboard** that allows users to explore the dataset visually.

---

## Key Features

### Interactive Dashboard
Built with **Streamlit**, allowing users to explore data through an interactive interface.

### Salary Analysis
- Distribution of annual salaries
- Highest-paying AI job titles
- Salary comparison across experience levels

### Experience Insights
Visualization of the relationship between **years of experience and salary** using statistical plots.

### Market Trends
Analysis of:

- Job categories
- Education requirements
- Geographic distribution of AI jobs
- Remote work availability


### Machine Learning Model
A simple **classification model** predicts whether a job role is related to **Large Language Models (LLM)** based on job characteristics.

---

## 📊 Dataset Description

The analysis is based on the dataset:
The dataset contains information about AI job postings including:

### Job Information
- Job title
- Job category
- Experience level
- Required education

### Financial Data
- Annual salary in USD
- Salary ranges
- AI salary premium percentage

### Geographic Data
- City
- Country

### Technical Indicators
- Required skills
- LLM-related role indicator

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/LanaAljuaid/ai-jobs-market-2025-2026-dashboard.git
cd ai-jobs-market-2025-2026-dashboard 
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Streamlit application
```bash
streamlit run testeda.py
```

## Technologies Used

Python – Core programming language

Streamlit – Interactive dashboard development

Pandas – Data manipulation and cleaning

NumPy – Numerical operations

Matplotlib – Data visualization

Seaborn – Statistical data visualization

Scikit-learn – Machine learning model
