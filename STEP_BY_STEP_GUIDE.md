# 📖 Step-by-Step Execution Guide - Task 2

## Complete walkthrough for Task 2: Semantic Product Search and Ranking

---

## 🎯 Overview

This guide will walk you through the complete implementation process, from setup to submission.

**Total Time Estimate:** 4-6 hours (including training)

**Requirements:**

- Windows 11 PC (8GB RAM, Intel i5-12th Gen)
- Kaggle account (for GPU training)
- Python 3.8+
- VS Code
- Internet connection

---

## 📅 Day-by-Day Plan

### **Day 1-2: Setup and Data Preparation** (2-3 hours)

- Local environment setup
- Dataset download
- Data exploration

### **Day 3-4: Model Training on Kaggle** (3-4 hours)

- Upload code to Kaggle
- Train model (automatic, ~3 hours)
- Download results

### **Day 5: Local Deployment** (1 hour)

- Setup local files
- Run web application
- Test functionality

### **Day 6-7: Documentation** (2-3 hours)

- Write LaTeX report
- Generate visualizations
- Prepare submission

---

## 🚀 Phase 1: Local Environment Setup

### Step 1.1: Create Project Directory

Open PowerShell/CMD and run:

```powershell
# Create project directory
mkdir task2_semantic_search
cd task2_semantic_search

# Create subdirectories
mkdir data\raw
mkdir data\processed
mkdir models
mkdir src
mkdir kaggle_training
```

### Step 1.2: Setup Python Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Verify activation (you should see (venv) in prompt)
```

### Step 1.3: Save Project Files

**IMPORTANT:** Save all the files I provided in their correct locations:

```
task2_semantic_search/
├── src/
│   ├── data_loader.py          # Copy from File 5
│   ├── preprocessor.py         # Copy from File 6
│   ├── model.py                # Copy from File 7
│   ├── evaluate.py             # Copy from File 8
│   └── utils.py                # Copy from File 9
├── app.py                      # Copy from File 10
├── setup_local.py              # Copy from File 11
├── requirements.txt            # Copy from File 2
├── requirements_kaggle.txt     # Copy from File 3
├── README.md                   # Copy from File 1
├── prompts.txt                 # Copy from File 12 (fill in your details)
└── STEP_BY_STEP_GUIDE.md       # This file
```

### Step 1.4: Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt

# This will install:
# - pandas, numpy (data processing)
# - torch (deep learning)
# - transformers (BERT models)
# - sentence-transformers (embeddings)
# - streamlit (web interface)
# - nltk (text preprocessing)
# - scikit-learn (metrics)
# - ranx (ranking metrics)
# - and more...

# Wait for installation to complete (~5-10 minutes)
```

### Step 1.5: Download NLTK Data

```powershell
# Run Python and download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"
```

### Step 1.6: Verify Setup

```powershell
# Run setup script to verify everything
python setup_local.py
```

You should see:

```
✅ Python version OK
✅ All dependencies installed
✅ All directories created
✅ NLTK data downloaded
```

**⚠️ If you see errors:** Check the error messages and install missing packages.

---

## 📥 Phase 2: Dataset Download

### Step 2.1: Download Dataset Files

1. Go to: https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset

2. Download these files:

   - `shopping_queries_dataset_examples.parquet` (~150 MB)
   - `shopping_queries_dataset_products.parquet` (~20 MB)
   - `shopping_queries_dataset_sources.csv` (optional)

3. **IMPORTANT:** Place files in `data/raw/` directory:
   ```
   task2_semantic_search/
   └── data/
       └── raw/
           ├── shopping_queries_dataset_examples.parquet
           └── shopping_queries_dataset_products.parquet
   ```

### Step 2.2: Verify Dataset

```powershell
# Verify files are present and readable
python -c "import pandas as pd; df = pd.read_parquet('data/raw/shopping_queries_dataset_examples.parquet'); print(f'✅ Examples loaded: {len(df):,} rows')"
```

**Expected output:**

```
✅ Examples loaded: 2,621,738 rows
```

---

## 🎓 Phase 3: Training on Kaggle (CRITICAL)

This is where the model gets trained. You CANNOT run this locally due to memory constraints.

### Step 3.1: Kaggle Account Setup

1. Go to https://www.kaggle.com
2. Sign in or create account
3. Go to **Settings** → **API** → **Create New API Token**
4. Save `kaggle.json` (you might need it later)

### Step 3.2: Create New Kaggle Notebook

1. Click **Create** → **New Notebook**
2. **Settings** (right sidebar):
   - ✅ Enable **GPU** (P100 or T4)
   - ✅ Enable **Internet**
   - Session timeout: **12 hours**

### Step 3.3: Upload Dataset to Kaggle

**Option A: Upload Files Directly**

1. Click **Add Data** (right sidebar) → **Upload**
2. Upload both parquet files:
   - `shopping_queries_dataset_examples.parquet`
   - `shopping_queries_dataset_products.parquet`
3. Wait for upload (2-3 minutes)
4. Note the path: `/kaggle/input/your-dataset-name/`

**Option B: Use Existing Dataset**

1. Search for "Amazon ESCI Shopping Queries" in Kaggle Datasets
2. Click **Add Data** to notebook

### Step 3.4: Copy Training Code to Kaggle

1. Open the file `kaggle_training/task2_kaggle_training.ipynb` (File 4 I provided)
2. **IMPORTANT:** Copy the ENTIRE code content
3. Paste into Kaggle notebook
4. **UPDATE the dataset path** in Cell 3:
   ```python
   # Change this line to match your dataset path
   DATA_DIR = "/kaggle/input/your-dataset-name/"  # ← Update this!
   ```

### Step 3.5: Run Training

1. Click **Run All** button (or Ctrl+/)
2. Training will take approximately **3-4 hours**
3. You can:
   - Close the browser (training continues)
   - Monitor progress by refreshing
   - Check GPU usage in sidebar

**Training Progress:**

```
Cell 1-6: Setup and data loading (5 min)
Cell 7-9: Model initialization (2 min)
Cell 10: Training loop (3-4 hours) ← This is the long one
Cell 11-16: Evaluation and saving (10 min)
```

### Step 3.6: Monitor Training

Watch for these indicators:

- ✅ Epoch 1/5, 2/5, 3/5... (progressing)
- ✅ Training loss decreasing
- ✅ Validation loss decreasing
- ✅ "Best model saved" messages

**⚠️ If training fails:**

- Check GPU is enabled
- Reduce `BATCH_SIZE` to 16 in Cell 3
- Ensure dataset path is correct

### Step 3.7: Download Trained Files

**After training completes successfully**, download these files from Kaggle output:

1. Click **Output** tab (top right)
2. Download ALL these files:

   ```
   ✅ best_model.pth (~250 MB)
   ✅ product_embeddings.npy (~50 MB)
   ✅ product_metadata.parquet (~5 MB)
   ✅ train_history.json
   ✅ evaluation_results.json
   ✅ training_curves.png
   ✅ comprehensive_results.png
   ```

3. **IMPORTANT:** Place in local `models/` directory:
   ```
   task2_semantic_search/
   └── models/
       ├── best_model.pth
       ├── product_embeddings.npy
       ├── product_metadata.parquet
       ├── train_history.json
       ├── evaluation_results.json
       ├── training_curves.png
       └── comprehensive_results.png
   ```

**Verification:**

```powershell
# Check files are present
python setup_local.py
```

Should show:

```
✅ models/best_model.pth (250.45 MB)
✅ models/product_embeddings.npy (52.18 MB)
✅ models/product_metadata.parquet (4.87 MB)
```

---

## 🌐 Phase 4: Local Web Application

Now that you have the trained model, run the web application locally!

### Step 4.1: Verify All Files Present

```powershell
# Make sure you're in project directory
cd task2_semantic_search

# Verify setup
python setup_local.py
```

All checks should pass ✅

### Step 4.2: Launch Streamlit App

```powershell
# Activate virtual environment if not already
.\venv\Scripts\activate

# Run Streamlit app
streamlit run app.py
```

**Expected output:**

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Step 4.3: Open Browser

1. Browser should open automatically
2. If not, manually go to: `http://localhost:8501`
3. Wait for models to load (10-15 seconds first time)

You should see:

```
🔍 Semantic Product Search
Powered by Deep Learning & Transformer Models

✅ Models loaded successfully!
```

### Step 4.4: Test the Application

**Test Query 1:** "wireless bluetooth headphones"

Expected:

- Results appear in 1-2 seconds
- Top 10 products displayed
- Relevance scores shown (0.0-1.0)
- Products ranked by relevance

**Test Query 2:** "gaming laptop rtx"

**Test Query 3:** "running shoes women"

**Test Query 4:** "stainless steel water bottle"

### Step 4.5: Take Screenshots

**For your report**, take screenshots of:

1. Search interface (empty)
2. Search results for query "wireless headphones"
3. Different relevance scores
4. Search metrics (time, # results)

Save screenshots to `screenshots/` folder.

---

## 📊 Phase 5: Results Analysis

### Step 5.1: Review Training Results

Open `models/comprehensive_results.png` to see:

- Training/validation loss curves
- Metrics bar chart (NDCG, Precision, Recall, F1)
- NDCG across K values
- Prediction scatter plot

### Step 5.2: Review Evaluation Metrics

Open `models/evaluation_results.json`:

```powershell
# View metrics in terminal
python -c "import json; data = json.load(open('models/evaluation_results.json')); print(json.dumps(data, indent=2))"
```

Expected metrics (approximate):

```json
{
  "ndcg@5": 0.82,
  "ndcg@10": 0.84,
  "ndcg@20": 0.86,
  "precision@5": 0.68,
  "precision@10": 0.65,
  "precision@20": 0.6,
  "recall@5": 0.55,
  "recall@10": 0.7,
  "recall@20": 0.8,
  "f1@5": 0.61,
  "f1@10": 0.67,
  "f1@20": 0.68,
  "map": 0.75
}
```

### Step 5.3: Create Results Table

For your LaTeX report, format metrics as:

| Metric    | @5   | @10  | @20  |
| --------- | ---- | ---- | ---- |
| NDCG      | 0.82 | 0.84 | 0.86 |
| Precision | 0.68 | 0.65 | 0.60 |
| Recall    | 0.55 | 0.70 | 0.80 |
| F1        | 0.61 | 0.67 | 0.68 |

MAP: 0.75

---

## 📝 Phase 6: Documentation

### Step 6.1: LaTeX Report

**Template Structure:**

1. **Title Page**

   - Title: "Semantic Product Search and Ranking using Deep Learning"
   - Your name, roll number
   - Course: Generative AI (Fall 2025)
   - Date: November 2025

2. **Abstract** (150-200 words)

   - Problem statement
   - Approach summary
   - Key results

3. **Introduction** (1-2 pages)

   - Background on product search
   - Limitations of keyword-based search
   - Motivation for semantic search
   - Objectives

4. **Related Work** (1 page)

   - Traditional IR methods (TF-IDF, BM25)
   - Neural ranking models
   - BERT for semantic search
   - Amazon ESCI dataset

5. **Methodology** (3-4 pages)

   - Dataset description
   - Text preprocessing pipeline
   - Embedding methods (TF-IDF, BERT)
   - Model architecture (with diagram)
   - Training procedure
   - Hyperparameters

6. **Experiments** (1 page)

   - Experimental setup
   - Hardware (Kaggle P100 GPU)
   - Software (PyTorch, Transformers)
   - Train/val/test split

7. **Results** (2-3 pages)

   - Quantitative results (tables)
   - Training curves (figures)
   - Ablation studies
   - Error analysis

8. **Web Application** (1 page)

   - Architecture
   - User interface
   - Search pipeline
   - Screenshots

9. **Challenges and Solutions** (1 page)

   - Memory constraints → Solution
   - Training time → Solution
   - Inference speed → Solution

10. **Conclusion** (1 page)

    - Summary
    - Achievements
    - Future work

11. **References**
    - Amazon ESCI paper
    - BERT paper
    - Relevant papers

### Step 6.2: Use Overleaf

1. Go to https://www.overleaf.com
2. Create new project → **Springer LNCS Template**
3. Start writing report using above structure
4. Upload figures to Overleaf:
   - `training_curves.png`
   - `comprehensive_results.png`
   - Screenshots from web app

### Step 6.3: Code Documentation

Ensure all code has:

- ✅ Docstrings for all functions
- ✅ Type hints
- ✅ Inline comments for complex logic
- ✅ README.md is complete

### Step 6.4: Update prompts.txt

Fill in the `prompts.txt` template (File 12) with:

- All prompts you used
- Responses summary
- Your name and details

---

## 📦 Phase 7: Prepare Submission

### Step 7.1: Create Submission Package

Create ZIP file with this structure:

```
ROLLNO_NAME.ZIP
├── task2_main.ipynb              # Kaggle notebook (download from Kaggle)
├── report.pdf                    # LaTeX report exported as PDF
├── prompts.txt                   # All prompts used
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── model.py
│   ├── evaluate.py
│   └── utils.py
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── best_model.pth
│   ├── evaluation_results.json
│   └── training_curves.png
└── screenshots/
    ├── web_interface.png
    ├── search_results.png
    └── metrics.png
```

### Step 7.2: Final Checklist

Before submission, verify:

- ✅ Code runs without errors
- ✅ All evaluation metrics calculated
- ✅ Training curves generated
- ✅ Web interface functional
- ✅ LaTeX report complete (PDF exported)
- ✅ prompts.txt filled
- ✅ README.md accurate
- ✅ All files in correct locations
- ✅ ZIP file named correctly: `ROLLNO_NAME.ZIP`

### Step 7.3: Test Your Submission

**Important:** Test that everything works!

1. Extract ZIP to new folder
2. Follow README instructions
3. Verify code runs
4. Check PDF opens

---

## 🐛 Common Issues and Solutions

### Issue 1: "Module not found" error

**Solution:**

```powershell
# Reinstall requirements
pip install -r requirements.txt
```

### Issue 2: NLTK data not found

**Solution:**

```powershell
python -c "import nltk; nltk.download('all')"
```

### Issue 3: Streamlit won't start

**Solution:**

```powershell
# Check if port 8501 is in use
netstat -ano | findstr :8501

# Use different port
streamlit run app.py --server.port 8502
```

### Issue 4: Model loading takes too long

**Solution:**

- First load always takes longer (downloading tokenizers)
- Subsequent loads are cached
- Check internet connection

### Issue 5: Out of memory on Kaggle

**Solution:**

```python
# In Cell 3 of Kaggle notebook, change:
BATCH_SIZE = 16  # Instead of 32
GRADIENT_ACCUMULATION_STEPS = 2
```

### Issue 6: Training loss not decreasing

**Solution:**

- Check dataset loaded correctly
- Verify ESCI labels converted to scores
- Reduce learning rate to 1e-5

---

## ⏱️ Time Management

**Estimated Time Breakdown:**

| Phase            | Time      | When to Do          |
| ---------------- | --------- | ------------------- |
| Setup            | 1 hour    | Day 1 morning       |
| Dataset download | 30 min    | Day 1 afternoon     |
| Kaggle training  | 3-4 hours | Day 2-3 (automatic) |
| Local deployment | 1 hour    | Day 4               |
| Testing          | 30 min    | Day 4               |
| Report writing   | 3 hours   | Day 5-6             |
| Final checks     | 1 hour    | Day 7               |

**Total: 10-12 hours** spread over 7 days

---

## 📞 Getting Help

If you encounter issues:

1. **Check error messages carefully**
2. **Review README.md** for that component
3. **Check this guide** for similar issues
4. **Review code comments** in relevant file
5. **Google the specific error** (often helps!)

---

## ✅ Success Criteria

You've successfully completed when:

- ✅ Model trained on Kaggle with good metrics (NDCG@10 > 0.80)
- ✅ Web app runs locally without errors
- ✅ Search returns relevant results in <2 seconds
- ✅ All evaluation metrics calculated
- ✅ LaTeX report complete with all sections
- ✅ Code well-documented
- ✅ Submission ZIP created correctly

---

## 🎉 Final Notes

**Congratulations on completing Task 2!**

**Key Achievements:**

- Built a production-ready semantic search system
- Trained a state-of-the-art ranking model
- Implemented comprehensive evaluation
- Deployed a working web application

**Skills Gained:**

- Deep learning for NLP
- Transformer models (BERT)
- Information retrieval
- Model deployment
- Full-stack ML pipeline

**Good luck with your submission!** 🚀

---
