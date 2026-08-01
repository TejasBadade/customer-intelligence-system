# 🚖 Ride Booking Data Analysis

An exploratory data analysis (EDA) project on a large-scale ride-booking dataset with **150,000 records** and **21 features**, built using Python.

---

## 📁 Project Structure

```
├── Final_Project.ipynb        # Main analysis notebook
├── Dataset.csv                # Raw dataset
├── final_Dataset_cleaned.csv  # Cleaned dataset (output)
├── requirements.txt           # Required libraries
└── README.md                  # Project documentation
```

---

## 📊 Dataset Overview

| Property | Details |
|----------|---------|
| Rows | 1,50,000 |
| Columns | 21 |
| Time Period | 365 days (2024) |
| Key Columns | Booking Status, Vehicle Type, Pickup/Drop Location, Ride Distance, Booking Value, Driver Ratings, Customer Rating, Payment Method |

---

## 🔍 Analysis Performed

### 1️⃣ Data Cleaning & Preprocessing
- Removed hidden spaces from column names
- Converted `Date` and `Time` columns to proper datetime formats
- Handled missing values (dropped nulls in critical columns, filled numeric columns with median)
- Standardized categorical variables (strip + title case)

### 2️⃣ Descriptive Analysis
- Most and least popular vehicle types
- Average ride distance and booking value
- Distribution of driver and customer ratings
- Most common cancellation reasons (by customer and driver)

### 3️⃣ Customer Behavior Insights
- Frequent cancellers by Customer ID
- Cancellation patterns by time of day and day of week
- Correlation between ride value, distance, and customer satisfaction

### 4️⃣ Driver Performance Evaluation
- Highest, lowest, and average driver ratings
- Driver cancellation count and reasons

### 5️⃣ Operational Metrics
- Average VTAT and CTAT across vehicle types
- Peak demand pickup and drop locations
- Peak demand time slots

---

## ▶️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook Final_Project.ipynb
   ```

4. Make sure `Dataset.csv` is in the **same folder** as the notebook before running.

---

## 🛠️ Libraries Used

- `pandas` — data manipulation
- `numpy` — numerical operations
- `matplotlib` — data visualization
- `seaborn` — statistical plots

---

## 👤 Author

**Tejas**  
Aspiring Data Analyst | Python & Data Science Enthusiast
