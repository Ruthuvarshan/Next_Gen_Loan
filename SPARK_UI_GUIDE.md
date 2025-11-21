# 🚀 Spark UI Setup & Usage Guide

## Overview

This guide shows you how to run PySpark jobs and monitor them using the **Spark UI** (web interface at http://localhost:4040).

---

## ✅ Prerequisites Verified

- ✅ **Spark Installed**: `R:\spark`
- ✅ **Python Environment**: `.venv` with PySpark installed
- ✅ **Spark UI Enabled**: Configured in `src/utils/spark_config.py`

---

## 🎯 Quick Start - See Spark UI in Action

### Step 1: Activate Virtual Environment

```cmd
cd r:\SSF\Next_Gen_Loan
.venv\Scripts\activate
```

### Step 2: Generate Sample Data

```cmd
python scripts\generate_sample_data.py
```

**Output**: Creates `data/sample/loan_data_sample.csv` with 10,000 loan records

### Step 3: Run Spark UI Demo

```cmd
python scripts\spark_ui_demo.py
```

### Step 4: Open Spark UI in Browser

Once the Spark session starts, open your browser:

```
http://localhost:4040
```

**You'll see**:
- 📊 **Jobs Tab**: All Spark jobs (data loading, transformations, training)
- 🎯 **Stages Tab**: Detailed task execution and timing
- 💾 **Storage Tab**: Cached DataFrames and RDDs
- 🌍 **Environment Tab**: All Spark configurations
- ⚡ **Executors Tab**: CPU and memory usage
- 📈 **SQL Tab**: DataFrame query plans and optimizations

---

## 📊 What You'll See in Spark UI

### Jobs Tab
- Lists all Spark jobs executed
- Shows job duration, stages, and tasks
- Green = completed, Blue = running, Red = failed

### Stages Tab
- Shows each stage of computation
- Task metrics: duration, shuffle read/write, memory usage
- DAG visualization of job execution plan

### Storage Tab
- Shows cached/persisted RDDs and DataFrames
- Memory and disk usage
- Number of partitions

### SQL Tab (most important for ML)
- Shows DataFrame transformations
- Physical and logical query plans
- Optimization details from Catalyst optimizer

### Environment Tab
- All Spark configuration properties
- JVM settings
- System properties

---

## 🎓 Understanding the Demo Script

The `spark_ui_demo.py` script runs through these stages:

1. **Session Creation** → Creates Spark UI server
2. **Data Loading** → Reads CSV (visible as a job)
3. **Data Preprocessing** → Transformations and feature engineering
4. **Pipeline Building** → ML pipeline construction
5. **Model Training** → GBT training (multiple jobs)
6. **Evaluation** → Predictions and metrics
7. **Feature Importance** → Analyze model
8. **Save Model** → Persist to disk

Each step generates Spark jobs that you can monitor in real-time!

---

## 🔧 Advanced: Training Full Model with Spark UI

### Option 1: Use the Full Training Script

```cmd
python scripts\train_spark_model.py --data-path data\sample\loan_data_sample.csv --output-dir models\ --with-nlp
```

### Option 2: Integrate with API

The API automatically uses Spark models if they exist:

```cmd
# Train model
python scripts\train_spark_model.py --data-path data\sample\loan_data_sample.csv --output-dir models\

# Start API (will use Spark model)
.venv\Scripts\uvicorn.exe src.api.main:app --reload --port 8000
```

---

## 📈 Spark UI Features Explained

### 1. DAG Visualization
- Shows execution plan as a directed acyclic graph
- Visualizes data flow through transformations
- Identifies bottlenecks and skipped stages

### 2. Event Timeline
- Timeline view of when jobs/stages started and completed
- Helps identify resource contention
- Shows parallel vs sequential execution

### 3. Task Metrics
- **Duration**: Time taken per task
- **Shuffle Read/Write**: Data movement between stages
- **GC Time**: Garbage collection overhead
- **Memory Spill**: When data doesn't fit in memory

### 4. SQL Query Plans
- **Logical Plan**: What you asked Spark to do
- **Optimized Plan**: After Catalyst optimization
- **Physical Plan**: Actual execution strategy

---

## 🔍 Monitoring Different Job Types

### Data Loading Jobs
```python
df = spark.read.csv("data.csv")
```
- Generates 1 job with 1 stage
- Check: Number of partitions, data size

### Transformation Jobs (Lazy)
```python
df.withColumn("new_col", F.col("old_col") * 2)
```
- Doesn't execute immediately
- Added to execution plan

### Action Jobs (Trigger Execution)
```python
df.count()  # Triggers execution
df.show()   # Triggers execution
df.write.csv("output")  # Triggers execution
```
- Executes all pending transformations
- Creates jobs in Spark UI

### ML Training Jobs
```python
model = pipeline.fit(train_df)
```
- Multiple jobs for feature engineering
- Iterative jobs for tree building
- Checkpointing and materialization jobs

---

## 🎯 Key Spark UI Tabs for ML Workloads

### For Data Scientists:

1. **SQL Tab** (Most Important)
   - See how your DataFrame operations are optimized
   - Identify expensive operations
   - Check for data skew

2. **Jobs Tab**
   - Overall progress of training
   - Identify long-running jobs

3. **Stages Tab**
   - Task-level details
   - Identify stragglers (slow tasks)

4. **Storage Tab**
   - Check if data is cached properly
   - Memory usage of cached DataFrames

---

## 🛠️ Customizing Spark UI

### Change UI Port (if 4040 is busy)

Edit `src/utils/spark_config.py`:

```python
conf.set("spark.ui.port", "4041")  # Change to any available port
```

### Keep More Job History

```python
conf.set("spark.ui.retainedJobs", "2000")      # Keep more jobs
conf.set("spark.ui.retainedStages", "2000")    # Keep more stages
conf.set("spark.sql.ui.retainedExecutions", "2000")  # Keep more SQL queries
```

### Enable History Server (View Past Jobs)

The demo already saves event logs. To view them after session ends:

1. Start Spark History Server:
```cmd
R:\spark\bin\spark-class.cmd org.apache.spark.deploy.history.HistoryServer
```

2. Access at: `http://localhost:18080`

3. Event logs are in: `spark-warehouse/spark-events/`

---

## 📝 Tips for Effective Monitoring

### 1. Use Cache Wisely
```python
df.cache()  # Cache data you'll reuse
df.count()  # Materialize the cache
```
- Check Storage tab to verify caching
- Uncache when done: `df.unpersist()`

### 2. Monitor Task Skew
- In Stages tab, check task durations
- If some tasks take much longer → data skew
- Fix: Repartition data or use salting

### 3. Watch Shuffle Operations
- Expensive operations: `groupBy`, `join`, `repartition`
- Check Shuffle Read/Write in Stage metrics
- Minimize shuffles when possible

### 4. Track Memory Usage
- Check Executors tab for memory usage
- Look for "Memory Spill to Disk" warnings
- Increase executor memory if needed

---

## 🚨 Troubleshooting

### Spark UI Not Opening?

**Check if port 4040 is available:**
```cmd
netstat -ano | findstr :4040
```

**If busy, use different port:**
```python
conf.set("spark.ui.port", "4041")
```

### "Address already in use" Error

Another Spark session is running. Either:
1. Stop the existing session
2. Use a different port
3. Reuse existing session: `SparkSession.builder.getOrCreate()`

### UI Shows No Jobs

- Jobs only appear when actions are executed
- Transformations are lazy (won't show until action)
- Run `df.count()` or `df.show()` to trigger

### Event Logs Not Saving

Check directory exists and is writable:
```cmd
dir spark-warehouse\spark-events
```

---

## 🎓 Learning Resources

### Spark UI Tabs to Explore:

1. **Start with Jobs Tab** - Get overall picture
2. **Dive into Stages Tab** - Understand task execution
3. **Check SQL Tab** - See query optimizations
4. **Monitor Storage Tab** - Verify caching
5. **Review Executors Tab** - Check resource usage

### Useful Metrics to Track:

- **Job Duration**: How long jobs take
- **Stage Duration**: Which stages are slow
- **Shuffle Size**: Data movement overhead
- **Task GC Time**: Garbage collection impact
- **Cached Data Size**: Memory efficiency

---

## 🎯 Next Steps

### For Development:
1. ✅ Run `spark_ui_demo.py` to see basic workflow
2. ✅ Experiment with caching and see Storage tab
3. ✅ Try different transformations and see SQL plans

### For Production:
1. Train full model with `train_spark_model.py`
2. Set up History Server for persistent monitoring
3. Optimize based on UI insights
4. Deploy to cluster (YARN/K8s) with UI on master node

---

## 📚 Summary

| Action | Command | Spark UI URL |
|--------|---------|--------------|
| **Generate Data** | `python scripts\generate_sample_data.py` | N/A |
| **Run Demo** | `python scripts\spark_ui_demo.py` | http://localhost:4040 |
| **Train Model** | `python scripts\train_spark_model.py --data-path data\sample\loan_data_sample.csv --output-dir models\` | http://localhost:4040 |
| **History Server** | `R:\spark\bin\spark-class.cmd org.apache.spark.deploy.history.HistoryServer` | http://localhost:18080 |

---

## ✅ Checklist

- [ ] Activate virtual environment
- [ ] Generate sample data
- [ ] Run Spark UI demo
- [ ] Open http://localhost:4040 in browser
- [ ] Explore Jobs, Stages, SQL, Storage tabs
- [ ] Train a model and monitor it
- [ ] (Optional) Set up History Server for persistent logs

---

**Happy Spark Monitoring! 🚀**
