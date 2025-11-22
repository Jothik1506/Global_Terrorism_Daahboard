# 🚀 Streamlit Cloud Deployment Guide

## Prerequisites
1. A GitHub account
2. Your dashboard code (`FinalDashboard.py`)
3. A `requirements.txt` file with dependencies

---

## Step 1: Prepare Your Files

### 1.1 Create/Update `requirements.txt`
Make sure you have a `requirements.txt` file in your project root with:
```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.17.0
openpyxl>=3.1.0
```

### 1.2 Update File Paths in `FinalDashboard.py`
Before deploying, update the hardcoded paths to be relative or remove them:

**Find these lines (around line 39-40):**
```python
LOCAL_FILE = r"D:\VS CODE\Python\dashboard_data.csv"
SIDEBAR_LOGO = r"D:\VS CODE\Python\WhatsApp Image 2025-11-11 at 7.19.27 PM.jpeg"
```

**Change to:**
```python
# For deployment, these will be optional - users can upload files
LOCAL_FILE = None  # or "dashboard_data.csv" if you include it in the repo
SIDEBAR_LOGO = None  # or "logo.jpeg" if you include it in the repo
```

**Update the sidebar logo section (around line 102-105):**
```python
# Sidebar setup
if SIDEBAR_LOGO and os.path.exists(SIDEBAR_LOGO):
    st.sidebar.image(SIDEBAR_LOGO, use_container_width=True)
```

**Update the local file loading (around line 120-125):**
```python
else:
    # Try local file
    if LOCAL_FILE and os.path.exists(LOCAL_FILE):
        with st.spinner("Loading local data..."):
            df, source = load_data_from_file(file_path=LOCAL_FILE)
    else:
        st.warning("⚠️ No local file found. Please upload a dataset using the sidebar uploader.")
        st.info("💡 Upload a CSV or XLSX file to get started.")
        st.stop()
```

---

## Step 2: Push to GitHub

### 2.1 Initialize Git Repository (if not already done)
```bash
cd "D:\VS CODE\Python"
git init
git add FinalDashboard.py requirements.txt
git commit -m "Initial dashboard commit"
```

### 2.2 Create GitHub Repository
1. Go to [GitHub.com](https://github.com)
2. Click **"New repository"**
3. Name it (e.g., `terrorism-dashboard`)
4. **Don't** initialize with README (if you already have files)
5. Click **"Create repository"**

### 2.3 Push Your Code
```bash
git remote add origin https://github.com/YOUR_USERNAME/terrorism-dashboard.git
git branch -M main
git push -u origin main
```

---

## Step 3: Deploy to Streamlit Cloud

### 3.1 Sign Up / Sign In
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your **GitHub account**

### 3.2 Deploy Your App
1. Click **"New app"**
2. Select your repository: `YOUR_USERNAME/terrorism-dashboard`
3. Select branch: `main`
4. Main file path: `FinalDashboard.py`
5. Click **"Deploy"**

### 3.3 Wait for Deployment
- Streamlit will install dependencies and run your app
- This usually takes 1-2 minutes
- You'll see a live URL once it's ready (e.g., `https://your-app.streamlit.app`)

---

## Step 4: Post-Deployment

### 4.1 Test Your App
- Visit your deployed URL
- Upload a dataset using the sidebar uploader
- Verify all features work correctly

### 4.2 Update App Settings (Optional)
- Go to your app's settings in Streamlit Cloud
- You can:
  - Change the app URL
  - Set environment variables
  - Configure secrets (for API keys, etc.)

---

## Troubleshooting

### Issue: "Module not found"
**Solution:** Check your `requirements.txt` includes all dependencies.

### Issue: "File not found" errors
**Solution:** Make sure you've updated hardcoded paths to be relative or optional.

### Issue: App is slow
**Solution:** 
- Use the sampling feature for large datasets
- Consider optimizing data loading

### Issue: Deployment fails
**Solution:**
- Check the deployment logs in Streamlit Cloud
- Ensure `FinalDashboard.py` is in the root directory
- Verify `requirements.txt` syntax is correct

---

## Quick Reference Commands

```bash
# Check your files are ready
ls FinalDashboard.py requirements.txt

# Test locally before deploying
streamlit run FinalDashboard.py

# Commit changes
git add .
git commit -m "Update dashboard"
git push
```

---

## Notes
- **Free tier:** Streamlit Cloud offers free hosting with some limitations
- **Auto-updates:** Every push to your main branch automatically redeploys
- **File uploads:** Users can upload datasets directly in the deployed app
- **Data privacy:** Uploaded files are session-based and not stored permanently

---

## Need Help?
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Forum](https://discuss.streamlit.io/)

