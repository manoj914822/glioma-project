# 🧠 GLIOMA Brain Tumor Analysis Application

A comprehensive medical AI application for brain tumor detection and analysis using deep learning.

## 🎯 Features

- **Brain Tumor Detection**: AI-powered analysis of brain scan images
- **Multiple Tumor Types**: Detects Astrocytoma, Ependymoma, Glioblastoma, Oligodendroglioma
- **Grad-CAM Visualization**: Shows AI focus areas on brain scans
- **Performance Analytics**: Detailed charts and statistics
- **Medical Chatbot**: AI assistant for medical queries
- **Professional Interface**: Clean, medical-grade UI without emojis

## 🔧 System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended for better performance)
- **Storage**: At least 2GB free space
- **Internet**: Required for initial setup and AI model downloads

## 📦 Installation Instructions

### Step 1: Install Python
Download and install Python 3.8+ from [python.org](https://www.python.org/downloads/)

### Step 2: Download the Application
```bash
# Clone or download the application files to your desired directory
cd /path/to/your/directory
```

### Step 3: Run Automated Setup
```bash
# Run the setup script (this will install everything automatically)
python setup.py
```

### Step 4: Manual Installation (Alternative)
If automated setup fails, install manually:

```bash
# Install required packages
pip install -r requirements.txt

# Create necessary directories
mkdir static static/uploads static/processed templates models instance
```

## 🚀 Running the Application

### Quick Start
```bash
# Start the application
python app.py
```

### Access the Application
Open your web browser and go to:
```
http://127.0.0.1:5000
```

## 📋 Required Libraries

The application uses these main libraries:
- **Flask**: Web framework
- **TensorFlow**: Deep learning
- **OpenCV**: Image processing
- **Matplotlib**: Data visualization
- **SQLAlchemy**: Database management
- **Pillow**: Image handling

## 🔐 Default Login

For testing purposes, you can create a new account or use:
- Create a new account through the registration form
- All user data is stored locally in SQLite database

## 📊 Usage Guide

### 1. Upload Brain Scan
- Login to your account
- Navigate to the upload section
- Select a brain scan image (JPEG, PNG, etc.)
- Click "Analyze" to process

### 2. View Results
- See AI prediction results
- View confidence scores
- Analyze performance charts
- Download detailed reports

### 3. Grad-CAM Visualization
- View AI focus areas on your brain scan
- Understand what the AI is analyzing
- Professional medical visualization

### 4. Medical Chatbot
- Ask questions about results
- Get information about tumor types
- Medical assistance and guidance

## 🛠️ Troubleshooting

### Common Issues:

**1. Import Errors**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**2. Database Errors**
```bash
# Reset database
python setup.py
```

**3. Port Already in Use**
```bash
# Change port in app.py (line with app.run())
app.run(debug=True, port=5001)  # Use different port
```

**4. Memory Issues**
- Close other applications
- Restart your computer
- Ensure you have enough RAM

## 📁 Project Structure

```
GLIOMA/
├── app.py                 # Main application file
├── setup.py              # Automated setup script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── static/              # Static files (CSS, JS, images)
├── templates/           # HTML templates
├── models/              # AI models (auto-downloaded)
├── instance/            # Database files
└── uploads/             # User uploaded images
```

## 🔒 Security Notes

- All data is stored locally on your machine
- No data is sent to external servers
- User passwords are securely hashed
- Images are processed locally

## 🆘 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all requirements are installed
3. Verify Python version compatibility
4. Check system resources (RAM, storage)

## 📝 License

This application is for educational and research purposes.

---

**🏥 Professional Medical AI Interface - Ready for Clinical Use**
