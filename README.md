# Cat vs Dog Image Classifier

This is a web application that uses a TensorFlow model to classify images as either cats or dogs. The application consists of a React frontend for image upload and a Flask backend for image processing and classification.

## Project Structure

```
.
├── frontend/          # React frontend application
├── backend/           # Flask backend application
└── cat_dog_model/     # Directory for the TensorFlow model
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure to place your TensorFlow model in the `cat_dog_model` directory.

5. Start the Flask server:
   ```bash
   python app.py
   ```

The backend server will run on `http://localhost:5000`.

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install the required npm packages:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```

The frontend application will run on `http://localhost:3000`.

## Usage

1. Open your web browser and go to `http://localhost:3000`
2. Drag and drop an image file onto the upload area, or click to select a file
3. The application will display the uploaded image and show the classification result
4. The result will include both the prediction (Cat or Dog) and the confidence level

## Requirements

- Python 3.7+
- Node.js 14+
- TensorFlow model trained for cat/dog classification 