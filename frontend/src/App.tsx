import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import './App.css';

interface PredictionResult {
    prediction: string;
    confidence: number;
}

function App() {
    const [prediction, setPrediction] = useState<PredictionResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    const onDrop = useCallback(async (acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (!file) return;

        // Create preview URL
        setPreviewUrl(URL.createObjectURL(file));
        setIsLoading(true);
        setError(null);
        setPrediction(null);

        // Create form data
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const result = await response.json();
            setPrediction(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setIsLoading(false);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.jpeg', '.jpg', '.png']
        },
        multiple: false
    });

    return (
        <div className="App">
            <h1>Cat or Dog Classifier</h1>

            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
                <input {...getInputProps()} />
                {isDragActive ? (
                    <p>Drop the image here...</p>
                ) : (
                    <p>Drag and drop an image here, or click to select a file</p>
                )}
            </div>

            {previewUrl && (
                <div className="preview">
                    <img src={previewUrl} alt="Preview" />
                </div>
            )}

            {isLoading && <div className="loading">Analyzing image...</div>}

            {error && <div className="error">{error}</div>}

            {prediction && (
                <div className="result">
                    <h2>Result: {prediction.prediction}</h2>
                    <p>Confidence: {(prediction.confidence * 100).toFixed(2)}%</p>
                </div>
            )}
        </div>
    );
}

export default App; 