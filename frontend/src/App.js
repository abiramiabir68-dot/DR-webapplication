import React, { useState, useCallback } from 'react';
import { Container, Row, Col, Card, Button, Alert } from 'react-bootstrap';
import { useDropzone } from 'react-dropzone';
import { FaUpload, FaImage, FaBrain, FaCheckCircle, FaShieldAlt } from 'react-icons/fa';
import axios from 'axios';
import './index.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    multiple: false
  });

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/api/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setPrediction(response.data);
      } else {
        setError('Prediction failed. Please try again.');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to connect to the server. Please ensure the backend is running.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
  };

  const getSeverityColor = (className) => {
    const colors = {
      'No_DR': '#28a745',
      'Mild': '#ffc107',
      'Moderate': '#fd7e14',
      'Severe': '#dc3545',
      'Proliferative_DR': '#6f42c1'
    };
    return colors[className] || '#6c757d';
  };

  const getSeverityGradient = (className) => {
    const gradients = {
      'No_DR': 'linear-gradient(135deg, #28a745 0%, #20c997 100%)',
      'Mild': 'linear-gradient(135deg, #ffc107 0%, #ffca2c 100%)',
      'Moderate': 'linear-gradient(135deg, #fd7e14 0%, #ff8c42 100%)',
      'Severe': 'linear-gradient(135deg, #dc3545 0%, #e4606d 100%)',
      'Proliferative_DR': 'linear-gradient(135deg, #6f42c1 0%, #8c68cd 100%)'
    };
    return gradients[className] || 'linear-gradient(135deg, #6c757d 0%, #868e96 100%)';
  };

  return (
    <div className="app-container">
      {/* Header */}
      <div className="app-header-top">
        <Container>
          <Row className="align-items-center">
            <Col>
              <h1 className="mb-0">🩺 Diabetic Retinopathy Prediction and Classification Web Application System</h1>
              <p className="mb-0 text-light">Powered by RA EfficientNetB3 Deep Learning</p>
            </Col>
            <Col xs="auto">
              <div className="security-badge">
                <FaShieldAlt className="me-2" />
                GDPR/PDPA Compliant
              </div>
            </Col>
          </Row>
        </Container>
      </div>

      <Container className="mt-4">
        {error && (
          <Alert variant="danger" className="alert-modern" dismissible onClose={() => setError(null)}>
            <strong>Error:</strong> {error}
          </Alert>
        )}

        <Row className="g-4">
          {/* Left Side - Upload Section */}
          <Col lg={6}>
            <Card className="upload-card h-100">
              <Card.Body className="d-flex flex-column">
                <h3 className="mb-4">
                  <FaImage className="me-2" />
                  Upload Retinal Image
                </h3>

                {!preview ? (
                  <div {...getRootProps()} className={`dropzone-modern ${isDragActive ? 'active' : ''}`}>
                    <input {...getInputProps()} />
                    <div className="dropzone-content">
                      <FaUpload className="upload-icon" />
                      <h4>Drop your image here</h4>
                      <p>or click to browse</p>
                      <small className="text-muted">Supports: PNG, JPG, JPEG</small>
                    </div>
                  </div>
                ) : (
                  <div className="preview-container">
                    <img src={preview} alt="Preview" className="preview-image-modern" />
                    
                    <div className="button-group mt-4">
                      <Button
                        variant="primary"
                        size="lg"
                        className="btn-modern btn-analyze"
                        onClick={handleAnalyze}
                        disabled={loading}
                      >
                        {loading ? (
                          <>
                            <span className="spinner-border spinner-border-sm me-2" />
                            Analyzing...
                          </>
                        ) : (
                          <>
                            <FaBrain className="me-2" />
                            Analyze Now
                          </>
                        )}
                      </Button>
                      <Button
                        variant="outline-light"
                        size="lg"
                        className="btn-modern"
                        onClick={handleClear}
                        disabled={loading}
                      >
                        Upload New
                      </Button>
                    </div>
                  </div>
                )}

                {loading && (
                  <div className="loading-container">
                    <div className="loading-spinner-modern"></div>
                    <p className="mt-3">System is analyzing the retinal image...</p>
                  </div>
                )}
              </Card.Body>
            </Card>
          </Col>

          {/* Right Side - Results Section */}
          <Col lg={6}>
            <Card className="results-card h-100">
              <Card.Body>
                {!prediction ? (
                  <div className="no-results">
                    <div className="info-icon">
                      <FaBrain />
                    </div>
                    <h3>Results</h3>
                    <p className="text-muted">Upload an image and click "Analyze Now" to see the results here</p>
                    
                    <div className="features-grid mt-4">
                      <div className="feature-item">
                        <div className="feature-number">81%</div>
                        <div className="feature-label">Accuracy</div>
                      </div>
                      <div className="feature-item">
                        <div className="feature-number">&lt;2s</div>
                        <div className="feature-label">Analysis Time</div>
                      </div>
                      <div className="feature-item">
                        <div className="feature-number">5</div>
                        <div className="feature-label">DR Classes</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="results-content">
                    <div className="result-header">
                      <FaCheckCircle className="success-icon" />
                      <h3>Diagnosis Complete</h3>
                    </div>

                    {/* Main Diagnosis */}
                    <div 
                      className="diagnosis-card"
                      style={{ background: getSeverityGradient(prediction.prediction.class) }}
                    >
                      <div className="diagnosis-class">
                        {prediction.prediction.class.replace('_', ' ')}
                      </div>
                      <div className="diagnosis-confidence">
                        {prediction.prediction.confidence.toFixed(1)}% Confidence
                      </div>
                      <div className="diagnosis-description">
                        {prediction.prediction.description}
                      </div>
                    </div>

                    {/* Probability Chart */}
                    <div className="probability-section">
                      <h5 className="mb-3">Probability Distribution</h5>
                      {Object.entries(prediction.all_probabilities)
                        .sort((a, b) => b[1] - a[1])
                        .map(([className, probability]) => (
                          <div key={className} className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label">{className.replace('_', ' ')}</span>
                              <span className="prob-value">{probability.toFixed(1)}%</span>
                            </div>
                            <div className="prob-bar-container">
                              <div 
                                className="prob-bar-fill"
                                style={{
                                  width: `${probability}%`,
                                  background: getSeverityGradient(className)
                                }}
                              >
                                {probability > 15 && <span className="prob-bar-text">{probability.toFixed(1)}%</span>}
                              </div>
                            </div>
                          </div>
                        ))}
                    </div>

                    {/* Security Info */}
                    <Alert variant="light" className="security-alert mt-3">
                      <FaShieldAlt className="me-2" />
                      Your data is encrypted and anonymized. GDPR/PDPA compliant.
                    </Alert>
                  </div>
                )}
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* Info Footer */}
        <Alert variant="info" className="disclaimer-alert mt-4">
          <strong>⚕️ Medical Disclaimer:</strong> This tool is for educational and screening purposes only. 
          Always consult with a qualified ophthalmologist for proper medical diagnosis and treatment.
        </Alert>
      </Container>
    </div>
  );
}

export default App;
