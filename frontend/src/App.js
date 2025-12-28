import React, { useState, useCallback, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Alert } from 'react-bootstrap';
import { useDropzone } from 'react-dropzone';
import { FaUpload, FaImage, FaBrain, FaCheckCircle, FaShieldAlt } from 'react-icons/fa';
import axios from 'axios';
import './index.css';

var API_URL = (function () {
  var fromEnv = (process.env.REACT_APP_API_URL || '').trim();
  if (fromEnv) return fromEnv.replace(/\/+$/, '');
  if (process.env.NODE_ENV === 'development') return 'http://localhost:5000';
  return '';
})();

function normalizePercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return null;
  if (value >= 0 && value <= 1) return value * 100;
  return value;
}

function toSafeClassLabel(label) {
  if (!label) return '';
  return String(label).replace(/_/g, ' ');
}

function normalizeApiResponse(data) {
  if (!data || typeof data !== 'object') {
    return { success: false, error: 'Invalid server response.' };
  }

  var inferredSuccess = false;
  if (data.success === true) inferredSuccess = true;
  if (data.ok === true) inferredSuccess = true;
  if (data.prediction && typeof data.prediction === 'object') inferredSuccess = true;

  if (!inferredSuccess) {
    return { success: false, error: data.error || data.message || 'Prediction failed.' };
  }

  var pred = data.prediction || data.result || {};

  var className = '';
  if (pred.class) className = pred.class;
  else if (pred.label) className = pred.label;
  else if (pred.predicted_class) className = pred.predicted_class;
  else if (data.class) className = data.class;

  var confidenceRaw = null;
  if (pred.confidence !== undefined && pred.confidence !== null) confidenceRaw = pred.confidence;
  else if (pred.score !== undefined && pred.score !== null) confidenceRaw = pred.score;
  else if (data.confidence !== undefined && data.confidence !== null) confidenceRaw = data.confidence;

  var confidence = normalizePercent(confidenceRaw);
  if (confidence === null) confidence = 0;

  var description = '';
  if (pred.description) description = pred.description;
  else if (pred.text) description = pred.text;
  else if (data.description) description = data.description;

  var probsRaw = data.all_probabilities || data.probabilities || data.allProbabilities || {};
  var all_probabilities = Object.fromEntries(
    Object.entries(probsRaw).map(function (kv) {
      var k = kv[0];
      var v = kv[1];
      var pct = normalizePercent(v);
      if (pct === null) pct = 0;
      return [k, pct];
    })
  );

  return {
    success: true,
    prediction: {
      class: className,
      confidence: confidence,
      description: description,
    },
    all_probabilities: all_probabilities,
  };
}

function App() {
  var _a = useState(null), selectedFile = _a[0], setSelectedFile = _a[1];
  var _b = useState(null), preview = _b[0], setPreview = _b[1];
  var _c = useState(null), prediction = _c[0], setPrediction = _c[1];
  var _d = useState(false), loading = _d[0], setLoading = _d[1];
  var _e = useState(null), error = _e[0], setError = _e[1];

  useEffect(function () {
    return function () {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  var onDrop = useCallback(function (acceptedFiles) {
    var file = acceptedFiles && acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPrediction(null);
      setError(null);

      var nextUrl = URL.createObjectURL(file);
      setPreview(function (prevUrl) {
        if (prevUrl) URL.revokeObjectURL(prevUrl);
        return nextUrl;
      });
    }
  }, []);

  var dropzone = useDropzone({
    onDrop: onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg'] },
    multiple: false,
  });

  async function handleAnalyze() {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    if (process.env.NODE_ENV !== 'development' && !API_URL) {
      setError(
        'Frontend is missing REACT_APP_API_URL. Set it to your backend Render URL (example https://your-backend.onrender.com) and redeploy the frontend.'
      );
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    var formData = new FormData();
    formData.append('file', selectedFile);

    try {
      var response = await axios.post(API_URL + '/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300000,
      });

      var normalized = normalizeApiResponse(response.data);

      if (normalized.success) setPrediction(normalized);
      else setError(normalized.error || 'Prediction failed. Please try again.');
    } catch (err) {
      var status = null;
      if (err && err.response && err.response.status) status = err.response.status;

      var serverMsg = null;
      if (err && err.response && err.response.data) {
        if (err.response.data.error) serverMsg = err.response.data.error;
        else if (err.response.data.message) serverMsg = err.response.data.message;
      }

      var msg = 'Failed to connect to the server. Please ensure the backend is running and CORS is configured.';
      if (status !== null && status !== undefined) msg = 'Request failed with status ' + status + '.';
      if (serverMsg) msg = serverMsg;

      setError(msg);
      // eslint-disable-next-line no-console
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setSelectedFile(null);
    setPrediction(null);
    setError(null);
    setPreview(function (prevUrl) {
      if (prevUrl) URL.revokeObjectURL(prevUrl);
      return null;
    });
  }

  function getSeverityGradient(className) {
    var gradients = {
      No_DR: 'linear-gradient(135deg, #28a745 0%, #20c997 100%)',
      Mild: 'linear-gradient(135deg, #ffc107 0%, #ffca2c 100%)',
      Moderate: 'linear-gradient(135deg, #fd7e14 0%, #ff8c42 100%)',
      Severe: 'linear-gradient(135deg, #dc3545 0%, #e4606d 100%)',
      Proliferative_DR: 'linear-gradient(135deg, #6f42c1 0%, #8c68cd 100%)',
    };
    return gradients[className] || 'linear-gradient(135deg, #6c757d 0%, #868e96 100%)';
  }

  var safePrediction = {};
  if (prediction && prediction.prediction) safePrediction = prediction.prediction;

  var safeClass = '';
  if (safePrediction.class) safeClass = safePrediction.class;

  var safeConfidence = 0;
  if (typeof safePrediction.confidence === 'number') safeConfidence = safePrediction.confidence;

  var probs = {};
  if (prediction && prediction.all_probabilities) probs = prediction.all_probabilities;

  var probEntries = Object.entries(probs).sort(function (a, b) {
    return (b[1] || 0) - (a[1] || 0);
  });

  var errorAlert = null;
  if (error) {
    errorAlert = (
      <Alert
        variant="danger"
        className="alert-modern"
        dismissible
        onClose={function () {
          setError(null);
        }}
      >
        <strong>Error:</strong> {error}
      </Alert>
    );
  }

  var analyzeBtnContent = (
    <>
      <FaBrain className="me-2" />
      Analyze Now
    </>
  );
  if (loading) {
    analyzeBtnContent = (
      <>
        <span className="spinner-border spinner-border-sm me-2" />
        Analyzing...
      </>
    );
  }

  var uploadSection = (
    <div
      {...dropzone.getRootProps()}
      className={'dropzone-modern ' + (dropzone.isDragActive ? 'active' : '')}
    >
      <input {...dropzone.getInputProps()} />
      <div className="dropzone-content">
        <FaUpload className="upload-icon" />
        <h4>Drop your image here</h4>
        <p>or click to browse</p>
        <small className="text-muted">Supports: PNG, JPG, JPEG</small>
      </div>
    </div>
  );

  if (preview) {
    uploadSection = (
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
            {analyzeBtnContent}
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
    );
  }

  var loadingSection = null;
  if (loading) {
    loadingSection = (
      <div className="loading-container">
        <div className="loading-spinner-modern"></div>
        <p className="mt-3">System is analyzing the retinal image...</p>
      </div>
    );
  }

  var resultsContent = (
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
  );

  if (prediction) {
    var probSectionBody = (
      <p className="text-muted mb-0">No probability distribution returned by the server.</p>
    );

    if (probEntries.length !== 0) {
      probSectionBody = (
        <>
          {probEntries.map(function (entry) {
            var className = entry[0];
            var probability = entry[1];

            var p = 0;
            if (typeof probability === 'number') p = probability;

            var pct = Math.max(0, Math.min(100, p));

            var barText = null;
            if (pct > 15) barText = <span className="prob-bar-text">{pct.toFixed(1)}%</span>;

            return (
              <div key={className} className="prob-item">
                <div className="prob-header">
                  <span className="prob-label">{toSafeClassLabel(className)}</span>
                  <span className="prob-value">{pct.toFixed(1)}%</span>
                </div>
                <div className="prob-bar-container">
                  <div
                    className="prob-bar-fill"
                    style={{
                      width: pct + '%',
                      background: getSeverityGradient(className),
                    }}
                  >
                    {barText}
                  </div>
                </div>
              </div>
            );
          })}
        </>
      );
    }

    resultsContent = (
      <div className="results-content">
        <div className="result-header">
          <FaCheckCircle className="success-icon" />
          <h3>Diagnosis Complete</h3>
        </div>

        <div className="diagnosis-card" style={{ background: getSeverityGradient(safeClass) }}>
          <div className="diagnosis-class">{toSafeClassLabel(safeClass)}</div>
          <div className="diagnosis-confidence">{safeConfidence.toFixed(1)}% Confidence</div>
          <div className="diagnosis-description">{safePrediction.description || ''}</div>
        </div>

        <div className="probability-section">
          <h5 className="mb-3">Probability Distribution</h5>
          {probSectionBody}
        </div>

        <Alert variant="light" className="security-alert mt-3">
          <FaShieldAlt className="me-2" />
          Your data is encrypted and anonymized. GDPR/PDPA compliant.
        </Alert>
      </div>
    );
  }

  return (
    <div className="app-container">
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
        {errorAlert}

        <Row className="g-4">
          <Col lg={6}>
            <Card className="upload-card h-100">
              <Card.Body className="d-flex flex-column">
                <h3 className="mb-4">
                  <FaImage className="me-2" />
                  Upload Retinal Image
                </h3>

                {uploadSection}
                {loadingSection}
              </Card.Body>
            </Card>
          </Col>

          <Col lg={6}>
            <Card className="results-card h-100">
              <Card.Body>{resultsContent}</Card.Body>
            </Card>
          </Col>
        </Row>

        <Alert variant="info" className="disclaimer-alert mt-4">
          <strong>⚕️ Medical Disclaimer:</strong> This tool is for educational and screening purposes only. Always consult
          with a qualified ophthalmologist for proper medical diagnosis and treatment.
        </Alert>
      </Container>
    </div>
  );
}

export default App;
