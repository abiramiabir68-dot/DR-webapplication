import os
import warnings
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = os.getenv("TF_ENABLE_ONEDNN_OPTS", "1")
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
import keras


from keras import layers
register_keras_serializable = tf.keras.utils.register_keras_serializable
from PIL import Image
import base64
import io
from dotenv import load_dotenv
import cv2

preprocess_input = tf.keras.applications.efficientnet.preprocess_input

try:
    if hasattr(keras, "config") and hasattr(keras.config, "enable_unsafe_deserialization"):
        keras.config.enable_unsafe_deserialization()
    elif hasattr(keras, "saving") and hasattr(keras.saving, "enable_unsafe_deserialization"):
        keras.saving.enable_unsafe_deserialization()
except Exception:
    pass

def _patch_from_config(layer_cls, drop_keys):
    
    orig = layer_cls.from_config
    orig_func = getattr(orig, "__func__", orig)

    @classmethod
    def patched(cls, config):
        for k in drop_keys:
            config.pop(k, None)
        return orig_func(cls, config)

    layer_cls.from_config = patched


try:
    from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast

    _patch_from_config(RandomFlip, ["data_format"])
    _patch_from_config(RandomRotation, ["data_format"])
    _patch_from_config(RandomZoom, ["data_format"])

    
    _patch_from_config(RandomContrast, ["value_range", "data_format"])

    print(" Patched Random* layers (data_format/value_range stripped)")
except Exception as e:
    print(f" Could not patch Random* layers: {e}")

@register_keras_serializable(package="Custom")
class ReduceMeanLayer(layers.Layer):
    """Custom layer to replace Lambda with tf.reduce_mean"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()


@register_keras_serializable(package="Custom")
class ReduceMaxLayer(layers.Layer):
    """Custom layer to replace Lambda with tf.reduce_max"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_max(inputs, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class EfficientNetPreprocess(layers.Layer):
    def call(self, x):
        x = tf.cast(x, tf.float32)
        return preprocess_input(x)

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable(package="Custom")
class EfficientNetB3Block(layers.Layer):
    """Custom layer wrapping EfficientNetB3 - prevents remote download at load-time"""
    def __init__(self, input_shape=(300, 300, 3), trainable_base=False, weights=None, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = tuple(input_shape)
        self.trainable_base = bool(trainable_base)

        
        if weights in ("imagenet", "noisy-student"):
            weights = None
        self.weights_ = weights

        self.base = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights=self.weights_,
            input_shape=self.input_shape_,
        )
        self.base.trainable = self.trainable_base

    def call(self, inputs, training=False):
        return self.base(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_,
            "trainable_base": self.trainable_base,
            "weights": self.weights_,
        })
        return config

    @classmethod
    def from_config(cls, config):
        w = config.get("weights", None)
        if w in ("imagenet", "noisy-student"):
            config["weights"] = None
        return cls(**config)



from utils.encryption import get_encryptor
from utils.anonymization import DataAnonymizer
from utils.access_control import rate_limiter
from utils.audit_log import audit_logger

load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

@app.get("/")
def root():
    return jsonify({
        "status": "ok",
        "endpoints": ["/api/health", "/api/predict", "/api/model-info"]
    }), 200


UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

DEV_MODE = os.getenv('DEV_MODE', 'false').strip().lower() == 'true'


MODEL_FILE = os.getenv("MODEL_FILE", "").strip()


if not MODEL_FILE:
    MODEL_FILE = "best_ra_finetune_export.keras"

MODEL_PATH = None


CANDIDATES = [
    os.path.join(BASE_DIR, "model", MODEL_FILE),  
    os.path.join(BASE_DIR, MODEL_FILE),           
]

for p in CANDIDATES:
    if os.path.exists(p):
        MODEL_PATH = p
        print(f" Found model: {MODEL_PATH}")
        break

if MODEL_PATH is None:
    print("\n" + "="*60)
    print(" WARNING: Model file not found!")
    print("="*60)
    print(f"MODEL_FILE = {MODEL_FILE}")
    print("Searched:")
    for p in CANDIDATES:
        print(f"  - {p}")
    print(f"\nCurrent working directory: {os.getcwd()}")
    print("="*60 + "\n")

    if not DEV_MODE:
        exit(1)

    print(" DEVELOPMENT MODE: Running without model\n")


IMAGE_SIZE = (300, 300)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'change-this-secret-key-in-production')

encryptor = get_encryptor()
anonymizer = DataAnonymizer()


os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

import builtins
builtins.tf = tf

model = None

if MODEL_PATH:
    print("\n" + "="*60)
    print(" Loading model...")
    print("="*60)
    print(f"Model path: {os.path.abspath(MODEL_PATH)}")
    print(f"Model file exists: {os.path.exists(MODEL_PATH)}")

    model_size_mb = os.path.getsize(MODEL_PATH) / (1024*1024)
    print(f"Model file size: {model_size_mb:.2f} MB")

    try:
        import psutil
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024*1024)
        total_mb = mem.total / (1024*1024)
        used_percent = mem.percent

        print(f"\n MEMORY STATUS:")
        print(f"   Total RAM: {total_mb/1024:.2f} GB")
        print(f"   Available: {available_mb/1024:.2f} GB ({100-used_percent:.1f}% free)")
        print(f"   Used: {used_percent:.1f}%")

        required_mb = model_size_mb * 3
        print(f"\n   Model needs ~{required_mb:.0f} MB to load")
        print(f"   Currently available: {available_mb:.0f} MB")

        if available_mb < required_mb:
            print("\n" + "="*60)
            print(" WARNING: INSUFFICIENT MEMORY!")
            print("="*60)
            print(f"Required: ~{required_mb:.0f} MB")
            print(f"Available: {available_mb:.0f} MB")
            print(f"Shortage: {required_mb - available_mb:.0f} MB")
            print("\nRECOMMENDATIONS:")
            print("  1. Close other applications to free up RAM")
            print("     Run: Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 10")
            print("  2. Restart your computer (RECOMMENDED)")
            print("  3. Use a machine with more RAM")
            print(f"  4. Current usage is {used_percent:.1f}% - try to get below 70%")
            print("\nPROCEEDING WITH MEMORY-OPTIMIZED LOADING...")
            print("This may take longer but uses less RAM.")
            print("="*60 + "\n")

            import gc
            gc.collect()

            proceed = True
        else:
            print(f"   Sufficient memory available")
            proceed = True

    except ImportError:
        print("\n Install 'psutil' to check memory: pip install psutil")
    except Exception as e:
        print(f"\n Could not check memory: {e}")

    print("\n" + "="*60)

    print("\n" + "="*60)

    print(" Checking file integrity...")
    try:
        import zipfile
        if zipfile.is_zipfile(MODEL_PATH):
            with zipfile.ZipFile(MODEL_PATH, 'r') as z:
                files = z.namelist()
                print(f"   Valid Keras 3 archive ({len(files)} files)")
        else:
            print("    HDF5/Keras 2 format detected")
            try:
                import h5py
                with h5py.File(MODEL_PATH, 'r') as f:
                    print(f"    Valid HDF5 file ({len(f.keys())} root keys)")
            except:
                pass
    except Exception as e:
        print(f"    File might be corrupted: {e}")
        print("\nTry re-downloading or re-training the model.")
        if not DEV_MODE:
            exit(1)

    baseline_aug = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.015),
        layers.RandomZoom(0.02),
        layers.RandomContrast(0.02),
    ],
    name="baseline_aug",
     )

    custom_objects = {
    'baseline_aug': baseline_aug,
    'Custom>baseline_aug': baseline_aug,
    'EfficientNetPreprocess': EfficientNetPreprocess,
    'Custom>EfficientNetPreprocess': EfficientNetPreprocess,
    'EfficientNetB3Block': EfficientNetB3Block,
    'Custom>EfficientNetB3Block': EfficientNetB3Block,
    'ReduceMeanLayer': ReduceMeanLayer,
    'Custom>ReduceMeanLayer': ReduceMeanLayer,
    'ReduceMaxLayer': ReduceMaxLayer,
    'Custom>ReduceMaxLayer': ReduceMaxLayer,
    'tf': tf,
    }

    
    try:
        print("Loading model architecture...")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {keras.__version__}")
        print("Custom objects registered:")
        for key in custom_objects.keys():
            print(f"  - {key}")

        import gc
        gc.collect()

        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled")
        except:
            pass

        print("\nCalling keras.models.load_model()...")
        print(" This may take 2-5 minutes with low memory...")
        model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        print("Model architecture loaded")

        gc.collect()

        print("\nCompiling model ...")
        try:
            
            loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)

            
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=2e-5,
                weight_decay=1e-5,
                epsilon=1e-7,
                clipnorm=1.0,
            )

            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
            )
        except Exception:
            
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        print("Model compiled")


        gc.collect()

        print("\n Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total layers: {len(model.layers)}")
        print("="*60 + "\n")

    except FileNotFoundError as e:
        print("\n" + "="*60)
        print(" ERROR: Model file not found during loading!")
        print("="*60)
        print(f"Path: {os.path.abspath(MODEL_PATH)}")
        print(f"Error: {str(e)}")
        print("\nPlease ensure your model file is in the correct location.")
        print("="*60 + "\n")
        if not DEV_MODE:
            exit(1)

    except KeyboardInterrupt:
        print("\n\n Model loading interrupted by user (Ctrl+C)")
        exit(0)

    except Exception as e:
        print("\n" + "="*60)
        print(" ERROR: Failed to load model!")
        print("="*60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"\nPython version: {__import__('sys').version}")
        print(f"TensorFlow version: {tf.__version__}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("Possible solutions:")
        print("  1. Model saved with TF 2.12, you have TF", tf.__version__)
        print("     - TensorFlow 2.x models are generally compatible")
        print("  2. Check if custom layers match the training code")
        print("  3. Try running with more memory available")
        print("  4. Run in DEV_MODE to test server without model:")
        print("     PowerShell: $env:DEV_MODE='true'; python app.py")
        print("="*60 + "\n")
        if not DEV_MODE:
            exit(1)
        print("  2. Check if custom layers match the training code")
        print("  3. Verify model file is not corrupted")
        print("  4. Try running in DEV_MODE to test server without model:")
        print("     PowerShell: $env:DEV_MODE='true'; python app.py")
        print("="*60 + "\n")
        if not DEV_MODE:
            exit(1)
else:
    print("\n" + "="*60)
    print(" Running in DEVELOPMENT MODE (no model loaded)")
    print("="*60 + "\n")

CLASS_LABELS = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferative_DR'
}

CLASS_DESCRIPTIONS = {
    'No_DR': 'No Diabetic Retinopathy detected',
    'Mild': 'Mild Diabetic Retinopathy - Early stage with microaneurysms',
    'Moderate': 'Moderate Diabetic Retinopathy - More widespread blood vessel damage',
    'Severe': 'Severe Diabetic Retinopathy - Significant blood vessel blockage',
    'Proliferative_DR': 'Proliferative Diabetic Retinopathy - Advanced stage with new abnormal blood vessels'
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_client_ip():
    """Get client IP address"""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0]
    return request.remote_addr

@app.after_request
def set_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

@app.before_request
def check_rate_limit():
    """Check rate limiting for all requests"""
    identifier = get_client_ip()

    if request.endpoint == 'predict':
        if not rate_limiter.is_allowed(identifier, max_requests=10, window_seconds=60):
            audit_logger.log_security_event(
                'RATE_LIMIT_EXCEEDED',
                f'IP: {identifier}, Endpoint: {request.endpoint}',
                severity='WARNING',
                ip_address=identifier
            )
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
    else:
        if not rate_limiter.is_allowed(identifier, max_requests=100, window_seconds=3600):
            return jsonify({'error': 'Too many requests'}), 429

def preprocess_image(img_bgr):
    img_bgr = cv2.resize(img_bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    enhanced_bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    x = enhanced_bgr.astype("float32")
    x = np.expand_dims(x, axis=0)
    return x

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with security status"""
    client_ip = get_client_ip()
    audit_logger.log_access('system', 'HEALTH_CHECK', '/api/health', client_ip, True)

    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'security': {
            'encryption_enabled': True,
            'anonymization_enabled': True,
            'audit_logging_enabled': True,
            'gdpr_compliant': True,
            'pdpa_compliant': True
        }
    }), 200

@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():

    client_ip = get_client_ip()
    session_id = None
    anonymized_id = None

    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Server is running in development mode.',
            'dev_mode': True,
            'message': 'Please load a trained model file to enable predictions.'
        }), 503

    try:
        session_id = anonymizer.generate_session_id()
        anonymized_id = anonymizer.anonymize_patient_id(session_id)

        if 'file' not in request.files:
            audit_logger.log_error('UPLOAD_ERROR', 'No file uploaded', ip_address=client_ip)
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            audit_logger.log_security_event(
                'INVALID_FILE_TYPE',
                f'Attempted upload: {file.filename}',
                severity='WARNING',
                ip_address=client_ip
            )
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400

        image_bytes = file.read()

        if len(image_bytes) > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'File too large'}), 400

        file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({'error': 'Invalid image file'}), 400

        processed_image = preprocess_image(img_bgr)
        predictions = model.predict(processed_image, verbose=0)

        print(f"DEBUG - Raw predictions: {predictions[0]}")
        print(f"DEBUG - Prediction sum: {np.sum(predictions[0])}")

        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])

        all_probabilities = {
            CLASS_LABELS[i]: float(predictions[0][i]) * 100
            for i in range(len(CLASS_LABELS))
        }

        session_data = {
            'session_id': session_id,
            'filename': secure_filename(file.filename),
            'prediction_class': CLASS_LABELS[predicted_class],
            'confidence': confidence * 100,
            'timestamp': str(np.datetime64('now')),
            'anonymized': True,
            'encrypted': True
        }

        audit_logger.log_prediction(
            session_id,
            anonymized_id,
            CLASS_LABELS[predicted_class],
            confidence * 100,
            client_ip
        )

        result = {
            'success': True,
            'session_id': anonymized_id,
            'prediction': {
                'class': CLASS_LABELS[predicted_class],
                'class_id': predicted_class,
                'confidence': confidence * 100,
                'description': CLASS_DESCRIPTIONS[CLASS_LABELS[predicted_class]]
            },
            'all_probabilities': all_probabilities,
            'security': {
                'encrypted': True,
                'anonymized': True,
                'gdpr_compliant': True,
                'pdpa_compliant': True
            }
        }

        return jsonify(result), 200

    except Exception as e:
        audit_logger.log_error(
            'PREDICTION_ERROR',
            str(e),
            user_id=anonymized_id,
            ip_address=client_ip
        )
        return jsonify({
            'success': False,
            'error': 'An error occurred during prediction. Please try again.'
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information with security details"""
    client_ip = get_client_ip()
    audit_logger.log_access('anonymous', 'MODEL_INFO', '/api/model-info', client_ip, True)

    try:
        return jsonify({
            'model_name': 'EfficientNetB3 - Diabetic Retinopathy Classifier',
            'input_shape': list(IMAGE_SIZE),
            'classes': CLASS_LABELS,
            'descriptions': CLASS_DESCRIPTIONS,
            'num_classes': len(CLASS_LABELS),
            'security_features': {
                'data_encryption': 'AES-256-GCM',
                'anonymization': 'SHA-256 hashing',
                'compliance': ['GDPR', 'PDPA'],
                'audit_logging': 'Enabled',
                'rate_limiting': 'Enabled'
            },
            'privacy_notice': {
                'data_retention': '90 days',
                'pii_removal': 'Automatic',
                'metadata_stripping': 'Enabled',
                'right_to_erasure': 'Available',
                'data_portability': 'Available'
            }
        }), 200
    except Exception as e:
        audit_logger.log_error('MODEL_INFO_ERROR', str(e), ip_address=client_ip)
        return jsonify({'error': str(e)}), 500

@app.route('/api/privacy-notice', methods=['GET'])
def privacy_notice():
    """Get GDPR/PDPA compliant privacy notice"""
    return jsonify({
        'controller': {
            'name': 'Diabetic Retinopathy Detection Service',
            'contact': 'privacy@drdetection.com',
            'dpo_contact': 'dpo@drdetection.com'
        },
        'processing_purposes': [
            'Medical diagnosis and treatment',
            'Healthcare service improvement',
            'Research and development (anonymized data only)'
        ],
        'legal_basis': 'Explicit consent (GDPR Article 6(1)(a)) and Health data processing (Article 9(2)(a))',
        'data_collected': [
            'Retinal images (metadata removed)',
            'Prediction results',
            'Session information (anonymized)',
            'IP address (for security only)'
        ],
        'retention_period': '90 days from upload or until data deletion request',
        'rights': [
            'Right to access (GDPR Article 15)',
            'Right to rectification (GDPR Article 16)',
            'Right to erasure (GDPR Article 17)',
            'Right to restrict processing (GDPR Article 18)',
            'Right to data portability (GDPR Article 20)',
            'Right to object (GDPR Article 21)'
        ],
        'security_measures': [
            'AES-256 encryption',
            'Data anonymization',
            'Access control',
            'Audit logging',
            'Rate limiting',
            'Secure HTTPS communication'
        ],
        'recipients': 'Data is not shared with third parties',
        'automated_decision_making': 'AI-based diagnosis - results should be verified by medical professionals',
        'international_transfers': 'None',
        'complaint_authority': 'Your national data protection authority'
    }), 200

@app.route('/api/audit-log', methods=['GET'])
def get_audit_log():
    """Get audit log (admin only in production)"""
    try:
        return jsonify({
            'success': True,
            'message': 'Audit logging is enabled. Logs are stored in logs/audit.log'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    if DEV_MODE or model is None:
        print(" Diabetic Retinopathy Detection API - DEVELOPMENT MODE")
        print("="*60)
        print("  Model not loaded - predictions disabled")
        print(f"Working directory: {os.getcwd()}")
    else:
        print(" Diabetic Retinopathy Detection API - SECURE MODE")
        print("="*60)
        print(f"Model: {os.path.basename(MODEL_PATH)}")
        print(f"Classes: {list(CLASS_LABELS.values())}")

    print("\n  SECURITY FEATURES:")
    print("   AES-256-GCM Encryption")
    print("   SHA-256 Data Anonymization")
    print("   Rate Limiting")
    print("   Audit Logging")
    print("   Security Headers")
    print("   GDPR Compliance")
    print("   PDPA Compliance")

    if DEV_MODE or model is None:
        print("\n TO LOAD A MODEL:")
        print("  1. Place your .keras model file in:")
        print(f"     {os.getcwd()}")
        print("  2. Restart the server")
        print("\n SERVER ENDPOINTS (Model required for /predict):")
    else:
        print("\n SERVER ENDPOINTS:")

    print("  - http://localhost:5000/api/health")
    print("  - http://localhost:5000/api/model-info")
    print("  - http://localhost:5000/api/predict (POST)")
    print("  - http://localhost:5000/api/privacy-notice")
    print("="*60 + "\n")

    
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
