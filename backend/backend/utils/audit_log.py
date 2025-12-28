import logging
from datetime import datetime
import json
import os

class AuditLogger:
    """Comprehensive audit logging for compliance"""
    
    def __init__(self, log_file='logs/audit.log'):
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # File handler
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            
            self.logger.addHandler(fh)
    
    def log_access(self, user_id, action, resource, ip_address, success=True):
        """Log data access"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'ACCESS',
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'ip_address': ip_address,
            'success': success
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_prediction(self, session_id, anonymized_id, prediction_class, confidence, ip_address):
        """Log prediction events"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'PREDICTION',
            'session_id': session_id,
            'anonymized_id': anonymized_id,
            'prediction_class': prediction_class,
            'confidence': confidence,
            'ip_address': ip_address
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_consent(self, user_id, consent_type, granted):
        """Log consent changes"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'CONSENT',
            'user_id': user_id,
            'consent_type': consent_type,
            'granted': granted
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_data_modification(self, user_id, action, data_type):
        """Log data modifications"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'MODIFICATION',
            'user_id': user_id,
            'action': action,
            'data_type': data_type
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_security_event(self, event_type, details, severity='INFO', ip_address=None):
        """Log security events"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'SECURITY',
            'event': event_type,
            'details': details,
            'severity': severity,
            'ip_address': ip_address
        }
        
        if severity == 'CRITICAL':
            self.logger.critical(json.dumps(log_entry))
        elif severity == 'WARNING':
            self.logger.warning(json.dumps(log_entry))
        else:
            self.logger.info(json.dumps(log_entry))
    
    def log_error(self, error_type, error_message, user_id=None, ip_address=None):
        """Log application errors"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'ERROR',
            'error_type': error_type,
            'message': error_message,
            'user_id': user_id,
            'ip_address': ip_address
        }
        self.logger.error(json.dumps(log_entry))

# Singleton instance
audit_logger = AuditLogger()
