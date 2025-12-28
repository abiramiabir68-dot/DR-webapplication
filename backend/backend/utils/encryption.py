from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
import base64
import os

class AESEncryption:
    """AES-256 encryption/decryption utility"""
    
    def __init__(self, key=None):
        """
        Initialize AES encryption
        Args:
            key: Encryption key (32 bytes for AES-256)
        """
        if key is None:
            # Use environment variable or generate
            key = os.getenv('ENCRYPTION_KEY', 'default-key-change-in-production-32chars!')
        
        # Ensure key is 32 bytes for AES-256
        self.key = self._derive_key(key)
    
    def _derive_key(self, password):
        """Derive a 256-bit key from password using PBKDF2"""
        salt = b'diabetic_retinopathy_salt_2024'  # Store this securely
        if isinstance(password, str):
            password = password.encode('utf-8')
        return PBKDF2(password, salt, dkLen=32, count=100000)
    
    def encrypt(self, plaintext):
        """
        Encrypt plaintext using AES-256-GCM
        Args:
            plaintext: String or bytes to encrypt
        Returns:
            Base64 encoded encrypted data with nonce and tag
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        # Generate random nonce (12 bytes for GCM)
        nonce = get_random_bytes(12)
        
        # Create cipher
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
        
        # Encrypt and get authentication tag
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)
        
        # Combine nonce + tag + ciphertext
        encrypted_data = nonce + tag + ciphertext
        
        # Return base64 encoded
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def decrypt(self, encrypted_data):
        """
        Decrypt AES-256-GCM encrypted data
        Args:
            encrypted_data: Base64 encoded encrypted data
        Returns:
            Decrypted plaintext string
        """
        try:
            # Decode base64
            encrypted_data = base64.b64decode(encrypted_data)
            
            # Extract nonce, tag, and ciphertext
            nonce = encrypted_data[:12]
            tag = encrypted_data[12:28]
            ciphertext = encrypted_data[28:]
            
            # Create cipher
            cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
            
            # Decrypt and verify
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            
            return plaintext.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")
    
    def encrypt_file(self, file_bytes):
        """Encrypt file bytes (e.g., medical images)"""
        return self.encrypt(file_bytes)
    
    def decrypt_file(self, encrypted_data):
        """Decrypt file bytes"""
        return self.decrypt(encrypted_data)

# Singleton instance
_encryptor = None

def get_encryptor():
    """Get singleton encryption instance"""
    global _encryptor
    if _encryptor is None:
        _encryptor = AESEncryption()
    return _encryptor
