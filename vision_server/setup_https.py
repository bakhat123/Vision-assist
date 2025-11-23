#!/usr/bin/env python
"""
Generate self-signed SSL certificates for HTTPS support.
Run this once: python setup_https.py
"""

import os
import subprocess
import sys

cert_file = "cert.pem"
key_file = "key.pem"

if os.path.exists(cert_file) and os.path.exists(key_file):
    print(f"✓ Certificates already exist ({cert_file}, {key_file})")
    sys.exit(0)

print("Generating self-signed SSL certificate...")
try:
    # Generate a self-signed certificate valid for 365 days
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", key_file, "-out", cert_file,
        "-days", "365", "-nodes",
        "-subj", "/C=US/ST=State/L=City/O=Org/CN=10.7.33.86"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Certificate created: {cert_file}")
    print(f"✓ Key created: {key_file}")
    print("\nNow update main.py to use HTTPS, or run:")
    print("  uvicorn vision_server.main:app --host 0.0.0.0 --port 8443 --ssl-keyfile=vision_server/key.pem --ssl-certfile=vision_server/cert.pem")
except FileNotFoundError:
    print("ERROR: openssl not found. Install OpenSSL or use Python's built-in module.")
    print("Falling back to Python cryptography...")
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        from datetime import datetime, timedelta
        import ipaddress

        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # Build certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"State"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, u"City"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Org"),
            x509.NameAttribute(NameOID.COMMON_NAME, u"10.7.33.86"),
        ])

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.IPAddress(ipaddress.IPv4Address("10.7.33.86")),
                x509.DNSName(u"localhost"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256(), default_backend())

        # Write certificate
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        # Write key
        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        print(f"✓ Certificate created: {cert_file}")
        print(f"✓ Key created: {key_file}")
        print("\nNow update main.py to use HTTPS, or run:")
        print("  uvicorn vision_server.main:app --host 0.0.0.0 --port 8443 --ssl-keyfile=vision_server/key.pem --ssl-certfile=vision_server/cert.pem")
    except ImportError:
        print("ERROR: cryptography module not found.")
        print("Install it with: pip install cryptography")
        sys.exit(1)
