# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly:

1. **Do not** create a public GitHub issue for the vulnerability
2. Email the details to: security@dil-platform.example.com (or create a private security advisory on GitHub)
3. Include as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 5 business days
- **Fix Timeline**: 
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Within 90 days

## Security Measures Implemented

- **Authentication**: API keys with bcrypt hashing (work factor 12), constant-time verification
- **Rate Limiting**: 60 requests/minute per API key (Redis-backed for distributed deployments)
- **TLS Enforcement**: HTTPS required in production (configurable for development)
- **Input Validation**: Pydantic v2 validation on all endpoints
- **Audit Logging**: All inference requests logged with request ID, API key prefix, latency, outcome
- **CORS**: Restricted to configured origins only
- **API Key Management**: Keys never stored in plaintext, only bcrypt hashes; keys shown once at creation
- **Audit Trail**: Request ID, API key prefix, endpoint, method, status code, latency, client IP, error codes

## Compliance Considerations

- **HIPAA Alignment**: No PHI stored beyond request lifetime; audit logs exclude patient content
- **GDPR**: Right to deletion implemented via API key revocation
- **Data Retention**: Inference logs retained per configurable policy; images not persisted

## Disclosure Policy

Once a fix is released, we will:
1. Publish a security advisory on GitHub
2. Update the changelog
3. Notify affected users via email (if applicable)
4. Credit the reporter (unless they request anonymity)