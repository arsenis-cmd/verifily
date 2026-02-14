# Procurement FAQ

Common questions from enterprise procurement and security teams.

## Data Privacy & Security

### Q: Does Verifily see our raw data?

**A: No.** Verifily validates your data without storing or transmitting raw content.

How it works:
- Validation runs on your infrastructure
- Only metadata leaves your environment
- No data retention in Verifily systems
- Full audit trail of access

For SaaS deployments:
- Data processed in your cloud account
- End-to-end encryption
- Optional BYOK (Bring Your Own Key)

### Q: Where is data processed?

**A: Your choice.**

Options:
1. **On-Premises**: Run entirely in your data center
2. **Private Cloud**: Your AWS/GCP/Azure account
3. **VPC**: Isolated tenant in our infrastructure
4. **Hybrid**: Mix of on-prem and cloud

### Q: Is our data used to train AI models?

**A: No.** Your data is never used for:
- Training ML models
- Improving our service
- Creating aggregated insights
- Third-party sharing

## Compliance & Certifications

### Q: What compliance certifications does Verifily have?

**A:** Current and planned certifications:

| Certification | Status | ETA |
|---------------|--------|-----|
| SOC 2 Type II | In Progress | Q2 2025 |
| ISO 27001 | In Progress | Q3 2025 |
| GDPR Compliance | ✅ Complete | - |
| HIPAA | Planned | Q4 2025 |
| FedRAMP | Planned | 2026 |

### Q: Can we run a security assessment?

**A: Yes.** We provide:
- Security questionnaire responses
- Architecture documentation
- Penetration test results (annual)
- Source code review (enterprise)
- Custom security requirements

Contact: security@verifily.dev

### Q: How do you handle vulnerabilities?

**A:** Our security process:

1. **Discovery**: Automated scanning + bug bounty
2. **Assessment**: CVSS scoring within 24h
3. **Remediation**: Critical fixes within 72h
4. **Disclosure**: Coordinated disclosure for responsible findings
5. **Notification**: Customers notified of relevant issues

## Deployment & Operations

### Q: Can we run fully offline/air-gapped?

**A: Yes.** Verifily works completely offline.

Requirements:
- Download installer/packages
- No internet connectivity needed
- Local license validation
- All features functional

### Q: What are the infrastructure requirements?

**A:** Minimal footprint:

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| API Server | 1 core | 512MB | 1GB |
| Validator | 2 cores | 2GB | 10GB |
| Monitor | 1 core | 1GB | 50GB |

Scales horizontally for high throughput.

### Q: How is Verifily licensed?

**A:** Flexible licensing:

- **Per-Seat**: Individual user licenses
- **Per-Project**: Unlimited users per project
- **Enterprise**: Unlimited everything
- **Usage-Based**: Pay per validation run

Volume discounts available.

### Q: What is the pricing model?

**A:** Transparent, predictable pricing:

| Tier | Price | Includes |
|------|-------|----------|
| Starter | $99/mo | 1 project, 1K runs/mo |
| Professional | $499/mo | 10 projects, 10K runs/mo |
| Enterprise | Custom | Unlimited, SLA, support |

No hidden fees. No usage surprises.

## Integration & Support

### Q: Does it integrate with our CI/CD?

**A: Yes.** Native integrations for:

- GitHub Actions
- GitLab CI
- Azure DevOps
- Jenkins
- CircleCI
- Travis CI
- Custom webhooks

### Q: What APIs are available?

**A:** Full REST API + SDKs:

- **REST API**: OpenAPI 3.0 spec
- **Python SDK**: Sync and async
- **CLI**: Full-featured command line
- **Webhook**: Event-driven integrations

### Q: What support is included?

**A:** Tiered support:

| Tier | Response Time | Channels |
|------|---------------|----------|
| Community | Best effort | GitHub issues |
| Professional | 24h | Email, Slack |
| Enterprise | 4h | Email, Slack, phone |
| Premium | 1h | Dedicated Slack, phone |

### Q: Do you offer professional services?

**A: Yes.** Available services:

- **Implementation**: Setup and configuration
- **Integration**: Custom CI/CD integration
- **Training**: Team onboarding
- **Custom Contracts**: Domain-specific validation
- **Consulting**: ML pipeline optimization

## Technical Questions

### Q: Is Verifily deterministic?

**A: Yes.** Same input always produces same output.

- Fixed random seeds
- Deterministic algorithms
- Versioned contracts
- Reproducible validation runs

### Q: How do you handle scale?

**A:** Built for enterprise scale:

- **Throughput**: 10K+ validations/hour
- **Dataset Size**: Tested up to 1TB
- **Latency**: P99 < 200ms for API calls
- **Availability**: 99.9% SLA (enterprise)

### Q: What happens if Verifily goes down?

**A:** Designed for resilience:

- Stateless architecture
- No single point of failure
- Automatic failover
- Data stays in your environment

**Offline mode**: Validation continues even without connectivity.

### Q: Can we customize validation logic?

**A: Yes.** Multiple extension points:

- **Custom Contracts**: YAML-defined rules
- **Plugins**: Python extensions
- **Webhooks**: External validation services
- **Integrations**: Connect to your systems

## Contract & Legal

### Q: What is the minimum contract term?

**A:** Flexible terms:

- **Monthly**: Cancel anytime
- **Annual**: 2 months free
- **Multi-year**: Custom pricing
- **Pilot**: 30-day free trial

### Q: What is your SLA?

**A:** Enterprise SLA includes:

| Metric | Guarantee | Credit |
|--------|-----------|--------|
| Uptime | 99.9% | 10% monthly fee |
| API Latency | P99 < 500ms | 5% monthly fee |
| Support Response | 4 hours | 10% monthly fee |

### Q: Can we terminate at any time?

**A: Yes.**

- No long-term lock-in
- Data export on request
- Pro-rated refunds
- No termination fees

### Q: Who owns the data?

**A: You do.**

- Customer data remains customer property
- No rights to use data
- Full data portability
- Right to deletion

### Q: Where is Verifily based?

**A:**

- **HQ**: San Francisco, CA, USA
- **Legal Entity**: Verifily Inc.
- **Data Residency**: Configurable by region
- **Support**: 24/7 global coverage

## Contact

**Sales**: sales@verifily.dev  
**Security**: security@verifily.dev  
**Support**: support@verifily.dev  
**Legal**: legal@verifily.dev

**Phone**: +1 (555) 123-4567  
**Address**: 123 ML Street, San Francisco, CA 94102
