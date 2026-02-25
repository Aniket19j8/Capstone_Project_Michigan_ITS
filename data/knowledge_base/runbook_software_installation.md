# Software Installation & License Troubleshooting

## Standard Software Catalog
All approved software is available through the Company Software Center.

### Installation Failures
1. **Error: Insufficient permissions**: User needs local admin → Submit elevation request
2. **Error: Disk space**: Need minimum 2GB free → Help user clear temp files
3. **Error: Compatibility**: Check system requirements against user's hardware
4. **Error: Network timeout**: Switch to wired connection, retry during off-peak

### License Issues
- **Office 365**: Licenses managed through Azure AD groups. Add user to correct license group.
- **Adobe Creative Cloud**: Limited seats. Check with department admin for availability.
- **Specialized software (SAP, Salesforce)**: Requires manager approval + specific role assignment.

### Common License Errors
- **"Product activation failed"**: Run `ospp.vbs /act` from admin command prompt
- **"License limit reached"**: Deactivate on old device first, or contact vendor
- **"Subscription expired"**: Verify user's license assignment in admin portal

**Resolution time target: 30 minutes for standard software, 24 hours for specialized**
