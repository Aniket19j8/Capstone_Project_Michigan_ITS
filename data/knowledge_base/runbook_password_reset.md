# Password Reset & Account Lockout Runbook

## Symptom: Account Locked Out

### Step 1: Verify Identity
Confirm user identity through:
- Employee ID verification
- Manager confirmation (if remote)
- Security questions (if configured)

### Step 2: Check Lockout Reason
1. Open Active Directory Users and Computers
2. Find user account → Properties → Account tab
3. Check "Account is locked out" checkbox
4. Review Event Viewer on domain controller for Event ID 4740

### Common Causes:
- **Multiple failed login attempts**: Usually 5+ failures within 30 minutes
- **Cached credentials**: Old password stored on mobile device or mapped drive
- **Service account**: Application using expired credentials
- **Brute force attempt**: Check source IP in security logs

### Step 3: Unlock Account
1. In AD: Right-click user → Unlock Account
2. If password expired: Reset password, require change at next login
3. If MFA issue: Reset MFA enrollment in Azure AD / Okta admin portal

### Step 4: Prevent Recurrence
- Have user update credentials on all devices
- Check for service accounts using their credentials
- Enable self-service password reset if not already active

**Resolution time target: 15 minutes**
