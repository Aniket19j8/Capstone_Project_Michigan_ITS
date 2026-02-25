# Email / Outlook Troubleshooting Runbook

## Symptom: Outlook Not Syncing / Cannot Send or Receive

### Step 1: Quick Checks
1. Verify internet connectivity
2. Check Microsoft 365 service health: https://status.office365.com
3. Try Outlook Web App (OWA) - if OWA works, issue is client-side

### Step 2: Client-Side Fixes
- **Clear Outlook cache**: Close Outlook → Delete files in `%localappdata%\Microsoft\Outlook\RoamCache\`
- **Repair Office**: Settings → Apps → Microsoft 365 → Modify → Quick Repair
- **Recreate profile**: Control Panel → Mail → Show Profiles → Add new profile
- **Disable add-ins**: File → Options → Add-ins → Manage COM Add-ins → Uncheck all

### Step 3: Server-Side Checks
- Check mailbox size (quota: 50GB for standard, 100GB for executives)
- Verify mail flow rules aren't blocking
- Check transport queue for stuck messages
- Review message trace in Exchange admin center

### Step 4: Shared Mailbox Issues
- Verify user has Full Access and Send As permissions
- Remove and re-add the shared mailbox in Outlook
- Check auto-mapping is enabled in Exchange

**Resolution time target: 20 minutes for client issues, 1 hour for server issues**
