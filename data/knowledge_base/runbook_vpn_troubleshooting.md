# VPN Troubleshooting Runbook

## Symptom: VPN Connection Drops Intermittently

### Step 1: Check Client Version
Ensure the VPN client is updated to the latest version (v4.2+).
- Windows: Check via Settings > Apps > VPN Client
- macOS: Check via Applications > VPN Client > About

### Step 2: Network Diagnostics
1. Run `ping vpn-gateway.company.com` to verify connectivity
2. Check for packet loss: `tracert vpn-gateway.company.com`
3. Verify DNS resolution: `nslookup vpn-gateway.company.com`

### Step 3: Common Fixes
- **Split tunneling conflict**: Disable split tunneling in VPN settings
- **MTU mismatch**: Set MTU to 1400: `netsh interface ipv4 set subinterface "VPN" mtu=1400`
- **Firewall interference**: Add VPN client to firewall exceptions
- **Proxy conflict**: Disable proxy while connected to VPN

### Step 4: Escalation
If above steps don't resolve, collect:
- VPN client logs (export from client settings)
- Network trace (30 seconds during disconnect event)
- Escalate to Network Engineering team (Tier 2)

**Resolution time target: 30 minutes for Tier 1, 2 hours for Tier 2**
