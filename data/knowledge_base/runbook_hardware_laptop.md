# Laptop Hardware Troubleshooting Runbook

## Symptom: Laptop Not Turning On / Blue Screen / Overheating

### Not Turning On
1. Perform hard reset: Remove power, hold power button 15 seconds
2. Try with only AC power (remove battery if removable)
3. Check power adapter LED indicator
4. Try external monitor to rule out display failure
5. If no POST: Likely motherboard or RAM failure → Escalate to hardware depot

### Blue Screen of Death (BSOD)
1. Note the stop code (e.g., IRQL_NOT_LESS_OR_EQUAL, PAGE_FAULT_IN_NONPAGED_AREA)
2. Check Event Viewer → Windows Logs → System for critical errors
3. Common fixes by stop code:
   - **IRQL_NOT_LESS_OR_EQUAL**: Driver issue → Update/rollback recent drivers
   - **CRITICAL_PROCESS_DIED**: System file corruption → Run `sfc /scannow`
   - **MEMORY_MANAGEMENT**: RAM issue → Run Windows Memory Diagnostic
   - **KERNEL_DATA_INPAGE_ERROR**: Disk issue → Run `chkdsk /f /r`
4. If recurring: Check for recent Windows updates, driver changes

### Overheating
1. Check Task Manager for high CPU processes
2. Verify fans are running (listen/feel for airflow)
3. Clean vents with compressed air
4. Check thermal paste age (if >3 years, may need replacement)
5. Use cooling pad as temporary measure
6. BIOS update may improve thermal management

**Escalation**: Hardware depot for physical repairs. Ship via standard IT logistics process.
