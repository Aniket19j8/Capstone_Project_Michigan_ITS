# Knowledge Base: Network Printer Setup and Common Issues

## Setup: Adding a Network Printer
1. Open Settings → Bluetooth & Devices → Printers & Scanners
2. Click "Add device" → Select printer from list
3. If not found: Click "Add manually" → Enter IP address (see printer label)
4. Install driver from Company Software Center if prompted

## Common Issues

### Print Jobs Stuck in Queue
1. Open Services (services.msc) → Stop "Print Spooler"
2. Delete files in `C:\Windows\System32\spool\PRINTERS\`
3. Restart Print Spooler service
4. Retry print job

### Printer Offline
1. Check physical connection (network cable, power)
2. Ping printer IP address
3. Remove and re-add printer
4. Check if print server is running (IT Operations can verify)

### Poor Print Quality
1. Run printer self-cleaning cycle (from printer menu)
2. Check toner/ink levels
3. Verify correct paper type is selected in print settings
4. If persistent: Submit maintenance request to Facilities

## Secure Print
All printers support secure print. Documents are held until user authenticates at the printer with their badge. Enable via: Print → Properties → Secure Print → Enter PIN.
