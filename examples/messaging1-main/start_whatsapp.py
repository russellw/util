import subprocess

# The AppUserModelID for WhatsApp. Replace with the actual ID obtained from PowerShell if different.
app_user_model_id = '5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App'

# PowerShell command to launch WhatsApp
powershell_command = f'Start-Process shell:appsFolder\\{app_user_model_id}'

# Execute the PowerShell command
subprocess.run(['powershell', '-Command', powershell_command])
