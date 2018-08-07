import smtplib
from email.message import EmailMessage

msg = EmailMessage()
fromaddr = 'automail31715@gmail.com'
password = 'google10152018'
toaddr = 'wuling31715@gmail.com'
msg['Subject'] = 'IIS'
msg['From'] = 'wuling31715@gmail.com'
msg['To'] = 'automail31715@gmail.com'
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(fromaddr, password)
text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)
server.quit()
print('done.')
