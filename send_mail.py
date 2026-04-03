import os
import smtplib
from email.message import EmailMessage


def send_photocopy_email(sender_email, sender_password, receiver_email):
    # Sends an email with Photocopy_Pdf.txt attached.
    file_path = "output_results/photocopy_pdf.txt"

    if not os.path.exists(file_path):
        print("⚠️ File not found:", file_path)
        return

    # Read file content (optional, just to check)
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read()

    msg = EmailMessage()
    msg['Subject'] = "YOLO Photocopy/PDF Detection Results"
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content(f"Hello,\n\nPlease review the attached Photocopy/PDF results.\n\nTotal entries: {len(lines.splitlines())}")

    # Attach the txt file
    with open(file_path, "rb") as f:
        file_data = f.read()
        msg.add_attachment(file_data, maintype="text", subtype="plain", filename="Photocopy_Pdf.txt")

    try:
        # Gmail SMTP server
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print("✅ Email sent successfully!")
    except Exception as e:
        print("❌ Failed to send email:", e)