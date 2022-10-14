from __future__ import print_function
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.audio import MIMEAudio
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import base64, mimetypes
import pandas as pd
from logger import root


# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.compose', 'https://www.googleapis.com/auth/gmail.send']
SMTP_SERVER = "smtp.gmail.com"
PORT = 587  # For starttls
SENDER_EMAIL = "bormiopoli@gmail.com"
PASSWORD = "bormiopoli@gmail.com"


def generate_notification(gmail_service, dataframefun, bodies, i=0):
    message = f"Bi-LSTM 1h: EXECUTION {i}: This is the structure for the last strategy deployment:\n\n{bodies}"

    attachments=None
    if dataframefun is not None:
        attachments = []
        dataframefun.to_csv(f'{root}/all_quantile_delta_assets', sep='\t', mode='w')
        attachments.append(f'{root}/all_quantile_delta_assets')

    create_message_with_attachment(service=gmail_service, user_id='me', message_text=message,
                                   file=attachments)

    # create_message_with_attachment(service=gmail_service, user_id='me', message_text=message,
    #                                file=[#f'first_{TOP_N_STRATEGIES_INT}_strategies_and_reputation_considered',
    #                                      # f'{root}/standardised_strategies_values_weights',
    #                                      # f'{root}/strategies_weight.png',
    #                                      f'{root}/contributions.png'
    #                                      # , f'{root}/rebalanced_structure.png'
    #                                      , f'{root}/all_quantile_delta_assets'
    #                                ])


# def generate_message_best_strategies_update(dict_of_strategies_reputation):
#     zipped_tickers_and_reputations = zip(list(dict_of_strategies_reputation.keys()),
#                                          list(dict_of_strategies_reputation.values()))
#     string_tuple_to_write = [str(ticker) + "\t" + str(reputation) + "\n" for ticker, reputation in
#                              zipped_tickers_and_reputations]
#     message = f"""SHORT-TERM - ASSETS:
#     Some strategies have been updated.
#     Restarting the BOT
#     Strategies: Ticker\tREPUTATION {''.join(string_tuple_to_write)}"""
#     return message


# def generate_message_strategies_assets_weights(list_of_tuples):
#
#     zipped_tickers_and_reputations = zip(list_of_tuples[0].tolist(),
#                                          list_of_tuples[1].tolist(),
#                                          list_of_tuples[2].tolist(),
#                                          list_of_tuples[3].tolist(),
#                                          list_of_tuples[4].tolist())
#
#     string_tuple_to_write = [str(strategy) + "\t" + str(asset)+ "\t" + str(asset_weight) + "\t" + str(srs) + "\t" + str(aw) + "\n" for strategy, asset, asset_weight, srs, aw in
#                              zipped_tickers_and_reputations]
#     message = f"""SHORT-TERM - ASSETS:
#     Some strategies have been updated.
#     Restarting the BOT
#     STRATEGY\tASSET\tASSET_WEIGHT\tSTRATEGY_REPUTATION_SCORE\tASSET_WEIGHTED\n{''.join(string_tuple_to_write)}"""
#     return message


def gmail_authenticate(creds=None):

    if os.path.exists(f'{root}/token.json'):
        creds = Credentials.from_authorized_user_file(f'{root}/token.json', SCOPES)
        # creds = creds.with_subject("bormiopoli@gmail.com")

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                f'{root}/client_secret.json',
                SCOPES)
            creds = flow.run_local_server(port=0)
            # creds = creds.with_subject("bormiopoli@gmail.com")

        # Save the credentials for the next run
        with open(f'{root}/token.json', 'w') as token:
            token.write(creds.to_json())

    return creds


def authenticate_for_gmail_notifications():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    creds = gmail_authenticate(creds = None)
    service = None
    try:
        # Call the Gmail API
        service = build('gmail', 'v1', credentials=creds)
    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f'ST ASSETS - An error occurred authenticating into Gmail: {error}')

    return service


def generate_attachment_message_part(file, message):
    content_type, encoding = mimetypes.guess_type(file)

    if content_type is None or encoding is not None:
        content_type = 'application/octet-stream'
    main_type, sub_type = content_type.split('/', 1)
    if main_type == 'text':
        fp = open(file, 'rb')
        msg = MIMEText(fp.read(), _subtype=sub_type)
        fp.close()
    elif main_type == 'image':
        fp = open(file, 'rb')
        msg = MIMEImage(fp.read(), _subtype=sub_type)
        fp.close()
    elif main_type == 'audio':
        fp = open(file, 'rb')
        msg = MIMEAudio(fp.read(), _subtype=sub_type)
        fp.close()
    else:
        fp = open(file, 'rb')
        msg = MIMEBase(main_type, sub_type)
        msg.set_payload(fp.read())
        fp.close()
    filename = os.path.basename(file)
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    message.attach(msg)

    return message


def create_message_with_attachment(service, user_id,
    sender=SENDER_EMAIL, to=SENDER_EMAIL, subject="YoBOT \\|/ Update /|\\", message_text='', file=None):
  """Create a message for an email.

  Args:
    sender: Email address of the sender.
    to: Email address of the receiver.
    subject: The subject of the email message.
    message_text: The text of the email message.
    file: The path to the file to be attached.

  Returns:
    An object containing a base64url encoded email object.
  """
  if file is not None:
    message = MIMEMultipart()
    message.attach(MIMEText(message_text))
  else:
      message = MIMEText(message_text)

  message['to'] = SENDER_EMAIL
  message['from'] = "Pluto e le malefatte"
  message['subject'] = "Update of YoBOT"

  if file is not None:
      if type(file) == list:
          for el in file:
              message = generate_attachment_message_part(el, message)
      else:
        message = generate_attachment_message_part(file, message)

  send_message(service=service, user_id=user_id,
               message={'raw': base64.urlsafe_b64encode(message.as_string().encode('UTF-8')).decode('UTF-8')})


def send_message(service, user_id, message):
  """Send an email message.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    message: Message to be sent.

  Returns:
    Sent Message.
  """
  try:
    message = (service.users().messages().send(userId=user_id, body=message)
               .execute())
    print ('Message Id: {0}'.format(message['id']))
    return message
  except HttpError as error:
    print (f'ST ASSETS - An error occurred in sending message: {error}')


