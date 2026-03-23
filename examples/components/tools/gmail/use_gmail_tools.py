import os

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes.tools.gmail import (
    ArchiveEmail,
    CreateDraft,
    CreateDraftReply,
    ForwardEmail,
    MarkAsRead,
    ModifyEmailLabels,
    ModifyThreadLabels,
    ReplyToEmail,
    RetrieveEmailsById,
    SearchEmails,
    SearchThreads,
    SendDraft,
    SendEmail,
)


def get_connection() -> GoogleOAuth2:
    """Create a Gmail connection with optional custom credential paths."""
    return GoogleOAuth2(
        access_token=os.getenv("GOOGLE_OAUTH2_ACCESS_TOKEN"),
    )


def create_draft():
    """Create a draft email."""
    tool = CreateDraft(id="create-draft", connection=get_connection())
    result = tool.run(
        input_data={
            "to": "recipient@example.com",
            "subject": "Project Update: Q2 Progress",
            "body": "Hi there,\n\nHere's a quick update on our Q2 milestones. Let me know if you have any questions.",
            "attachments": ["path_to_file"],
        }
    )
    tool.close()
    print("Create Draft:", result.output)


def create_draft_reply():
    """Create a reply as a draft to an existing message."""
    tool = CreateDraftReply(id="create-draft-reply", connection=get_connection())
    result = tool.run(
        input_data={
            "message_id": "your_message_id_here",
            "body": "Hi,\n\nThanks for your message. I'll follow up shortly with more details.\n\nBest,\nYour Name",
        }
    )
    tool.close()
    print("Create Draft Reply:", result.output)


def send_draft():
    """Send an existing draft email."""
    tool = SendDraft(id="send-draft", connection=get_connection())
    result = tool.run(input_data={"draft_id": "your_draft_id_here"})
    tool.close()
    print("Send Draft:", result.output)


def archive_email():
    """Archive an email by its message ID."""
    tool = ArchiveEmail(id="archive-email", connection=get_connection())
    result = tool.run(input_data={"message_id": "your_message_id_here"})
    tool.close()
    print("Archive Email:", result.output)


def forward_email():
    """Forward an existing email to a different recipient."""
    tool = ForwardEmail(id="forward-email", connection=get_connection())
    result = tool.run(input_data={"message_id": "your_message_id_here", "to": "forward_recipient@example.com"})
    tool.close()
    print("Forward Email:", result.output)


def mark_as_read():
    """Mark a specific email as read."""
    tool = MarkAsRead(id="mark-as-read", connection=get_connection())
    result = tool.run(input_data={"message_id": "your_message_id_here"})
    tool.close()
    print("Mark As Read:", result.output)


def modify_email_labels():
    """Add or remove labels from a specific email."""
    tool = ModifyEmailLabels(id="modify-labels", connection=get_connection())
    result = tool.run(
        input_data={
            "message_id": "your_message_id_here",
            "add_label_ids": ["UNREAD"],
            "remove_label_ids": ["CATEGORY_PROMOTIONS"],
        }
    )
    tool.close()
    print("Modify Labels:", result.output)


def reply_to_email():
    """Reply to an existing email thread."""
    tool = ReplyToEmail(id="reply-email", connection=get_connection())
    result = tool.run(
        input_data={
            "message_id": "your_message_id_here",
            "body": "Hi again,\n\nJust following up as promised. Let me know your thoughts.\n\nCheers,\nYour Name",
        }
    )
    tool.close()
    print("Reply to Email:", result.output)


def search_emails():
    """Search for recent emails with optional filters."""
    tool = SearchEmails(id="search-emails", connection=get_connection())
    result = tool.run(input_data={"label_ids": ["INBOX"], "query": "project update", "max_results": 5})
    tool.close()
    print("Search Emails:", result.output)


def send_email():
    """Send a new email with subject and body."""
    tool = SendEmail(id="send-email", connection=get_connection())
    result = tool.run(
        input_data={
            "to": "recipient@example.com",
            "subject": "Team Sync: Meeting Agenda",
            "body": "Hello Team,\n\nPlease find attached the agenda for tomorrow’s sync.",
        }
    )
    tool.close()
    print("Send Email:", result.output)


def search_threads():
    tool = SearchThreads(id="search-threads", connection=get_connection())
    result = tool.run(input_data={"query": "project update", "max_results": 5})
    tool.close()
    print("Search Emails:", result.output)


def modify_thread_labels():
    tool = ModifyThreadLabels(id="modify-labels", connection=get_connection())
    result = tool.run(
        input_data={
            "thread_id": "196fc4053043bbb9",
            "add_label_ids": ["UNREAD"],
            "remove_label_ids": ["CATEGORY_PROMOTIONS"],
        }
    )
    tool.close()
    print("Modify Labels:", result.output)


def retrieve_emails_threads():
    tool = RetrieveEmailsById(id="retrieve-emails", connection=get_connection())
    result = tool.run(
        input_data={
            "message_ids": ["your_message_id_here"],
            "thread_ids": ["your_thread_id_here"],
        }
    )
    tool.close()
    print("Result:", result.output)


if __name__ == "__main__":
    create_draft()
    create_draft_reply()
    send_draft()
    archive_email()
    forward_email()
    mark_as_read()
    modify_email_labels()
    reply_to_email()
    search_emails()
    send_email()
    search_threads()
    modify_thread_labels()
    retrieve_emails_threads()
