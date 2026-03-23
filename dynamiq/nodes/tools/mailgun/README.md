## Description

The **MailGunTool** is a Python utility for sending emails via the [Mailgun API](https://www.mailgun.com/). This tool simplifies email delivery by allowing users to send plain text or HTML-formatted emails, including support for CC, BCC, and custom tags. Built with extensibility in mind, this tool leverages `dynamiq` for seamless integration into workflows requiring email communication.

## Example of Usage

Here’s how to set up and use the `MailGunTool` in your Python project:

```python
from dynamiq.connections import HttpApiKey
from dynamiq.nodes.tools.mailgun import MailGunTool, MailGunInputSchema

# Initialize the Mailgun connection
connection = HttpApiKey(
    api_key="your-mailgun-api-key",
    url="https://api.mailgun.net/v3"
)

# Create an instance of the tool
mailgun_tool = MailGunTool(
    connection=connection,
    domain_name="your-domain.com"
)

# Define the email details
input_data = MailGunInputSchema(
    from_email="sender@yourdomain.com",
    to_emails=["recipient@example.com"],
    subject="Test Email",
    text="This is a test email",
    html="<h1>This is a test email</h1>",
    tags=["test", "example"]
)

# Execute the tool and send the email
result = mailgun_tool.run(input_data)
print(result)
```

## Parameters

### Input Schema (`MailGunInputSchema`)

| **Parameter** | **Type**         | **Description**                                                                                          | **Required** | **Default** |
|----------------|------------------|----------------------------------------------------------------------------------------------------------|--------------|-------------|
| `from_email`   | `str`           | Sender's email address. Must belong to the Mailgun-verified domain.                                       | Yes          | None        |
| `to_emails`    | `str` or `list[str]` | Recipient(s) email address(es). Can be a single email or a list.                                          | Yes          | None        |
| `subject`      | `str`           | The email's subject line.                                                                                 | Yes          | None        |
| `text`         | `str` (optional)| Plain text version of the email body.                                                                     | No           | None        |
| `html`         | `str` (optional)| HTML version of the email body.                                                                           | No           | None        |
| `cc`           | `str` or `list[str]` (optional) | CC (carbon copy) recipient(s). Can be a single email or a list.                                              | No           | None        |
| `bcc`          | `str` or `list[str]` (optional) | BCC (blind carbon copy) recipient(s). Can be a single email or a list.                                      | No           | None        |

### MailGunTool Attributes

| **Attribute**   | **Type**           | **Description**                                                                                       | **Default**  |
|------------------|--------------------|-------------------------------------------------------------------------------------------------------|--------------|
| `name`          | `str`              | Name of the tool.                                                                                     | "Mailgun Email Sender" |
| `group`         | `NodeGroup`        | Specifies the group for this node. Always set to `NodeGroup.TOOLS`.                                   | `NodeGroup.TOOLS` |
| `connection`    | `HttpApiKey`       | Connection object to the Mailgun API, including API key and base URL.                                 | None         |
| `domain_name`   | `str`              | Your Mailgun-verified domain name.                                                                    | None         |
| `success_codes` | `list[int]`        | HTTP status codes indicating successful requests.                                                     | `[200]`      |
| `timeout`       | `float`            | Timeout for API requests in seconds.                                                                  | `30`         |

## Features

- **Flexible Input:** Supports both single and multiple recipients.
- **Content Options:** Send plain text or HTML-formatted emails.
- **Advanced Features:** Include CC, BCC, and custom tags for enhanced email functionality.
- **Error Handling:** Comprehensive exception handling for API errors and missing parameters.
