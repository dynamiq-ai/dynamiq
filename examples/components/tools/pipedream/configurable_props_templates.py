JIRA_CREATE_ISSUE_PROPS = {
    "configurableProps": [
        {"name": "app", "type": "app", "app": "jira"},
        {
            "name": "cloudId",
            "type": "string",
            "label": "Cloud ID",
            "description": "The cloud ID.",
            "useQuery": True,
            "remoteOptions": True,
        },
        {
            "name": "historyMetadata",
            "type": "object",
            "label": "History Metadata",
            "description": "Additional issue history details.",
            "optional": True,
        },
        {
            "name": "properties",
            "type": "string",
            "label": "Properties",
            "description": "Details of issue properties to be add or update, please provide an array of objects "
            "with keys and values.",
            "optional": True,
        },
        {
            "name": "update",
            "type": "object",
            "label": "Update",
            "description": "A Map containing the field name and a list of operations to perform on the issue screen "
            "field. Note that fields included in here cannot be included in `fields`.",
            "optional": True,
        },
        {
            "name": "additionalProperties",
            "type": "object",
            "label": "Additional properties",
            "description": "Extra properties of any type may be provided to this object.",
            "optional": True,
        },
        {
            "name": "updateHistory",
            "type": "boolean",
            "label": "Update history",
            "description": "Whether the project in which the issue is created is added to the user's **Recently "
            "viewed** project list, as shown under **Projects** in Jira.",
            "optional": True,
        },
        {
            "name": "projectId",
            "type": "string",
            "label": "Project ID",
            "description": "The project ID.",
            "useQuery": True,
            "remoteOptions": True,
        },
        {
            "name": "issueTypeId",
            "type": "string",
            "label": "Issue Type",
            "description": "An ID identifying the type of issue, [Check the API docs]"
            "(https://developer.atlassian.com/cloud/jira/platform/rest/v3/#api-rest-api-3-issue-post) "
            "to see available options",
            "remoteOptions": True,
            "reloadProps": True,
        },
        {
            "name": "summary",
            "type": "string",
            "label": "Summary",
            "description": "Set your field value",
            "optional": False,
        },
        {
            "name": "issuerestriction",
            "type": "object",
            "label": "Restrict to",
            "description": "Set your field value",
            "optional": True,
        },
        {
            "name": "parent",
            "type": "string",
            "label": "Parent",
            "description": "Set your field value",
            "optional": True,
            "remoteOptions": True,
        },
        {
            "name": "description",
            "type": "string",
            "label": "Description",
            "description": "Set your field value",
            "optional": True,
        },
        {
            "name": "customfield_10020_gh-sprint",
            "type": "string[]",
            "label": "Sprint",
            "description": "Set your field value",
            "optional": True,
        },
        {
            "name": "customfield_10021_multicheckboxes",
            "type": "string[]",
            "label": "Flagged",
            "description": "Set your field value",
            "optional": True,
        },
        {
            "name": "customfield_10000_devsummarycf",
            "type": "object",
            "label": "development",
            "description": "Set your field value",
            "optional": True,
        },
        {
            "name": "customfield_10001_atlassian-team",
            "type": "object",
            "label": "Team",
            "description": "Set your field value",
            "optional": True,
        },
        {
            "name": "customfield_10034_vulnerabilitycf",
            "type": "object",
            "label": "Vulnerability",
            "description": "Set your field value",
            "optional": True,
        },
        {
            "name": "customfield_10035_designcf",
            "type": "string[]",
            "label": "Design",
            "description": "Set your field value",
            "optional": True,
        },
        {
            "name": "labels",
            "type": "string[]",
            "label": "Labels",
            "description": "Set your field value",
            "optional": True,
            "remoteOptions": True,
        },
        {
            "name": "customfield_10016_jsw-story-points",
            "type": "object",
            "label": "Story point estimate",
            "description": "Set your field value",
            "optional": True,
        },
        {
            "name": "customfield_10019_gh-lexo-rank",
            "type": "object",
            "label": "Rank",
            "description": "Set your field value",
            "optional": True,
        },
        {
            "name": "attachment",
            "type": "string[]",
            "label": "Attachment",
            "description": "Set your field value",
            "optional": True,
        },
        {
            "name": "issuelinks",
            "type": "string[]",
            "label": "Linked Issues",
            "description": "Set your field value",
            "optional": True,
        },
        {
            "name": "assignee",
            "type": "string",
            "label": "Assignee",
            "description": "Set your field value",
            "optional": True,
        },
    ]
}

GOOGLE_DRIVE_EXRACT_FILES_PROPS = {
    "configurable_props": [
        {
            "name": "googleDrive",
            "type": "app",
            "app": "google_drive",
        },
        {
            "name": "drive",
            "type": "string",
            "label": "Drive",
            "description": "Defaults to `My Drive`. To select a [Shared Drive]"
            "(https://support.google.com/a/users/answer/9310351) instead, select it from this list.",
            "optional": True,
            "default": "My Drive",
            "remoteOptions": True,
        },
        {
            "name": "folderId",
            "type": "string",
            "label": "Parent Folder",
            "description": "The ID of the parent folder which contains the file. If not specified, it will list files "
            "from the drive's top-level folder.",
            "remoteOptions": True,
            "optional": True,
        },
        {
            "name": "fields",
            "type": "string",
            "label": "Fields",
            "description": "The fields you want included in the response [(see the documentation for available fields)]"
            "(https://developers.google.com/drive/api/reference/rest/v3/files). If not specified, "
            "the response includes a default set of fields specific to this method. For development "
            "you can use the special value `*` to return all fields, but you'll achieve greater "
            "performance by only selecting the fields you need.\n\n**eg:** "
            "`files(id,mimeType,name,webContentLink,webViewLink)`",
            "optional": True,
        },
        {
            "name": "filterText",
            "label": "Filter Text",
            "description": "Filter by file name that contains a specific text",
            "type": "string",
            "optional": True,
            "reloadProps": True,
        },
        {
            "name": "trashed",
            "label": "Trashed",
            "type": "boolean",
            "description": "If `true`, list **only** trashed files. If `false`, list **only** non-trashed files. "
            "Keep it empty to include both.",
            "optional": True,
        },
    ],
}
