import enum


class FileType(str, enum.Enum):
    IMAGE = "image"
    DOCUMENT = "document"
    DOCX_DOCUMENT = "docx_document"
    PDF = "pdf"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    PPTX_PRESENTATION = "pptx_presentation"
    ARCHIVE = "archive"
    AUDIO = "audio"
    VIDEO = "video"
    FONT = "font"
    EXECUTABLE = "executable"
    DATABASE = "database"
    EBOOK = "ebook"
    HTML = "html"
    TEXT = "text"
    MARKDOWN = "markdown"


EXTENSION_MAP = {
    FileType.IMAGE: {
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "webp",
        "tiff",
        "svg",
        "dwg",
        "xcf",
        "jpx",
        "apng",
        "cr2",
        "jxr",
        "psd",
        "ico",
        "heic",
        "avif",
    },
    FileType.DOCX_DOCUMENT: {"docx"},
    FileType.DOCUMENT: {"doc", "docx", "odt", "rtf"},
    FileType.PDF: {"pdf"},
    FileType.SPREADSHEET: {"xlsx", "csv", "tsv"},
    FileType.PRESENTATION: {"ppt", "pptx", "odp"},
    FileType.PPTX_PRESENTATION: {"pptx"},
    FileType.ARCHIVE: {
        "zip",
        "rar",
        "tar",
        "7z",
        "gz",
        "bz2",
        "xz",
        "lz",
        "lz4",
        "lzo",
        "zstd",
        "Z",
        "cab",
        "deb",
        "ar",
        "rpm",
        "br",
        "dcm",
        "ps",
        "crx",
    },
    FileType.AUDIO: {"mp3", "wav", "flac", "m4a", "aac", "ogg", "mid", "amr", "aiff"},
    FileType.VIDEO: {"mp4", "mkv", "avi", "mov", "wmv", "webm", "3gp", "m4v", "mpg", "flv"},
    FileType.FONT: {"woff", "woff2", "ttf", "otf"},
    FileType.EXECUTABLE: {"exe"},
    FileType.DATABASE: {"sqlite"},
    FileType.EBOOK: {"epub"},
    FileType.HTML: {"html"},
    # Plain-text data formats: read as text rather than handed back as raw bytes.
    FileType.TEXT: {
        "txt",
        "text",
        "log",
        "json",
        "jsonl",
        "ndjson",
        "yaml",
        "yml",
        "toml",
        "xml",
        "ini",
        "cfg",
        "conf",
    },
    FileType.MARKDOWN: {"md", "markdown"},
}
